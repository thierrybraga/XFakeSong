"""Diagnóstico pós-treino da série de épocas.

MOTIVAÇÃO 2026-08-09: o campo ``converged`` do artefato olha apenas AUC e
acurácia do **checkpoint selecionado**, então descreve se o artefato promovido
é utilizável — não como o treino chegou até ele. No run
``clean_benchmark_15k`` o Conformer divergiu na época ~14 e ficou em
``val_accuracy = 0.500`` da época 22 à 100; o checkpoint da época 10 sobreviveu
e o artefato saiu com ``converged: True``. Nada no JSON dizia que 90 das 100
épocas do orçamento declarado não produziram nada.

``CollapseAbort`` (``app/domain/models/training/trainer.py``) passou a abortar
esse padrão *durante* o treino, mas só protege execuções novas e não deixa
registro em quem já rodou. Este módulo aplica o MESMO critério à série completa
depois do fato, para que o artefato conte a história do treino.

Sem dependência de TensorFlow — o histórico entra como dict de listas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Mesmos defaults de ``CollapseAbort``, para que o veredito pós-hoc não possa
#: divergir da guarda que roda durante o treino.
DEFAULT_ARM_THRESHOLD = 0.6
DEFAULT_CHANCE_ACCURACY = 0.5
DEFAULT_TOLERANCE = 0.01
DEFAULT_COLLAPSE_PATIENCE = 15
DEFAULT_NAN_PATIENCE = 3


def _as_floats(values: Optional[Iterable[Any]]) -> List[float]:
    if values is None:
        return []
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _pick(history: Dict[str, Any], *names: str) -> List[float]:
    for name in names:
        if name in history:
            series = _as_floats(history.get(name))
            if series:
                return series
    return []


def _argmin_finite(values: Sequence[float]) -> Optional[int]:
    best_i: Optional[int] = None
    best_v = float("inf")
    for i, v in enumerate(values):
        if math.isfinite(v) and v < best_v:
            best_v, best_i = v, i
    return best_i


def _terminal_streak(flags: Sequence[bool]) -> int:
    """Comprimento da sequência de ``True`` que termina no último elemento."""
    n = 0
    for flag in reversed(flags):
        if not flag:
            break
        n += 1
    return n


def _longest_streak(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def analyze_training_stability(
    history: Optional[Dict[str, Any]],
    *,
    epochs_budget: Optional[int] = None,
    arm_threshold: float = DEFAULT_ARM_THRESHOLD,
    chance_accuracy: float = DEFAULT_CHANCE_ACCURACY,
    tolerance: float = DEFAULT_TOLERANCE,
    collapse_patience: int = DEFAULT_COLLAPSE_PATIENCE,
    nan_patience: int = DEFAULT_NAN_PATIENCE,
) -> Dict[str, Any]:
    """Classifica a série de épocas em estável / colapsada / divergida.

    ``status`` assume um de:

    ``stable``
        Nada anômalo: o treino nunca caiu ao nível do acaso depois de ter
        aprendido, e a perda de validação permaneceu finita.
    ``collapsed``
        O monitor caiu ao nível do acaso (``chance_accuracy + tolerance``)
        DEPOIS de ter passado de ``arm_threshold``, e ficou lá até a última
        época por pelo menos ``collapse_patience`` épocas. Foi o caso do
        Conformer. O checkpoint pode continuar bom — o que não vale é chamar de
        "100 épocas" um orçamento que produziu resultado só nas primeiras.
    ``recovered_collapse``
        Mesmo padrão, mas o treino voltou antes do fim.
    ``diverged_nonfinite``
        ``val_loss`` não-finito nas últimas ``nan_patience`` épocas.
    ``unknown``
        Sem histórico utilizável (modelos clássicos, por exemplo).

    ``stable`` (bool) é falso para ``collapsed`` e ``diverged_nonfinite``;
    ``recovered_collapse`` conta como estável mas fica registrado em
    ``warnings``.
    """
    history = history or {}
    if not isinstance(history, dict):
        return {"status": "unknown", "stable": None, "reason": "histórico ausente"}

    monitor = _pick(history, "val_accuracy", "val_acc", "accuracy", "acc")
    losses = _pick(history, "val_loss", "loss")
    epochs = max(len(monitor), len(losses))
    if epochs == 0:
        return {"status": "unknown", "stable": None, "reason": "histórico vazio"}

    warnings: List[str] = []
    result: Dict[str, Any] = {
        "status": "stable",
        "stable": True,
        "epochs_recorded": epochs,
        "monitor": "val_accuracy" if monitor else None,
        "criteria": {
            "arm_threshold": arm_threshold,
            "chance_accuracy": chance_accuracy,
            "tolerance": tolerance,
            "collapse_patience": collapse_patience,
            "nan_patience": nan_patience,
        },
    }

    if epochs_budget:
        result["epochs_budget"] = int(epochs_budget)
        if epochs < int(epochs_budget):
            # Não é defeito por si só (retomada, aborto de colapso, timeout);
            # é uma discrepância que o artefato precisa declarar em vez de
            # deixar `epochs` passar por orçamento cumprido.
            warnings.append(
                f"histórico com {epochs} épocas contra orçamento de "
                f"{int(epochs_budget)}"
            )

    # Época selecionada: menor val_loss finita — o mesmo critério do
    # `ResumableModelCheckpoint` (`checkpoint_selection`).
    best_i = _argmin_finite(losses)
    if best_i is not None:
        result["best_epoch"] = best_i + 1
        result["best_val_loss"] = round(losses[best_i], 6)
        result["epochs_after_best"] = epochs - (best_i + 1)
        result["best_epoch_fraction"] = round((best_i + 1) / epochs, 4)

    nonfinite = [i + 1 for i, v in enumerate(losses) if not math.isfinite(v)]
    if nonfinite:
        result["nonfinite_loss_epochs"] = nonfinite
        terminal_nan = _terminal_streak([not math.isfinite(v) for v in losses])
        if terminal_nan >= nan_patience:
            result.update(
                status="diverged_nonfinite",
                stable=False,
                reason=(
                    f"val_loss não-finito nas últimas {terminal_nan} épocas "
                    f"(a partir da época {epochs - terminal_nan + 1})"
                ),
            )
            result["warnings"] = warnings
            return result
        warnings.append(f"val_loss não-finito em {len(nonfinite)} época(s)")

    if not monitor:
        result["reason"] = "sem monitor de validação; só a perda foi checada"
        result["warnings"] = warnings
        return result

    # Só arma depois que o modelo demonstrou aprender — um início lento ou um
    # warmup longo nunca é confundido com colapso (mesma regra do CollapseAbort).
    dead: List[bool] = []
    armed = False
    peak = float("-inf")
    for value in monitor:
        if math.isfinite(value):
            peak = max(peak, value)
            if value >= arm_threshold:
                armed = True
            dead.append(armed and value <= chance_accuracy + tolerance)
        else:
            dead.append(False)

    terminal = _terminal_streak(dead)
    longest = _longest_streak(dead)
    if math.isfinite(peak):
        result["peak_monitor"] = round(peak, 6)

    if terminal >= collapse_patience:
        first = epochs - terminal + 1
        result.update(
            status="collapsed",
            stable=False,
            collapse_epoch=first,
            collapse_epochs=terminal,
            wasted_epoch_fraction=round(terminal / epochs, 4),
            reason=(
                f"monitor no nível do acaso da época {first} à {epochs} "
                f"({terminal} épocas) depois de ter chegado a {peak:.4f}"
            ),
        )
    elif longest >= collapse_patience:
        result.update(
            status="recovered_collapse",
            collapse_epochs=longest,
            reason=(
                f"queda ao nível do acaso por {longest} épocas seguidas, com "
                "recuperação antes do fim do orçamento"
            ),
        )
        warnings.append("houve colapso temporário durante o treino")

    result["warnings"] = warnings
    return result
