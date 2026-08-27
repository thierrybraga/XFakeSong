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

#: MOTIVAÇÃO 2026-08-09 (auditoria por locutor do `clean_benchmark_15k`): o
#: critério de colapso acima só enxerga queda AO NÍVEL DO ACASO. O RawGAT-ST
#: nunca chega lá — ele oscila entre 0,62 e 0,90 da primeira à última época,
#: com o treino subindo monotonicamente até 0,998 — e saía do artefato como
#: ``stable``, indistinguível de um Conformer bem-comportado.
#:
#: Os limiares vêm das séries REAIS dos 9 modelos com histórico no run (um
#: teste trava esses valores contra elas), e exigem os DOIS sinais juntos:
#:
#:     modelo                  maior queda   dp últimas 50
#:     RawGAT-ST                    0,2232          0,0274   <- único a disparar
#:     AASIST                       0,1662          0,0109
#:     Hybrid CNN-Transformer       0,1360          0,0057
#:     MultiscaleCNN                0,0646          0,0016
#:     SpectrogramTransformer       0,0522          0,0032
#:     RawNet2                      0,0453          0,0155
#:     HuBERT / WavLM              <=0,0316        <=0,0062
#:
#: A conjunção importa: o Conformer colapsado tem queda de 0,2761 mas dp 0,0000
#: na cauda (fica travado em 0,5), e já é classificado como ``collapsed``. Uma
#: queda isolada seguida de recuperação também não basta sozinha.
DEFAULT_MAX_EPOCH_DROP = 0.20
DEFAULT_MONITOR_STD_TAIL = 0.02
DEFAULT_TAIL_EPOCHS = 50


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


def _argmax_finite(values: Sequence[float]) -> Optional[int]:
    best_i: Optional[int] = None
    best_v = float("-inf")
    for i, v in enumerate(values):
        if math.isfinite(v) and v > best_v:
            best_v, best_i = v, i
    return best_i


def _max_epoch_drop(values: Sequence[float]) -> Optional[float]:
    """Maior queda do monitor entre duas épocas CONSECUTIVAS.

    Mede amplitude de oscilação, não tendência: um treino que degrada devagar
    ao longo de 50 épocas tem quedas pequenas, enquanto um que salta de 0,85
    para 0,62 numa época só aparece aqui. Pares com valor não-finito são
    ignorados em vez de virarem ``nan`` e engolir a série inteira.
    """
    worst: Optional[float] = None
    for prev, cur in zip(values, values[1:]):
        if not (math.isfinite(prev) and math.isfinite(cur)):
            continue
        drop = prev - cur
        if worst is None or drop > worst:
            worst = drop
    return worst


def _tail_std(values: Sequence[float], tail: int) -> Optional[float]:
    """Desvio padrão populacional das últimas ``tail`` épocas finitas."""
    finite = [v for v in values[-tail:] if math.isfinite(v)]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    return (sum((v - mean) ** 2 for v in finite) / len(finite)) ** 0.5


def analyze_training_stability(
    history: Optional[Dict[str, Any]],
    *,
    epochs_budget: Optional[int] = None,
    arm_threshold: float = DEFAULT_ARM_THRESHOLD,
    chance_accuracy: float = DEFAULT_CHANCE_ACCURACY,
    tolerance: float = DEFAULT_TOLERANCE,
    collapse_patience: int = DEFAULT_COLLAPSE_PATIENCE,
    nan_patience: int = DEFAULT_NAN_PATIENCE,
    max_epoch_drop: float = DEFAULT_MAX_EPOCH_DROP,
    monitor_std_tail: float = DEFAULT_MONITOR_STD_TAIL,
    tail_epochs: int = DEFAULT_TAIL_EPOCHS,
    checkpoint_monitor: str = "val_loss",
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
    ``unstable_oscillation``
        O monitor balança sem caracterizar colapso: alguma queda entre épocas
        consecutivas ``>= max_epoch_drop`` E desvio na cauda
        ``>= monitor_std_tail``. Foi o caso do RawGAT-ST e do retreino do
        Conformer. O treino "funciona", só que a época escolhida vira sorteio —
        daí o ``selection_gap``.

        Pode haver quedas ao nível do acaso CURTAS (abaixo de
        ``collapse_patience``); elas são registradas em ``chance_level_epochs``
        / ``longest_chance_run`` e aparecem na ``reason``. A redação anterior
        afirmava "nunca cai ao acaso", o que era falso para o Conformer
        retreinado (4 épocas seguidas em 0,5000).
    ``unknown``
        Sem histórico utilizável (modelos clássicos, por exemplo).

    ``stable`` (bool) é falso para ``collapsed`` e ``diverged_nonfinite``;
    ``recovered_collapse`` e ``unstable_oscillation`` contam como estáveis mas
    ficam registrados em ``warnings``.

    Independente do status, o resultado traz a comparação entre a época que o
    protocolo SELECIONA e a que MAXIMIZA o monitor de colapso, em
    ``selection_gap``. Não muda a seleção — só deixa de escondê-la.

    ``checkpoint_monitor`` diz por qual métrica a época foi selecionada
    (``val_loss`` ou ``val_eer``) e define ``best_epoch``. Até 2026-08-22 este
    módulo assumia ``val_loss`` sempre, e os artefatos da bateria corrigida —
    que rodou com ``--checkpoint-monitor val_eer`` — declaravam como "melhor
    época" uma que não era a selecionada. ``best_epoch_by_val_loss`` e
    ``best_epoch_by_val_eer`` continuam disponíveis lado a lado: a distância
    entre as duas é o descompasso calibração×ordenação que motivou a troca de
    critério.

    O monitor de COLAPSO segue sendo a acurácia de validação, e é outra coisa:
    detecta o treino que morre, não a época que se publica.
    """
    history = history or {}
    if not isinstance(history, dict):
        return {"status": "unknown", "stable": None, "reason": "histórico ausente"}

    monitor = _pick(history, "val_accuracy", "val_acc", "accuracy", "acc")
    losses = _pick(history, "val_loss", "loss")
    eers = _pick(history, "val_eer")
    criterio = str(checkpoint_monitor or "val_loss").strip() or "val_loss"
    # Cai para val_loss quando o histórico não traz a série pedida — é o caso
    # dos artefatos antigos e do runner SSL, que seleciona por perda.
    if criterio == "val_eer" and not eers:
        criterio = "val_loss"
    selecao = eers if criterio == "val_eer" else losses
    epochs = max(len(monitor), len(losses), len(eers))
    if epochs == 0:
        return {"status": "unknown", "stable": None, "reason": "histórico vazio"}

    warnings: List[str] = []
    result: Dict[str, Any] = {
        "status": "stable",
        "stable": True,
        "epochs_recorded": epochs,
        "monitor": "val_accuracy" if monitor else None,
        "checkpoint_monitor": criterio,
        "criteria": {
            "arm_threshold": arm_threshold,
            "chance_accuracy": chance_accuracy,
            "tolerance": tolerance,
            "collapse_patience": collapse_patience,
            "nan_patience": nan_patience,
            "max_epoch_drop": max_epoch_drop,
            "monitor_std_tail": monitor_std_tail,
            "tail_epochs": tail_epochs,
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

    # Época selecionada: o mínimo finito da série que o `checkpoint_monitor`
    # aponta — o mesmo critério do `ResumableModelCheckpoint`
    # (`checkpoint_selection`). Ambas as séries ficam registradas.
    loss_i = _argmin_finite(losses)
    eer_i = _argmin_finite(eers)
    if loss_i is not None:
        result["best_epoch_by_val_loss"] = loss_i + 1
        result["best_val_loss"] = round(losses[loss_i], 6)
    if eer_i is not None:
        result["best_epoch_by_val_eer"] = eer_i + 1
        result["best_val_eer"] = round(eers[eer_i], 6)

    best_i = _argmin_finite(selecao)
    if best_i is not None:
        result["best_epoch"] = best_i + 1
        result["epochs_after_best"] = epochs - (best_i + 1)
        result["best_epoch_fraction"] = round((best_i + 1) / epochs, 4)
    if loss_i is not None and eer_i is not None and loss_i != eer_i:
        # O descompasso que motivou a troca de critério, medido no próprio
        # histórico: quantas épocas separam o mínimo da perda do mínimo do EER.
        result["selection_criteria_gap_epochs"] = abs(eer_i - loss_i)

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

    # As quedas ao nível do acaso são registradas SEMPRE, mesmo curtas demais
    # para armar um veredito de colapso (2026-08-11). Sem isto o retreino do
    # Conformer saía como `unstable_oscillation` com a justificativa "sem cair
    # ao nível do acaso" — e o histórico dele tem 5 épocas em 0,5000, sendo 4
    # consecutivas. O veredito estava certo (4 < collapse_patience); a
    # afirmação que o acompanhava, não. Quem lê o artefato precisa da série,
    # não só do rótulo.
    chance_epochs = [i + 1 for i, morto in enumerate(dead) if morto]
    if chance_epochs:
        result["chance_level_epochs"] = chance_epochs
        result["chance_level_epoch_count"] = len(chance_epochs)
        result["longest_chance_run"] = longest

    if math.isfinite(peak):
        result["peak_monitor"] = round(peak, 6)

    # Custo do critério de seleção. O protocolo escolhe por menor `val_loss`;
    # quando a `val_loss` é ruidosa, essa época pode não ser a que maximiza o
    # monitor. No RawGAT-ST a diferença é de 5,3 pp (época 8 contra 79).
    peak_i = _argmax_finite(monitor)
    if peak_i is not None:
        result["best_epoch_by_monitor"] = peak_i + 1
        if best_i is not None and best_i < len(monitor):
            selected = monitor[best_i]
            if math.isfinite(selected) and math.isfinite(monitor[peak_i]):
                result["monitor_at_selected_epoch"] = round(selected, 6)
                result["selection_gap"] = round(selected - monitor[peak_i], 6)

    drop = _max_epoch_drop(monitor)
    tail_std = _tail_std(monitor, tail_epochs)
    if drop is not None:
        result["max_epoch_drop"] = round(drop, 6)
    if tail_std is not None:
        result["monitor_std_tail"] = round(tail_std, 6)

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
    elif (
        drop is not None
        and tail_std is not None
        and drop >= max_epoch_drop
        and tail_std >= monitor_std_tail
    ):
        # Exige os dois sinais: a queda sozinha pega um tropeço isolado que se
        # recupera, e o desvio sozinho pega um treino que ainda está subindo.
        # A justificativa precisa refletir ESTE histórico. Dizer "sem cair ao
        # nível do acaso" quando houve quedas curtas é falso — e é justamente o
        # caso do retreino do Conformer (4 épocas seguidas em 0,5000, abaixo da
        # paciência de 15 que armaria `recovered_collapse`).
        if chance_epochs:
            queda_txt = (
                f"com {len(chance_epochs)} época(s) no nível do acaso "
                f"(maior sequência: {longest}, abaixo da paciência de "
                f"{collapse_patience} que caracterizaria colapso)"
            )
        else:
            queda_txt = "sem cair ao nível do acaso"
        result.update(
            status="unstable_oscillation",
            reason=(
                f"monitor oscilando: maior queda entre épocas de {drop:.4f} e "
                f"desvio de {tail_std:.4f} nas últimas "
                f"{min(tail_epochs, epochs)} épocas, {queda_txt}"
            ),
        )
        warnings.append(
            "treino instável (oscilação do monitor) — a época selecionada "
            "depende fortemente do ruído da val_loss"
        )

    result["warnings"] = warnings
    return result
