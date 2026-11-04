"""Callbacks de treino Keras: checkpoint retomável, histórico persistente,
guarda de colapso, ``val_eer`` por época e log de progresso.

Saíram de ``trainer.py`` (P2-15); o ``ModelTrainer`` os monta em
``_prepare_callbacks``/``train``.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from .metrics import validation_eer

_save_logger = logging.getLogger(__name__)
_progress_logger = logging.getLogger("training.progress")


def _process_rss_mb() -> Optional[float]:
    """RSS atual do processo em MB, via ``/proc/self/status`` (Linux/container).

    Diagnostico leve para o crescimento de RAM observado durante o treino em
    si (nao so no preparo dos dados, ja corrigido em `log_mel_batch` e
    `_prepare_protocol_splits`) — ver investigacao do SpectrogramTransformer
    2026-07-31. `None` fora de Linux (Windows/mac dev local): sem custo, so
    nao loga a linha.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


class ResumableModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` cujo melhor valor sobrevive a reinícios do treino.

    AJUSTE 2026-08-04 (queda de energia): o ``BackupAndRestore`` restaura
    pesos, otimizador e contador de épocas, mas NÃO o estado dos demais
    callbacks. Ao retomar, ``self.best`` volta a ``None`` e
    ``MonitorCallback._is_improvement(x, None)`` retorna ``True``
    incondicionalmente — a primeira época pós-retomada grava por cima do
    melhor checkpoint mesmo sendo pior. Com reinícios recorrentes isso
    degrada em silêncio o artefato que o protocolo declara em
    ``checkpoint_selection`` (``minimum_clean_validation_loss`` ou
    ``..._eer``, conforme ``config.checkpoint_monitor``).

    Persiste o melhor valor num arquivo ao lado do checkpoint (escrita
    atômica, para que uma queda no meio da gravação não corrompa o estado) e
    o devolve em ``on_train_begin``.
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self._best_state_path = Path(f"{filepath}.best.json")

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        # `best` já definido (ex.: initial_value_threshold explícito) manda.
        if self.best is not None or not self._best_state_path.exists():
            return
        try:
            state = json.loads(self._best_state_path.read_text(encoding="utf-8"))
            monitor = state["monitor"]
            best = float(state["best"])
        except (OSError, ValueError, KeyError, TypeError):
            _save_logger.warning(
                "[CKPT] estado de melhor valor ilegível em %s — retomando sem "
                "ele (a próxima época pode sobrescrever o melhor checkpoint)",
                self._best_state_path,
            )
            return
        if monitor != self.monitor:
            return
        # NaN como baseline TRAVA o checkpoint para sempre: `_is_improvement`
        # compara com `ops.less(x, nan)`, que é False para qualquer x, então
        # nenhuma época voltaria a gravar. Um treino que divergiu não deve
        # ditar o baseline do treino seguinte.
        if not np.isfinite(best):
            _save_logger.warning(
                "[CKPT] melhor %s persistido é %s (treino anterior divergiu) — "
                "ignorado; a seleção recomeça do zero",
                self.monitor,
                best,
            )
            return
        self.best = best
        _save_logger.warning(
            "[CKPT] melhor %s restaurado como %.6f — checkpoint só será "
            "substituído por um resultado efetivamente melhor",
            self.monitor,
            best,
        )

    def _save_model(self, epoch, batch, logs):
        # Uma época que divergiu não é "o melhor checkpoint": sem esta guarda,
        # `_is_improvement(nan, None)` retorna True na primeira época e o
        # artefato promovido nasce com pesos não-finitos.
        current = (logs or {}).get(self.monitor)
        if current is not None and not np.isfinite(current):
            _save_logger.warning(
                "[CKPT] época com %s=%s descartada para seleção de checkpoint",
                self.monitor,
                current,
            )
            return
        previous = self.best
        super()._save_model(epoch=epoch, batch=batch, logs=logs)
        if self.best is None or self.best == previous:
            return
        self._persist_best()

    def _persist_best(self) -> None:
        if not np.isfinite(self.best):
            return
        payload = json.dumps({"monitor": self.monitor, "best": float(self.best)})
        tmp_path = self._best_state_path.with_suffix(".json.tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self._best_state_path)
        except OSError as exc:
            _save_logger.warning(
                "[CKPT] falha ao persistir melhor %s: %s", self.monitor, exc
            )


class PersistentEpochHistory(tf.keras.callbacks.Callback):
    """Preserva o histórico COMPLETO de épocas através de retomadas.

    ``model.fit()`` devolve em ``history.history`` apenas as épocas DESTA
    execução. Com ``BackupAndRestore``, uma retomada na época 84 produz um
    histórico de 17 entradas para um treino de 100 — exatamente o que
    aconteceu no ``clean_benchmark_15k`` com RawNet2 (17/100) e RawGAT-ST
    (91/100): os dois treinaram as 100 épocas (está nos ``run.log``), mas o
    ``metrics.json`` só guardou o trecho pós-retomada, então as figuras de
    convergência mostram um fragmento e qualquer "melhor época" lida do
    artefato sai errada.

    Grava uma linha JSON por época, indexada pela época ABSOLUTA, e
    reconstrói a série inteira em :meth:`merged`.

    O arquivo fica FORA do ``backup_dir``: aquele diretório é apagado ao fim
    do treino (``delete_checkpoint=True``) e levaria o histórico junto.
    """

    def __init__(self, path, label: str = ""):
        super().__init__()
        self.path = Path(path)
        self.label = label or "training"
        self._records: dict[int, dict[str, float]] = {}

    def on_train_begin(self, logs=None):
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._records[int(record["epoch"])] = {
                    k: float(v) for k, v in record.items() if k != "epoch"
                }
        except (OSError, ValueError, KeyError, TypeError):
            _save_logger.warning(
                "[HIST] histórico persistido ilegível em %s — a série desta "
                "execução começa do zero",
                self.path,
            )
            self._records = {}
            return
        if self._records:
            _save_logger.warning(
                "[HIST] %s: %d épocas anteriores recuperadas de %s",
                self.label,
                len(self._records),
                self.path.name,
            )

    def on_epoch_end(self, epoch, logs=None):
        values: dict[str, float] = {}
        for key, value in (logs or {}).items():
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                continue
        # Índice absoluto: numa retomada o Keras devolve a época real (84), e
        # regravar a mesma chave torna a operação idempotente.
        self._records[int(epoch)] = values
        self._flush()

    def _flush(self) -> None:
        lines = [
            json.dumps({"epoch": epoch, **values}, ensure_ascii=False)
            for epoch, values in sorted(self._records.items())
        ]
        payload = "\n".join(lines) + "\n"
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError as exc:
            _save_logger.warning("[HIST] falha ao persistir histórico: %s", exc)

    def merged(self) -> dict[str, list[float]]:
        """Série completa no formato de ``History.history`` (listas por métrica)."""
        if not self._records:
            return {}
        ordered = [self._records[e] for e in sorted(self._records)]
        metrics: dict[str, list[float]] = {}
        for record in ordered:
            for key in record:
                metrics.setdefault(key, [])
        for record in ordered:
            for key, series in metrics.items():
                # Uma métrica ausente numa época (ex.: val_* numa época sem
                # validação) não pode deslocar as demais séries.
                series.append(record.get(key, float("nan")))
        return metrics


class CollapseAbort(tf.keras.callbacks.Callback):
    """Aborta um treino que degenerou para o palpite constante.

    MOTIVAÇÃO 2026-08-06: no run `clean_benchmark_15k` o Conformer divergiu na
    época ~14 e ficou em ``loss = ln 2 = 0.693`` / ``val_accuracy = 0.500`` da
    época 22 à 100 — 85 épocas (~50 min de GPU) produzindo nada, em duas
    sessões independentes. O melhor checkpoint era da época 10 e nunca mais
    seria superado.

    NÃO é early stopping, e não conflita com ``fixed_epoch_budget``: o early
    stopping interrompe um modelo que ainda melhora devagar. Esta guarda tem
    dois gatilhos, ambos exigindo evidência positiva de falha:

    1. **colapso** — o modelo JÁ ESTEVE bom (``arm_threshold``), caiu para o
       nível do acaso e ficou lá por ``patience`` épocas seguidas;
    2. **nunca generalizou** — passou ``arm_deadline`` sem cruzar
       ``arm_threshold`` *enquanto o treino abria* ``generalization_gap``
       de vantagem sobre a validação.

    O que nenhum dos dois faz é matar um modelo que ainda não começou: com
    treino e validação no acaso juntos, a guarda se cala e o orçamento fixo de
    épocas é quem limita. O melhor checkpoint é preservado — quem restaura é o
    ``ResumableModelCheckpoint``.

    O aborto fica registrado em ``self.triggered``/``self.reason`` para que o
    artefato diga o que aconteceu, em vez de parecer um treino curto qualquer.
    """

    def __init__(
        self,
        patience: int = 15,
        nan_patience: int = 3,
        chance_accuracy: float = 0.5,
        tolerance: float = 0.01,
        arm_threshold: float = 0.6,
        # Prazo para o modelo CRUZAR `arm_threshold` pela primeira vez. Sem
        # isto o guarda tinha um ponto cego: ele só arma DEPOIS de o modelo
        # ficar bom, então um treino que nunca aprende não é abortado nunca.
        #
        # Observado em 2026-08-16 no retune do RawGAT-ST (dropout 0,35->0,50):
        # `val_accuracy` ficou em 0,5000 exato da época 1 à 25 enquanto o
        # treino subia a 95,4% — 7,6 h de GPU sem nenhum aborto, porque o
        # guarda nunca chegou a armar.
        #
        # 15 é conservador por medida: no run `clean_benchmark_15k`, TODAS as
        # nove arquiteturas neurais cruzaram 0,6 até a época 3 (a mais lenta
        # foi justamente o RawGAT-ST). O prazo dá 5x essa folga.
        arm_deadline: int = 15,
        # Folga mínima treino-validação para o prazo acima poder abortar.
        #
        # SEM ESTA CONDIÇÃO O PRAZO É AMBÍGUO (corrigido em 2026-08-17). Olhando
        # só `val_accuracy`, "nunca aprendeu" e "warmup longo" produzem a MESMA
        # curva — acaso sustentado — e o prazo sozinho mataria os dois. O
        # segundo caso é justamente o que a guarda existe para NÃO fazer
        # (`test_collapse_abort_ignora_inicio_lento`): um modelo que fica no
        # acaso por 30 épocas e depois sobe a 0,97 é treino legítimo.
        #
        # O que separa os dois é o TREINO. No braço (d) do RawGAT-ST ele estava
        # em 0,9048 na época 15 com a validação em 0,5000 — folga de 40 pontos:
        # o modelo aprendeu o conjunto de ajuste e não generalizou nada. Num
        # warmup genuíno treino e validação estão no acaso JUNTOS, e aí a
        # guarda se cala: quem limita esse caso é o orçamento fixo de épocas.
        #
        # 0,20 fica bem acima da folga de qualquer run saudável do escopo
        # oficial na época 15 e bem abaixo dos 0,40 medidos no braço (d).
        generalization_gap: float = 0.20,
        monitor: str = "val_accuracy",
        loss_monitor: str = "val_loss",
        train_monitor: str | None = None,
        label: str = "",
    ):
        super().__init__()
        self.patience = max(1, int(patience))
        self.nan_patience = max(1, int(nan_patience))
        self.chance_accuracy = float(chance_accuracy)
        self.tolerance = float(tolerance)
        self.arm_threshold = float(arm_threshold)
        self.arm_deadline = max(1, int(arm_deadline))
        self.generalization_gap = float(generalization_gap)
        self.monitor = monitor
        self.loss_monitor = loss_monitor
        # Métrica de treino correspondente ao monitor: `val_accuracy` ->
        # `accuracy`. Derivar em vez de fixar mantém o par coerente se o
        # chamador monitorar outra métrica.
        self.train_monitor = train_monitor or (
            monitor[4:] if monitor.startswith("val_") else monitor
        )
        self.label = label or "training"
        self.triggered = False
        self.reason = ""
        self._armed = False
        self._best_acc = float("-inf")
        self._dead_streak = 0
        self._nan_streak = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        human_epoch = int(epoch) + 1

        loss = logs.get(self.loss_monitor)
        if loss is not None and not np.isfinite(loss):
            self._nan_streak += 1
            if self._nan_streak >= self.nan_patience:
                self._abort(
                    human_epoch,
                    f"{self.loss_monitor} não-finito por {self._nan_streak} "
                    "épocas seguidas",
                )
                return
        else:
            self._nan_streak = 0

        acc = logs.get(self.monitor)
        if acc is None:
            return
        acc = float(acc)
        self._best_acc = max(self._best_acc, acc)
        # Só arma depois que o modelo demonstrou aprender de fato — assim um
        # início lento (ou um warmup longo) nunca é confundido com colapso.
        if acc >= self.arm_threshold:
            self._armed = True

        # Nunca aprendeu: passou o prazo sem cruzar `arm_threshold` uma vez E
        # o treino já disparou na frente. É falha distinta do colapso (que
        # pressupõe ter estado bom antes) e precisa de aborto próprio — ver a
        # justificativa e o porquê da folga em __init__.
        if not self._armed and human_epoch >= self.arm_deadline:
            train_acc = logs.get(self.train_monitor)
            # Sem a métrica de treino não dá para distinguir memorização de
            # warmup longo. Na dúvida a guarda se cala: matar um treino bom
            # custa mais do que deixar um ruim correr até o fim do orçamento.
            if train_acc is not None and np.isfinite(train_acc):
                folga = float(train_acc) - acc
                if folga >= self.generalization_gap:
                    self._abort(
                        human_epoch,
                        f"{self.monitor} nunca alcançou "
                        f"{self.arm_threshold:.2f} em {human_epoch} épocas "
                        f"(melhor: {self._best_acc:.4f}) enquanto "
                        f"{self.train_monitor} chegou a {float(train_acc):.4f} "
                        f"— folga de {folga:.4f}: o modelo memoriza o treino e "
                        "não generaliza",
                    )
                    return

        if self._armed and acc <= self.chance_accuracy + self.tolerance:
            self._dead_streak += 1
            if self._dead_streak >= self.patience:
                self._abort(
                    human_epoch,
                    f"{self.monitor}={acc:.4f} (nível do acaso) por "
                    f"{self._dead_streak} épocas seguidas, depois de ter "
                    f"chegado a {self._best_acc:.4f}",
                )
        else:
            self._dead_streak = 0

    def _abort(self, epoch: int, reason: str) -> None:
        self.triggered = True
        self.reason = reason
        self.model.stop_training = True
        _progress_logger.warning(
            "[COLAPSO] %s: treino ABORTADO na época %d — %s. O melhor "
            "checkpoint anterior ao colapso foi preservado; revise o LR de "
            "pico/warmup antes de retreinar.",
            self.label,
            epoch,
            reason,
        )


class ValidationEER(tf.keras.callbacks.Callback):
    """Publica ``val_eer`` nos logs de época, para seleção de checkpoint.

    MOTIVO
    ------
    A seleção por ``val_loss`` (entropia cruzada) e a avaliação por EER medem
    coisas diferentes: a entropia é sensível à CALIBRAÇÃO, o EER mede apenas a
    ORDENAÇÃO das pontuações. Um detector pode piorar a entropia e melhorar o
    EER simplesmente ficando mais confiante nos acertos e nos erros.

    Esse descompasso não é hipotético neste projeto. No run publicado, o
    critério de menor ``val_loss`` escolhe para o RawGAT-ST a época 17
    (val_acc 84,7%) quando o pico foi 90,0% -- 5,3 pontos abaixo. E no retune
    com L2=3e-3 o mínimo de ``val_loss`` cai na ÉPOCA 1, onde o modelo ainda é
    desinformativo: perda desinformativa vale ln(2)=0,693, e um modelo que
    aprende mas erra com confiança nunca bate esse valor.

    O EER é a métrica primária das campanhas ASVspoof, que é o referencial do
    protocolo deste trabalho -- selecionar por ele alinha o critério de parada
    ao critério de avaliação.
    """

    def __init__(self, validation_data, label: str = "", batch_size: int = 32):
        super().__init__()
        self.validation_data = validation_data
        self.label = label or "model"
        # `predict` de LOTE INTEIRO estoura a VRAM nas arquiteturas de forma de
        # onda: a validação do benchmark são 1.456 janelas de 48.000 amostras
        # (266 MB) atravessando o grafo do RawGAT-ST numa RTX 3060 de 12 GB,
        # com a memória do treino já alocada. O smoke `_smoke_eer3` falhou
        # assim — e em silêncio, porque a exceção caía no `except` abaixo.
        self.batch_size = max(1, int(batch_size))
        self.falhas = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self.validation_data is None:
            return
        try:
            # Fonte ÚNICA do EER: `validation_eer` é a mesma função que a
            # restauração guardada de checkpoint chama. Enquanto o cálculo
            # vivia só aqui, o guard comparava OUTRA métrica (val_loss) e podia
            # descartar a época que este callback tinha eleito.
            logs["val_eer"] = validation_eer(
                self.model, self.validation_data, self.batch_size
            )
        except Exception as exc:  # noqa: BLE001
            # NÃO derruba um treino de horas — mas também NÃO cala.
            #
            # Este callback é auxiliar só quando o monitor é outro. Quando o
            # ModelCheckpoint está apontado para `val_eer`, uma falha aqui
            # significa que NENHUM checkpoint será salvo, e o log em DEBUG
            # escondia isso: o smoke `_smoke_eer3` treinou até o fim com o
            # aviso "Can save best model only with val_eer available" e
            # terminou sem `best.json`. WARNING, com contagem, para que a
            # próxima vez apareça no log do run.
            self.falhas += 1
            _save_logger.warning(
                "[%s] val_eer NÃO publicado na época %d (%d falha(s) "
                "seguidas): %s — se o checkpoint monitora val_eer, nenhuma "
                "época será salva",
                self.label,
                int(epoch) + 1,
                self.falhas,
                exc,
            )


class EpochProgressLogger(tf.keras.callbacks.Callback):
    """Loga progresso de treino em linha única por época ou intervalo."""

    def __init__(self, interval: int = 1, label: str = ""):
        super().__init__()
        self.interval = max(1, int(interval or 1))
        self.label = label or "model"
        self._started_at = 0.0
        self._epoch_started_at = 0.0
        self._epoch_index = 0
        self._last_batch_log_at = 0.0
        # Épocas concluídas NESTE processo. Após uma retomada via
        # `BackupAndRestore`, `epoch` volta com o índice absoluto (ex.: 84)
        # enquanto `_started_at` marca o restart — dividir o tempo decorrido
        # pelo índice absoluto subestimava o custo por época na mesma
        # proporção (84×), e o ETA saía perto de zero.
        self._epochs_this_run = 0
        self.batch_log_interval_s = max(
            0,
            int(os.getenv("XFAKE_TRAIN_BATCH_LOG_INTERVAL_S", "60") or "0"),
        )

    def on_train_begin(self, logs=None):
        self._started_at = time.time()
        total = self.params.get("epochs", "?")
        steps = self.params.get("steps", "?")
        rss = _process_rss_mb()
        _progress_logger.warning(
            "[TRAIN] %s iniciado: epochs=%s steps_per_epoch=%s%s",
            self.label,
            total,
            steps,
            f" rss_mb={rss:.0f}" if rss is not None else "",
        )

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_index = int(epoch) + 1
        self._epoch_started_at = time.time()
        self._last_batch_log_at = self._epoch_started_at

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_log_interval_s <= 0:
            return
        steps = self.params.get("steps")
        if not isinstance(steps, int) or steps <= 0:
            return
        now = time.time()
        current_batch = int(batch) + 1
        if (
            current_batch < steps
            and now - self._last_batch_log_at < self.batch_log_interval_s
        ):
            return
        self._last_batch_log_at = now
        elapsed = now - self._started_at
        epoch_elapsed = now - self._epoch_started_at
        pct = min(100.0, 100.0 * current_batch / max(1, steps))
        rss = _process_rss_mb()
        _progress_logger.warning(
            "[TRAIN] %s epoch=%s batch=%d/%d %.1f%% epoch_elapsed_min=%.1f elapsed_min=%.1f%s",
            self.label,
            self._epoch_index or "?",
            current_batch,
            steps,
            pct,
            epoch_elapsed / 60.0,
            elapsed / 60.0,
            f" rss_mb={rss:.0f}" if rss is not None else "",
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total = int(self.params.get("epochs") or 0)
        current = int(epoch) + 1
        self._epochs_this_run += 1
        should_log = (
            current == 1 or (total and current == total) or current % self.interval == 0
        )
        if not should_log:
            return

        elapsed = time.time() - self._started_at
        epoch_s = time.time() - self._epoch_started_at
        eta_min = None
        if total and current < total and self._epochs_this_run > 0:
            seconds_per_epoch = elapsed / self._epochs_this_run
            eta_min = seconds_per_epoch * (total - current) / 60.0
        metric_bits = []
        # `val_eer` entra aqui porque e a metrica que SELECIONA o
        # checkpoint quando `--checkpoint-monitor val_eer` esta ativo. Sem
        # ela na linha de log, uma bateria de dezenas de horas nao permite
        # verificar pelo log se a selecao esta funcionando: o
        # `ValidationEER` so escreve em `logs`, e so fala quando FALHA.
        for key in (
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
            "val_eer",
            "learning_rate",
        ):
            if key in logs:
                try:
                    metric_bits.append(f"{key}={float(logs[key]):.6g}")
                except Exception:
                    metric_bits.append(f"{key}={logs[key]}")
        _progress_logger.warning(
            "[TRAIN] %s epoch=%d/%s epoch_s=%.1f elapsed_min=%.1f eta_min=%s %s",
            self.label,
            current,
            total or "?",
            epoch_s,
            elapsed / 60.0,
            f"{eta_min:.1f}" if eta_min is not None else "-",
            " ".join(metric_bits),
        )
