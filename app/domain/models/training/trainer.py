"""Módulo de Treinamento de Modelos

Este módulo implementa o treinador principal para modelos de detecção de deepfake.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    BackupAndRestore,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from app.core.config.settings import TrainingConfig
from app.core.contracts.audio import IModelTrainer
from app.core.contracts.base import ProcessingResult, ProcessingStatus
from app.core.performance import optimize_tf_dataset

from .augmentation import AudioAugmenter
from .metrics import MetricsCalculator
from .optimization import OptimizerFactory
from .secure_training_pipeline import SecureTrainingConfig, SecureTrainingPipeline

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
                "execução começa do zero", self.path
            )
            self._records = {}
            return
        if self._records:
            _save_logger.warning(
                "[HIST] %s: %d épocas anteriores recuperadas de %s",
                self.label, len(self._records), self.path.name,
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


def scores_from_predictions(bruto) -> "np.ndarray":
    """Converte a saída BRUTA do modelo na pontuação que ordena spoof>bonafide.

    SAÍDA DE 2 COLUNAS é o caso NORMAL no escopo, não a exceção: as
    arquiteturas emitem softmax/logits sobre {bonafide, spoof}. A coluna 1 é a
    do FAKE, a mesma convenção de `benchmarks/runner.py` (`pred[:, 1]`).

    LOG-ODDS, nao a coluna 1 sozinha. Medido em 2026-08-19 no AASIST promovido:
    ele emite LOGITS CRUS (faixa -37,8 a +41,1; a soma das colunas tem media
    1,91 e desvio 4,51, ou seja NAO e softmax), e as duas colunas variam de
    forma independente. Ordenar so por `logit[1]` deu EER 0,0025 contra 0,0050
    pelo criterio correto -- OTIMISTA por um fator de 2.

    A diferenca serve aos DOIS casos e dispensa detectar o tipo de saida: com
    softmax as colunas somam 1, entao `col1 - col0 = 2*col1 - 1` e
    transformacao monotonica de col1 e o EER nao muda; com logits ela e a
    log-odds, que e o que ordena corretamente.
    """
    bruto = np.asarray(bruto)
    if bruto.ndim == 2 and bruto.shape[1] == 2:
        return (bruto[:, 1] - bruto[:, 0]).ravel()
    return bruto.ravel()


def eer_from_scores(y_true, scores) -> float:
    """EER a partir de rótulos e pontuações, com INTERPOLAÇÃO do cruzamento.

    Espelha `MetricsCalculator.calculate_eer` (que usa brentq). Usar só o ponto
    mais próximo (`argmin|fpr-fnr|`) fazia as duas implementações discordarem
    na terceira casa — num regime de EER de 0,1% é essa a casa que separa os
    modelos, e é esta métrica que SELECIONA o checkpoint. A interpolação linear
    entre os dois pontos que cercam a troca de sinal dá o mesmo resultado do
    brentq sem depender de scipy dentro de um callback de época.

    Levanta ValueError quando o EER não é definível (uma classe só, tamanhos
    incompatíveis, pontuações não-finitas) — quem chama decide se isso é aviso
    ou falha.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    if scores.size != y_true.size:
        raise ValueError(
            f"predição com {scores.size} scores para {y_true.size} rótulos"
        )
    if not np.isfinite(scores).all():
        raise ValueError("pontuações não-finitas (NaN/inf) na validação")
    if len(np.unique(y_true)) < 2:
        raise ValueError("validação tem uma classe só; EER indefinível")

    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    finito = np.isfinite(thr)
    fpr, fnr = fpr[finito], fnr[finito]
    if fpr.size == 0:
        raise ValueError("curva ROC vazia após descartar limiares não-finitos")

    d = fpr - fnr
    troca = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if troca.size:
        i = int(troca[0])
        passo = d[i] - d[i + 1]
        peso = float(d[i] / passo) if passo != 0 else 0.0
        eer = float(
            (fpr[i] + peso * (fpr[i + 1] - fpr[i]))
            + (fnr[i] + peso * (fnr[i + 1] - fnr[i]))
        ) / 2.0
    else:
        i = int(np.nanargmin(np.abs(d)))
        eer = float((fpr[i] + fnr[i]) / 2.0)
    return float(min(max(eer, 0.0), 1.0))


def validation_eer(model, validation_data, batch_size: int = 32) -> float:
    """EER do modelo no conjunto de validação, na convenção do projeto.

    Fonte ÚNICA do `val_eer`: o callback `ValidationEER` (que publica a métrica
    por época) e a restauração guardada de checkpoint (que decide se o
    checkpoint selecionado por ela fica) chamam esta função. Duas
    implementações do mesmo EER divergindo na terceira casa selecionariam
    épocas diferentes das que o log declara.
    """
    x, y = validation_data
    bruto = model.predict(x, verbose=0, batch_size=max(1, int(batch_size)))
    return eer_from_scores(np.asarray(y).ravel(), scores_from_predictions(bruto))


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
        if current_batch < steps and now - self._last_batch_log_at < self.batch_log_interval_s:
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
            current == 1
            or (total and current == total)
            or current % self.interval == 0
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


def save_inference_keras(model: "tf.keras.Model", path) -> None:
    """Salva um artefato de INFERÊNCIA (.keras) SEM o estado do otimizador.

    No Keras 3, `include_optimizer=False` é IGNORADO para o formato `.keras`: o
    estado do Adam (2 momentos por peso, ~2× os pesos) é sempre serializado,
    deixando o arquivo ~3× maior que o necessário para inferência.

    Removemos o otimizador ANTES de salvar e o restauramos DEPOIS. O grafo e os
    pesos ficam idênticos — a saída do modelo NÃO muda (neutro em acurácia) —,
    apenas o estado de treino deixa de ser gravado. Ganho típico: 561 MB → 188 MB
    (MultiscaleCNN), com load proporcionalmente mais rápido.

    Obs.: reconstruir via from_config+set_weights foi descartado por alterar a
    saída (NaN em modelos com BatchNormalization/camadas custom).
    """
    path = str(path)
    saved_opt = getattr(model, "optimizer", None)
    try:
        try:
            model.optimizer = None
        except Exception as e:
            _save_logger.debug(f"Não foi possível remover o otimizador: {e}")
        model.save(path)
    finally:
        # Restaura o otimizador para não afetar usos posteriores (ex.: continuar
        # o treino, avaliar com o mesmo objeto de modelo).
        if saved_opt is not None:
            try:
                model.optimizer = saved_opt
            except Exception:
                pass


class ModelTrainer(IModelTrainer):
    """Implementação do treinador de modelos com prevenção de data leakage."""

    def __init__(
        self, config: TrainingConfig, use_mixed_precision: Optional[bool] = None
    ):
        """
        Args:
            config: configuração de treinamento
            use_mixed_precision: Sprint 3.2 — Se None (default), auto-detecta:
                habilita mixed_float16 se houver GPU com Compute Capability >= 7.0
                (Volta+, RTX 20xx+). Setar True/False para forçar.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator()
        self.optimizer_factory = OptimizerFactory()
        self.augmenter = AudioAugmenter(config.augmentation_config)

        # Sprint 3.2: Mixed precision (float16) auto-detect em GPU
        # 2× speedup + metade da VRAM em GPUs Tensor Core (CC >= 7.0).
        if use_mixed_precision is None:
            use_mixed_precision = self._should_enable_mixed_precision()

        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                self.logger.info(
                    "Mixed precision training habilitado (mixed_float16): "
                    "~2× speedup + ~50% menos VRAM"
                )
            except Exception as e:
                self.logger.warning(f"Mixed precision indisponível: {e}")
        else:
            try:
                tf.keras.mixed_precision.set_global_policy("float32")
                self.logger.info("Mixed precision desabilitado para este treino.")
            except Exception as e:
                self.logger.warning(f"Falha ao definir política float32: {e}")

        # Configurar pipeline seguro para prevenção de data leakage
        secure_config = SecureTrainingConfig(
            test_size=getattr(config, "test_size", 0.2),
            validation_size=getattr(config, "validation_split", 0.2),
            random_state=42,
            use_temporal_split=getattr(config, "use_temporal_split", True),
            scaler_type=getattr(config, "scaler_type", "standard"),
            save_scaler=True,
            validate_no_leakage=True,
        )
        self.secure_pipeline = SecureTrainingPipeline(secure_config)

    def _array_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        batch_size: int,
        purpose: str,
        shuffle: bool = False,
    ) -> tuple[tf.data.Dataset, bool]:
        """Cria dataset sem materializar arrays grandes como Tensor constante.

        AJUSTE 2026-07-14 (`shuffle`): o Keras IGNORA `shuffle=True` do fit()
        quando `x` é um tf.data.Dataset — o caminho sem augmentation treinava
        com ordem de batches FIXA em todas as épocas. No benchmark isso é
        agravado pelo protocolo AWGN: o treino é [bloco limpo | bloco ruidoso]
        concatenados, então cada época via primeiro só amostras limpas e
        depois só ruidosas. `shuffle=True` (usar apenas no treino) embaralha
        por época nos dois caminhos (tensor_slices e generator).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        bytes_total = int(X.nbytes + y.nbytes)
        large_threshold = 256 * 1024 * 1024

        if bytes_total <= large_threshold:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                dataset = dataset.shuffle(
                    buffer_size=len(y), seed=42, reshuffle_each_iteration=True
                )
            # prefetch(AUTOTUNE): sem isso, a GPU fica ociosa esperando o
            # próximo lote em vez de sobrepor preparo de dado (CPU) com
            # computo (GPU) — o padrão clássico de baixa utilização de GPU
            # (~30% observado no Conformer) mesmo com o modelo saudável.
            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), False

        self.logger.info(
            "Dataset %s grande (%.1f MB) — usando generator em batches para "
            "reduzir cópias de RAM.",
            purpose,
            bytes_total / (1024 * 1024),
        )

        x_shape = (None,) + tuple(X.shape[1:])
        y_shape = (None,) + tuple(y.shape[1:])
        x_dtype = tf.as_dtype(X.dtype)
        y_dtype = tf.as_dtype(y.dtype)
        rng = np.random.default_rng(42)

        def batch_generator():
            n = len(y)
            # Permutação nova a cada passagem (época): o estado do rng
            # persiste no closure entre reinvocações do generator (repeat()).
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                yield X[idx], y[idx]

        n_batches = int(np.ceil(len(y) / max(1, batch_size)))
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x_dtype),
                tf.TensorSpec(shape=y_shape, dtype=y_dtype),
            ),
        )
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(n_batches))
        # Idem ao caminho from_tensor_slices: sem prefetch, o generator
        # Python (single-threaded, GIL) monta cada lote de forma síncrona e
        # bloqueia a GPU entre lotes. Com AUTOTUNE, o próximo lote é montado
        # numa thread em segundo plano enquanto a GPU processa o atual —
        # este é justamente o caminho usado pelos datasets grandes
        # (>256 MB: RawNet2/AASIST/RawGAT-ST/Conformer/etc. com a cópia
        # AWGN), onde o ganho de sobreposição CPU/GPU é maior.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, True

    def train(
        self,
        model: tf.keras.Model,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> ProcessingResult[Dict[str, Any]]:
        """Treina modelo com pipeline seguro para prevenção de data leakage."""
        try:
            self.logger.info("Iniciando treinamento seguro do modelo")

            X_train, y_train = train_data

            # Usar pipeline seguro para preparar dados se validation_data não
            # fornecida
            if validation_data is None:
                self.logger.info(
                    "Aplicando pipeline seguro para divisão e normalização dos dados"
                )

                # Preparar dados usando pipeline seguro
                preparation_result = self.secure_pipeline.prepare_data(
                    X_train, y_train, metadata
                )

                if preparation_result.status != ProcessingStatus.SUCCESS:
                    raise ValueError(
                        f"Erro na preparação segura dos dados: {preparation_result.errors}"
                    )

                prepared_data = preparation_result.data
                X_train = prepared_data["X_train"]
                X_val = prepared_data["X_val"]
                y_train = prepared_data["y_train"]
                y_val = prepared_data["y_val"]

                # Armazenar dados de teste para avaliação posterior
                self._test_data = (prepared_data["X_test"], prepared_data["y_test"])

                validation_data = (X_val, y_val)

                self.logger.info(
                    f"Dados preparados com segurança - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(self._test_data[0])}"
                )
            else:
                self.logger.warning(
                    "Dados de validação fornecidos externamente - pipeline seguro não aplicado"
                )

            # ── Compile-respect ─────────────────────────────────────────────
            # Cada arquitetura compila a si mesma com loss/otimizador corretos
            # (ex.: AASIST = AM-Softmax logits + loss from_logits=True + AdamW
            # clipnorm; Conformer/CCT/AST = WarmupCosineDecay; WavLM/HuBERT =
            # LR baixo p/ fine-tune SSL). Recompilar aqui DESCARTAVA tudo isso e
            # — pior — aplicava a string "sparse_categorical_crossentropy"
            # (from_logits=False) sobre logits crus do AASIST, quebrando o
            # treino. Agora: modelo já compilado é RESPEITADO; só ajustamos o
            # LR quando o chamador o definiu explicitamente (lr_is_explicit).
            already_compiled = getattr(model, "optimizer", None) is not None
            if already_compiled:
                self.logger.info(
                    "Modelo já compilado pela arquitetura — preservando "
                    "loss/otimizador originais."
                )
                if getattr(self.config, "lr_is_explicit", False):
                    try:
                        model.optimizer.learning_rate = self.config.learning_rate
                        self.logger.info(
                            f"LR explícito aplicado: {self.config.learning_rate}"
                        )
                    except Exception as e:
                        # LR pode ser um schedule (WarmupCosineDecay) — nesse
                        # caso o schedule da arquitetura prevalece.
                        self.logger.warning(
                            f"LR explícito ignorado (schedule da arquitetura "
                            f"prevalece): {e}"
                        )
            else:
                optimizer = self.optimizer_factory.create_optimizer(
                    self.config.optimizer,
                    learning_rate=self.config.learning_rate,
                )
                # A loss é resolvida conforme a SAÍDA real do modelo (inclusive
                # ativação linear → from_logits=True) + o formato dos labels.
                resolved_loss = self._resolve_loss(model, y_train)
                model.compile(
                    optimizer=optimizer,
                    loss=resolved_loss,
                    metrics=self._resolve_metrics(),
                )

            # Preparar callbacks
            callbacks = self._prepare_callbacks(**kwargs)

            # `val_eer` só existe nos logs se este callback estiver ativo, e o
            # ModelCheckpoint pode estar configurado para monitorá-lo
            # (config.checkpoint_monitor). Registrar apenas quando pedido evita
            # pagar uma inferência extra por época nos runs que usam val_loss.
            if str(getattr(self.config, "checkpoint_monitor", "")) == "val_eer":
                if validation_data is not None:
                    # INSERE NA FRENTE, não no fim. O Keras percorre os
                    # callbacks em ORDEM dentro de `on_epoch_end`, passando o
                    # MESMO dicionário `logs` a todos. Anexado ao fim, o
                    # ValidationEER publicaria `val_eer` depois de o
                    # ModelCheckpoint e o PersistentEpochHistory já terem lido
                    # o dicionário — e ambos veriam a métrica ausente.
                    #
                    # Foi o que o smoke `_smoke_eer2` (2026-08-17) mostrou:
                    # com o encanamento do monitor já corrigido, o Keras
                    # avisou "Can save best model only with val_eer available"
                    # nas duas épocas, NENHUM `best.json` foi gravado e o
                    # histórico saiu sem a coluna. Um retreino de 27 h
                    # terminaria sem artefato selecionado.
                    callbacks.insert(
                        0,
                        ValidationEER(
                            validation_data=validation_data,
                            label=getattr(self.config, "progress_label", "")
                            or "training",
                            # Mesmo lote do treino: se cabe treinar com ele,
                            # cabe inferir com ele.
                            batch_size=int(
                                getattr(self.config, "batch_size", 32) or 32
                            ),
                        ),
                    )
                else:
                    # Falha ALTO: seguir com o monitor apontando para uma
                    # métrica que ninguém publica faria o ModelCheckpoint nunca
                    # salvar — 27 h de treino sem artefato.
                    raise ValueError(
                        "checkpoint_monitor='val_eer' exige validation_data "
                        "para o callback ValidationEER; sem ela nenhum "
                        "checkpoint seria salvo."
                    )

            # Aplicar data augmentation se habilitado
            if self.config.use_augmentation:
                train_dataset = self.augmenter.create_augmented_dataset(
                    X_train, y_train, self.config.batch_size
                )
                # O dataset aumentado é FINITO (~2N amostras). Com
                # steps_per_epoch fixo e SEM repeat(), o iterator esgotava na
                # ~2ª época e o Keras interrompia o treino ("ran out of data").
                # repeat() + steps_per_epoch garante épocas completas sempre.
                train_dataset = train_dataset.repeat()
                steps_per_epoch = max(
                    1, len(X_train) // self.config.batch_size
                )
            else:
                train_dataset, train_streaming = self._array_dataset(
                    X_train,
                    y_train,
                    batch_size=self.config.batch_size,
                    purpose="train",
                    shuffle=True,
                )
                if train_streaming:
                    train_dataset = train_dataset.repeat()
                    steps_per_epoch = int(
                        np.ceil(len(y_train) / max(1, self.config.batch_size))
                    )
                else:
                    steps_per_epoch = None
            train_dataset = optimize_tf_dataset(
                train_dataset, cache=False, prefetch=True
            )

            # Sprint 2.4: Mixup data augmentation (opt-in).
            # IMPORTANTE: Mixup produz soft labels, então é incompatível com
            # losses sparse (sparse_categorical_crossentropy). Quando habilitado,
            # converte y para one-hot e usa categorical_crossentropy automaticamente.
            if getattr(self.config, "use_mixup", False):
                try:
                    num_classes = int(self._infer_num_classes(y_train))
                    alpha = float(getattr(self.config, "mixup_alpha", 0.2))
                    train_dataset = self.augmenter.apply_mixup_to_dataset(
                        train_dataset, alpha=alpha, num_classes=num_classes
                    )
                    # Class weighting é incompatível com soft labels do mixup
                    # (Keras espera int classes em class_weight dict)
                    self._mixup_enabled = True
                    self.logger.info(
                        f"Mixup habilitado: α={alpha}, num_classes={num_classes}"
                    )
                except Exception as e:
                    self.logger.warning(f"Falha ao habilitar Mixup: {e}")
                    self._mixup_enabled = False
            else:
                self._mixup_enabled = False

            # Preparar dados de validação
            X_val, y_val = validation_data
            val_dataset, _val_streaming = self._array_dataset(
                X_val,
                y_val,
                batch_size=self.config.batch_size,
                purpose="validation",
            )
            val_dataset = optimize_tf_dataset(val_dataset, cache=False, prefetch=True)

            # Class weighting automático para datasets desbalanceados
            # Desabilita quando Mixup está ativo (soft labels não são compatíveis
            # com class_weight dict de Keras)
            if self._mixup_enabled:
                class_weight = None
                self.logger.info(
                    "Class weighting pulado (incompatível com Mixup soft labels)"
                )
            else:
                class_weight = self._compute_class_weights(y_train)

            verbose = int(getattr(self.config, "verbose", 1))

            # Treinar modelo
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                class_weight=class_weight,
                verbose=verbose,
            )

            # Conjunto de calibração: por padrão inclui cópias com ruído (AWGN
            # nos SNRs do teste) para que temperatura/threshold reflitam o ponto
            # de operação sob ruído (não só em áudio limpo).
            calibration_data = self._build_calibration_set(validation_data)

            # Calibração automática de temperatura (post-hoc)
            self._calibrated_temperature = self._auto_calibrate_temperature(
                model, calibration_data
            )

            # Sprint 2.5: Calibra threshold de OOD detection no val set
            self._ood_threshold = self._compute_ood_threshold(
                model, calibration_data, self._calibrated_temperature
            )

            # Sprint 4.5: Calibra threshold EER (Equal Error Rate) no val set
            # Habilita threshold adaptativo por modelo na inferência (em vez de 0.5)
            self._eer_threshold, self._eer_value = self._compute_eer_threshold(
                model, calibration_data, self._calibrated_temperature
            )

            # Calcular métricas finais
            final_metrics = self._calculate_final_metrics(model, validation_data)

            # `history.history` cobre só as épocas DESTA execução; numa
            # retomada isso truncaria a série (RawNet2 saiu com 17 de 100 no
            # clean_benchmark_15k). O callback persistente devolve o treino
            # inteiro. Só substitui se for pelo menos tão completo quanto.
            full_history = None
            history_cb = getattr(self, "_history_callback", None)
            if history_cb is not None:
                merged = history_cb.merged()
                longest = max((len(v) for v in merged.values()), default=0)
                current = max(
                    (len(v) for v in (history.history or {}).values()), default=0
                )
                if longest >= current:
                    full_history = merged

            result = {
                "history": full_history or history.history,
                "final_metrics": final_metrics,
                "model_summary": self._get_model_summary(model),
                "training_config": self.config.__dict__,
            }

            # Um treino abortado por colapso NÃO pode passar por treino curto
            # normal: sem isso o artefato registraria só "menos épocas".
            collapse_cb = getattr(self, "_collapse_callback", None)
            if collapse_cb is not None and collapse_cb.triggered:
                result["collapsed"] = True
                result["collapse_reason"] = collapse_cb.reason
                self.logger.warning(
                    "Treinamento ABORTADO por colapso: %s", collapse_cb.reason
                )
                return ProcessingResult(status=ProcessingStatus.SUCCESS, data=result)

            self.logger.info("Treinamento concluído com sucesso")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=result)

        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def get_scaler(self):
        """Retorna o scaler treinado para uso em predições."""
        if hasattr(self, "secure_pipeline"):
            return self.secure_pipeline.get_scaler()
        else:
            self.logger.warning("Pipeline seguro não inicializado")
            return None

    def predict_with_scaler(self, model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
        """Faz predição aplicando o mesmo scaler usado no treinamento."""
        scaler = self.get_scaler()
        if scaler is None or scaler.scaler is None:
            self.logger.warning("Scaler não disponível - usando dados sem normalização")
            return model.predict(X, verbose=0)

        # Aplicar mesma normalização usada no treinamento
        X_scaled = scaler.transform_test(X)
        return model.predict(X_scaled, verbose=0)

    def save_training_artifacts(
        self, model: tf.keras.Model, save_dir: Union[str, Path]
    ) -> ProcessingResult[Dict[str, str]]:
        """Salva modelo e artefatos de treinamento (incluindo scaler)."""
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Salvar modelo no formato nativo Keras 3 (.keras) como artefato de
            # INFERÊNCIA — sem o estado do otimizador (~3× menor, load mais
            # rápido, saída idêntica). Ver save_inference_keras().
            model_path = save_dir / "model.keras"
            save_inference_keras(model, model_path)

            # Salvar scaler se disponível
            scaler_path = None
            scaler = self.get_scaler()
            if scaler is not None and scaler.scaler is not None:
                scaler_path = save_dir / "scaler.pkl"
                scaler.save_scaler(scaler_path)

            # Salvar configuração com input_contract para consistência train/inference
            config_path = save_dir / "training_config.json"
            import json

            # Construir input_contract a partir do modelo treinado
            input_contract = self._build_input_contract(model)

            config_dict = {
                "model_config": self.config.__dict__,
                "secure_pipeline_config": self.secure_pipeline.config.__dict__
                if hasattr(self, "secure_pipeline")
                else None,
                "training_timestamp": datetime.now().isoformat(),
                "input_contract": input_contract,
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            artifacts = {"model_path": str(model_path), "config_path": str(config_path)}

            if scaler_path:
                artifacts["scaler_path"] = str(scaler_path)

            # Sprint 3.4: ONNX export opcional
            if getattr(self.config, "export_onnx", False):
                onnx_paths = self._export_onnx_artifacts(model, save_dir)
                artifacts.update(onnx_paths)

            self.logger.info(f"Artefatos de treinamento salvos em: {save_dir}")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=artifacts)

        except Exception as e:
            self.logger.error(f"Erro ao salvar artefatos: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def evaluate(
        self,
        model: tf.keras.Model,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> ProcessingResult[Dict[str, float]]:
        """Avalia modelo usando dados de teste seguros."""
        try:
            self.logger.info("Iniciando avaliação segura do modelo")

            # Usar dados de teste do pipeline seguro se disponíveis
            if test_data is None and hasattr(self, "_test_data"):
                X_test, y_test = self._test_data
                self.logger.info("Usando dados de teste do pipeline seguro")
            elif test_data is not None:
                X_test, y_test = test_data
                self.logger.warning("Usando dados de teste fornecidos externamente")
            else:
                raise ValueError(
                    "Nenhum dado de teste disponível. Execute o treinamento primeiro ou forneça test_data."
                )

            # Avaliação básica
            test_loss, *test_metrics = model.evaluate(
                X_test, y_test, batch_size=self.config.batch_size, verbose=0
            )

            # Predições para métricas detalhadas
            y_pred = model.predict(
                X_test, batch_size=self.config.batch_size, verbose=0
            )
            from app.domain.services.detection.predictor import (
                normalize_logits_to_probs,
            )
            y_pred = normalize_logits_to_probs(y_pred)
            # Suporta saídas (N,1) sigmoid e (N,K) softmax
            y_pred_classes = (
                np.argmax(y_pred, axis=1)
                if (y_pred.ndim > 1 and y_pred.shape[-1] > 1)
                else (y_pred.ravel() > 0.5).astype(int)
            )

            # Calcular métricas detalhadas
            detailed_metrics = self.metrics_calculator.calculate_all_metrics(
                y_test, y_pred_classes, y_pred
            )

            # Combinar métricas
            metrics = {
                "test_loss": float(test_loss),
                **{
                    f"test_{metric}": float(value)
                    for metric, value in zip(self.config.metrics, test_metrics)
                },
                **detailed_metrics,
            }

            self.logger.info("Avaliação segura concluída com sucesso")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=metrics)

        except Exception as e:
            self.logger.error(f"Erro durante avaliação: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def save_model(
        self, model: tf.keras.Model, save_path: Union[str, Path]
    ) -> ProcessingResult[str]:
        """Salva modelo treinado."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Salvar modelo como artefato de inferência (sem estado do otimizador;
            # ~3× menor e load mais rápido, saída idêntica). Ver save_inference_keras.
            save_inference_keras(model, save_path)

            # Salvar configuração de treinamento com input_contract
            config_path = save_path.parent / f"{save_path.stem}_config.json"
            import json

            config_data = {
                **self.config.__dict__,
                "input_contract": self._build_input_contract(model),
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

            self.logger.info(f"Modelo salvo em: {save_path}")
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, data=str(save_path)
            )

        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def load_model(
        self, model_path: Union[str, Path]
    ) -> ProcessingResult[tf.keras.Model]:
        """Carrega modelo salvo."""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

            # Carregar modelo
            model = tf.keras.models.load_model(str(model_path))

            self.logger.info(f"Modelo carregado de: {model_path}")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=model)

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def _prepare_callbacks(self, **kwargs) -> List[tf.keras.callbacks.Callback]:
        """Prepara callbacks para treinamento."""
        callbacks = []

        # Termina o treinamento imediatamente se NaN/Inf aparecer na loss
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())

        progress_interval = int(getattr(self.config, "progress_log_interval", 0) or 0)
        if progress_interval > 0:
            callbacks.append(
                EpochProgressLogger(
                    interval=progress_interval,
                    label=getattr(self.config, "progress_label", "") or "training",
                )
            )

        # Sprint 2.3: Stochastic Weight Averaging (opt-in via TrainingConfig)
        if getattr(self.config, "use_swa", False):
            try:
                from app.domain.models.training.swa_callback import SWACallback

                swa_cb = SWACallback(
                    start_epoch=getattr(self.config, "swa_start_epoch", -1),
                    swa_freq=getattr(self.config, "swa_freq", 1),
                    bn_update_data=kwargs.get("bn_update_data", None),
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
                callbacks.append(swa_cb)
                self._swa_callback = swa_cb  # acessível depois do treino
                self.logger.info("SWA habilitado")
            except Exception as e:
                self.logger.warning(f"Falha ao adicionar SWA callback: {e}")

        # Histórico resistente a retomadas. Ancorado no diretório do
        # checkpoint, NÃO no `backup_dir` — este último é apagado ao fim do
        # treino (delete_checkpoint=True) e levaria o histórico junto.
        history_anchor = kwargs.get("checkpoint_path") or kwargs.get("backup_dir")
        if history_anchor:
            anchor = Path(str(history_anchor))
            history_dir = anchor.parent if anchor.suffix else anchor
            history_cb = PersistentEpochHistory(
                history_dir / "epoch_history.jsonl",
                label=getattr(self.config, "progress_label", "") or "training",
            )
            callbacks.append(history_cb)
            self._history_callback = history_cb

        # Guarda de colapso — independente do early stopping (ver docstring de
        # CollapseAbort: uma coisa é parar um modelo que ainda melhora, outra é
        # abortar um que virou palpite constante e não volta).
        if getattr(self.config, "abort_on_collapse", True):
            collapse_cb = CollapseAbort(
                patience=int(getattr(self.config, "collapse_patience", 15)),
                nan_patience=int(getattr(self.config, "collapse_nan_patience", 3)),
                label=getattr(self.config, "progress_label", "") or "training",
            )
            callbacks.append(collapse_cb)
            self._collapse_callback = collapse_cb

        # Early stopping
        if getattr(self.config, "early_stopping", True):
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
            )

        # Reduce learning rate. Arquiteturas com LearningRateSchedule próprio
        # não aceitam setar optimizer.learning_rate em runtime.
        if getattr(self.config, "reduce_lr_on_plateau", True):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
            )

        # Model checkpoint
        if "checkpoint_path" in kwargs:
            # AJUSTE 2026-07-14: quando o caminho termina em `.weights.h5`,
            # salva SÓ os pesos — o save de modelo completo serializava grafo
            # + estado do otimizador (~3× os pesos; ~1 GB por melhoria de
            # época no AST) e dependia da desserialização de camadas custom.
            # A restauração já usa load_weights, que aceita ambos os formatos.
            ckpt_path = str(kwargs["checkpoint_path"])
            # Monitor CONFIGURÁVEL, com `val_loss` como padrão.
            #
            # O padrão preserva a reprodutibilidade dos artefatos publicados,
            # todos selecionados por menor perda de validação. Mas o critério
            # tem um descompasso conhecido com a avaliação (ver `ValidationEER`):
            # seleciona por calibração, avalia por ordenação. Para runs novos
            # de anti-spoofing, `val_eer` alinha os dois — e exige que o
            # callback `ValidationEER` esteja ativo para publicar a métrica.
            monitor = str(getattr(self.config, "checkpoint_monitor", "")
                          or "val_loss")
            modo = "min" if monitor in ("val_loss", "val_eer") else "max"
            callbacks.append(
                ResumableModelCheckpoint(
                    filepath=ckpt_path,
                    monitor=monitor,
                    mode=modo,
                    save_best_only=True,
                    save_weights_only=ckpt_path.endswith(".weights.h5"),
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
            )

        # Recuperacao de falhas de infraestrutura (reinicio do host/Docker).
        # Diferentemente do melhor checkpoint, o backup preserva tambem o
        # estado do otimizador e a epoca concluida, permitindo que fit()
        # retome sem transformar a continuacao em um novo experimento.
        #
        # AJUSTE 2026-08-04 (queda de energia): double_checkpoint=True. O
        # save escreve os pesos POR CIMA do backup unico; uma queda no meio
        # dessa gravacao deixa um HDF5 truncado e sem fallback — perdendo o
        # treino inteiro, nao so a epoca corrente. Com a opcao ligada o Keras
        # mantem o estado anterior em `.bkp` e cai nele quando o atual falha
        # ao carregar. Custa o dobro de disco no diretorio de backup.
        if "backup_dir" in kwargs:
            callbacks.append(
                BackupAndRestore(
                    backup_dir=str(kwargs["backup_dir"]),
                    save_freq="epoch",
                    double_checkpoint=True,
                    delete_checkpoint=True,
                )
            )

        # TensorBoard
        if "tensorboard_dir" in kwargs:
            callbacks.append(
                TensorBoard(
                    log_dir=kwargs["tensorboard_dir"],
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                )
            )

        # CSV Logger
        if "csv_log_path" in kwargs:
            callbacks.append(CSVLogger(filename=kwargs["csv_log_path"], append=True))

        return callbacks

    @staticmethod
    def _should_enable_mixed_precision() -> bool:
        """Sprint 3.2: auto-detecta se mixed precision deve ser habilitado.

        Critério: existe ao menos uma GPU com Compute Capability >= 7.0
        (Volta+, ou seja: V100, T4, RTX 20xx, RTX 30xx, RTX 40xx, A100, H100).
        GPUs anteriores (Pascal/Maxwell) não têm Tensor Cores e mixed precision
        pode até reduzir performance ou causar overflow numérico.

        Retorna False se não houver GPU, ou se a detecção falhar.
        """
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return False
            # Inspeciona compute capability via details (TF 2.6+)
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    cc = details.get("compute_capability")
                    if cc is not None:
                        major = (
                            cc[0]
                            if isinstance(cc, (list, tuple))
                            else int(str(cc).split(".")[0])
                        )
                        if major >= 7:
                            return True
                except Exception:
                    # Detalhes não disponíveis — assume seguro habilitar se há GPU
                    return True
            return False
        except Exception:
            return False

    def _resolve_metrics(self) -> List[Any]:
        """Métricas seguras para o `compile` (só as nativas do Keras).

        O default do TrainingConfig inclui 'f1' — que NÃO é métrica nativa do
        Keras (`Could not interpret metric identifier: f1`). Além disso,
        Precision/Recall/AUC como classes podem quebrar com saída softmax de 2
        unidades + labels esparsos (esperam binário). Como F1, EER, precision e
        recall já são calculados post-hoc pelo MetricsCalculator
        (`calculate_all_metrics`), no `compile` usamos APENAS 'accuracy', que é
        robusta para binário/multiclasse e labels esparsos/one-hot.
        """
        return ["accuracy"]

    def _resolve_loss(self, model: tf.keras.Model, y_train: np.ndarray):
        """Escolhe a loss compatível com a saída do modelo e o formato dos labels.

        O `loss_function` default do TrainingConfig é `binary_crossentropy`
        (assume sigmoid de 1 unidade). Mas o TrainingService instancia modelos
        com `num_classes` detectado (2 para binário) → saída softmax de 2
        unidades. Compilar com binary_crossentropy nesse caso quebra o fit com
        "target and output must have the same rank". Aqui auto-corrigimos:

        - saída 1 unidade  → binary_crossentropy (labels esparsos ou (N,1))
        - saída K>1 + labels esparsos (N,)   → sparse_categorical_crossentropy
        - saída K>1 + labels one-hot (N,K)   → categorical_crossentropy

        Respeita a loss configurada quando ela já é compatível.
        """
        configured = self.config.loss_function
        try:
            out_units = int(model.output_shape[-1])
        except Exception:
            return configured

        y = np.asarray(y_train)
        labels_one_hot = y.ndim > 1 and y.shape[-1] > 1

        # Saída LINEAR (sem sigmoid/softmax) emite LOGITS CRUS — ex.: AASIST
        # com AMSoftmaxLayer. As strings de loss do Keras assumem
        # from_logits=False e fariam log/normalização de valores negativos
        # (gradiente sem sentido). Detecta e usa objetos com from_logits=True.
        from_logits = False
        try:
            last_act = getattr(model.layers[-1], "activation", None)
            from_logits = last_act is None or last_act is tf.keras.activations.linear
        except Exception:
            pass

        if from_logits:
            if out_units == 1:
                chosen = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            elif labels_one_hot:
                chosen = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            else:
                chosen = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            self.logger.info(
                "Saída linear detectada (logits crus) — loss com from_logits=True."
            )
            return chosen

        if out_units == 1:
            chosen = "binary_crossentropy"
        elif labels_one_hot:
            chosen = "categorical_crossentropy"
        else:
            chosen = "sparse_categorical_crossentropy"

        # Se a loss configurada já é compatível, mantém (evita sobrescrever
        # escolhas legítimas como focal loss customizada via string).
        compatible = {
            1: {"binary_crossentropy", "bce", "mse", "mae"},
        }.get(out_units, (
            {"categorical_crossentropy", "kl_divergence"}
            if labels_one_hot
            else {"sparse_categorical_crossentropy"}
        ))
        if configured in compatible:
            return configured

        if chosen != configured:
            self.logger.warning(
                f"Loss '{configured}' incompatível com saída de {out_units} "
                f"unidade(s) + labels "
                f"{'one-hot' if labels_one_hot else 'esparsos'}; "
                f"usando '{chosen}'."
            )
        return chosen

    def _infer_num_classes(self, y: np.ndarray) -> int:
        """Inferência de num_classes a partir de y (suporta sparse e one-hot)."""
        y_arr = np.asarray(y)
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            return int(y_arr.shape[-1])
        return max(int(np.unique(y_arr).size), 2)

    def _compute_class_weights(self, y_train: np.ndarray) -> Optional[Dict[int, float]]:
        """Calcula pesos por classe para compensar desbalanceamento.

        Usa `sklearn.utils.class_weight.compute_class_weight('balanced')`,
        equivalente a `n_samples / (n_classes * np.bincount(y))`.

        Returns:
            Dict {class_idx: weight} ou None se desabilitado/inválido.
        """
        if not getattr(self.config, "use_class_weighting", True):
            return None

        try:
            from sklearn.utils.class_weight import compute_class_weight

            # Suporta y one-hot (N, C) e categórico (N,) ou (N, 1)
            y_arr = np.asarray(y_train)
            if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
                y_for_weights = np.argmax(y_arr, axis=-1)
            else:
                y_for_weights = y_arr.ravel().astype(int)

            unique_classes = np.unique(y_for_weights)
            if len(unique_classes) < 2:
                self.logger.warning(
                    "Class weighting pulado: apenas 1 classe presente em y_train"
                )
                return None

            weights_array = compute_class_weight(
                "balanced", classes=unique_classes, y=y_for_weights
            )
            class_weight = {
                int(c): float(w) for c, w in zip(unique_classes, weights_array)
            }

            # Log: contagem por classe + pesos
            counts = {int(c): int((y_for_weights == c).sum()) for c in unique_classes}
            self.logger.info(
                f"Class weighting habilitado | counts={counts} | weights={class_weight}"
            )
            return class_weight

        except Exception as e:
            self.logger.warning(f"Erro ao calcular class weights: {e}")
            return None

    @staticmethod
    def _add_awgn(X: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
        """AWGN a um SNR alvo (por amostra). Paridade EXATA com
        benchmarks/data.add_awgn: a potência REALIZADA do ruído é normalizada
        por amostra (antes só a potência esperada era calibrada — SNR
        realizado desviava ~0,4% em janelas de 80k amostras; agora o SNR
        realizado é idêntico ao alvo, como no protocolo do benchmark)."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype="float32")
        flat = X.reshape(len(X), -1)
        sig_power = np.mean(flat ** 2, axis=1, keepdims=True)
        snr_lin = 10.0 ** (float(snr_db) / 10.0)

        unit_noise = rng.standard_normal(flat.shape).astype("float32")
        unit_power = np.mean(unit_noise ** 2, axis=1, keepdims=True)
        target_power = sig_power / max(snr_lin, 1e-12)
        scale = np.sqrt(target_power / np.maximum(unit_power, 1e-12))
        noise = unit_noise * scale
        return (flat + noise).reshape(X.shape).astype("float32")

    @staticmethod
    def _looks_like_waveform(X: np.ndarray) -> bool:
        """Heurística de forma de onda (paridade com benchmarks.data).

        (N, T) ou (N, T, 1) com T >= 1000 amostras. Espectrogramas do projeto
        têm (T~100, F~80) e nunca batem esse critério.
        """
        arr = np.asarray(X)
        if arr.ndim == 2:
            return arr.shape[1] >= 1000
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr.shape[1] >= 1000
        return False

    def _build_calibration_set(self, validation_data):
        """Val limpo + cópias com AWGN para calibrar sob ruído.

        Gated por `calibrate_under_noise` (default True) nos SNRs de
        `calibration_snr_db`. Retorna `(X, y)`; em falha, devolve o val original.

        Protocolo AWGN (2026-07-14): o ruído só é fisicamente válido na FORMA
        DE ONDA. Para modelos espectrais/tabulares (val já em log-mel ou
        features), adicionar AWGN aqui perturbaria o domínio errado — nesses
        casos a calibração usa apenas o val limpo (mesma regra que o benchmark
        aplica ao desativar calibrate_under_noise).
        """
        if validation_data is None:
            return validation_data
        if not getattr(self.config, "calibrate_under_noise", True):
            return validation_data
        snrs = list(getattr(self.config, "calibration_snr_db", []) or [])
        if not snrs:
            return validation_data
        if not self._looks_like_waveform(validation_data[0]):
            self.logger.info(
                "Calibração sob ruído pulada: entrada de validação não é forma "
                "de onda (AWGN só é aplicado no domínio do waveform; ver "
                "protocolo 2026-07-12). Calibrando com val limpo."
            )
            return validation_data
        try:
            X_val, y_val = validation_data
            X_val = np.asarray(X_val, dtype="float32")
            y_val = np.asarray(y_val)
            extra = [self._add_awgn(X_val, snr, seed=1234 + i)
                     for i, snr in enumerate(snrs)]
            X_cal = np.concatenate([X_val, *extra], axis=0)
            y_cal = np.concatenate([y_val] * (1 + len(snrs)), axis=0)
            self.logger.info(
                "Calibração sob ruído: val %d → %d amostras (SNRs=%s)",
                len(y_val), len(y_cal), snrs,
            )
            return (X_cal, y_cal)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Falha ao montar calibração com ruído: %s", e)
            return validation_data

    def _auto_calibrate_temperature(
        self, model: tf.keras.Model, validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """Calibração post-hoc de temperatura via grid search no conjunto de validação.

        Implementa Temperature Scaling (Guo et al., ICML 2017): busca o T que
        minimiza NLL nas predições do val set. O valor é salvo no input_contract
        e aplicado automaticamente pelo Predictor na inferência.

        Returns:
            Temperatura calibrada (1.0 se desabilitado/falhar).
        """
        default_t = 1.0
        if not getattr(self.config, "auto_calibrate_temperature", True):
            return default_t

        try:
            X_val, y_val = validation_data
            min_samples = getattr(self.config, "calibration_min_samples", 50)
            if len(X_val) < min_samples:
                self.logger.info(
                    f"Calibração de temperatura pulada: val set tem "
                    f"{len(X_val)} amostras (mínimo {min_samples})"
                )
                return default_t

            # Converte y para índices de classe (compatível com calibrate())
            y_for_calib = np.asarray(y_val)
            if y_for_calib.ndim > 1 and y_for_calib.shape[-1] > 1:
                y_for_calib = np.argmax(y_for_calib, axis=-1)
            else:
                y_for_calib = y_for_calib.ravel().astype(int)

            # Importação tardia para evitar ciclo (predictor importa do trainer)
            from app.domain.services.detection.predictor import TemperatureScaler

            scaler = TemperatureScaler()
            scaler.calibrate(model, X_val, y_for_calib)
            temperature = float(scaler.temperature)
            self.logger.info(
                f"Temperatura calibrada: T={temperature:.3f} "
                f"({len(X_val)} amostras de val)"
            )
            return temperature

        except Exception as e:
            self.logger.warning(f"Erro na calibração de temperatura: {e}")
            return default_t

    def _compute_ood_threshold(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray],
        temperature: float = 1.0,
    ) -> Optional[float]:
        """Sprint 2.5: Calibra threshold de OOD detection no val set.

        Computa energy scores para todas as amostras de validação (que são
        consideradas in-distribution por construção) e usa o quantil
        (1 - ood_quantile) como threshold. Amostras com energy score abaixo
        desse threshold serão flagged como OOD na inferência.

        Args:
            model: modelo treinado
            validation_data: (X_val, y_val)
            temperature: T calibrado (para consistência com energia)

        Returns:
            Threshold (float) ou None se desabilitado/falhar.
        """
        if not getattr(self.config, "compute_ood_threshold", True):
            return None

        try:
            from app.domain.services.detection.predictor import (
                apply_temperature_scaling,
                compute_energy_score,
                normalize_logits_to_probs,
            )

            X_val, _ = validation_data
            predictions = model.predict(
                X_val, batch_size=self.config.batch_size, verbose=0
            )
            # Aplica mesma temperatura que será usada em inferência
            predictions = normalize_logits_to_probs(predictions)
            predictions = apply_temperature_scaling(predictions, temperature)
            energy_scores = compute_energy_score(predictions, temperature=temperature)

            # Threshold = quantil inferior. ood_quantile=0.95 → 5% das amostras
            # in-distribution com menores energy scores serão falsos positivos OOD.
            q = 1.0 - float(getattr(self.config, "ood_quantile", 0.95))
            threshold = float(np.quantile(energy_scores, q))
            self.logger.info(
                f"OOD threshold calibrado: {threshold:.4f} "
                f"(quantile {q:.2f} de {len(energy_scores)} amostras val | "
                f"range=[{energy_scores.min():.3f}, {energy_scores.max():.3f}])"
            )
            return threshold

        except Exception as e:
            self.logger.warning(f"Erro ao calibrar OOD threshold: {e}")
            return None

    def _calculate_final_metrics(
        self, model: tf.keras.Model, validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Calcula métricas finais do modelo."""
        X_val, y_val = validation_data

        # Predições
        y_pred = model.predict(X_val, batch_size=self.config.batch_size, verbose=0)
        from app.domain.services.detection.predictor import (
            normalize_logits_to_probs,
        )
        y_pred = normalize_logits_to_probs(y_pred)
        # Suporta saídas (N,1) sigmoid e (N,K) softmax
        y_pred_classes = (
            np.argmax(y_pred, axis=1)
            if (y_pred.ndim > 1 and y_pred.shape[-1] > 1)
            else (y_pred.ravel() > 0.5).astype(int)
        )

        # Calcular métricas
        return self.metrics_calculator.calculate_all_metrics(
            y_val, y_pred_classes, y_pred
        )

    def _compute_eer_threshold(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray],
        temperature: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Sprint 4.5: Calibra Equal Error Rate threshold no val set.

        EER é o ponto onde FPR == FNR — métrica padrão em anti-spoofing
        (ASVspoof). O threshold associado é frequentemente mais informativo
        que o 0.5 fixo, pois balanceia falsos positivos e falsos negativos.

        Args:
            model: modelo treinado
            validation_data: (X_val, y_val)
            temperature: T calibrado (Sprint 1.4) para consistência

        Returns:
            (eer_threshold, eer_value), ambos float ou None se falhar.
            - eer_threshold: score acima do qual classificar como fake
            - eer_value: valor de EER (taxa de erro no ponto FPR=FNR)
        """
        try:
            from app.domain.services.detection.predictor import (
                apply_temperature_scaling,
                normalize_logits_to_probs,
            )

            X_val, y_val = validation_data
            predictions = model.predict(
                X_val, batch_size=self.config.batch_size, verbose=0
            )
            predictions = normalize_logits_to_probs(predictions)
            predictions = apply_temperature_scaling(predictions, temperature)

            # Extrai score de probabilidade da classe "fake" (índice 1)
            if predictions.ndim > 1 and predictions.shape[-1] > 1:
                scores = predictions[:, 1]
            elif predictions.ndim > 1 and predictions.shape[-1] == 1:
                scores = predictions[:, 0]
            else:
                scores = predictions.ravel()

            # y_val pode ser sparse ou one-hot
            y_arr = np.asarray(y_val)
            if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
                y_true = np.argmax(y_arr, axis=-1)
            else:
                y_true = y_arr.ravel().astype(int)

            # Usa MetricsCalculator.calculate_eer (já existente)
            eer_value, eer_threshold = self.metrics_calculator.calculate_eer(
                y_true, scores
            )
            self.logger.info(
                f"EER threshold calibrado: T={eer_threshold:.4f}, EER={eer_value:.4f} "
                f"({len(scores)} amostras val) — alternativa ao threshold 0.5"
            )
            return float(eer_threshold), float(eer_value)

        except Exception as e:
            self.logger.warning(f"Erro ao calibrar EER threshold: {e}")
            return None, None

    def _export_onnx_artifacts(
        self,
        model: tf.keras.Model,
        save_dir: Path,
    ) -> Dict[str, str]:
        """Sprint 3.4: Export ONNX FP32 e INT8 (opcional).

        Degrada graciosamente se tf2onnx/onnxruntime não estão instalados —
        retorna dict vazio sem levantar exceção (NÃO bloqueia save do .keras).
        """
        artifacts: Dict[str, str] = {}
        try:
            from app.domain.models.inference.onnx_export import (
                export_to_onnx,
                is_onnx_available,
                quantize_int8,
            )

            if not is_onnx_available():
                self.logger.info(
                    "ONNX export pulado: tf2onnx/onnxruntime não instalados. "
                    "Instale com: pip install tf2onnx onnxruntime"
                )
                return artifacts

            onnx_path = save_dir / "model.onnx"
            result = export_to_onnx(model, onnx_path)
            if result is not None:
                artifacts["onnx_path"] = str(result)

                # INT8 quantization opcional
                if getattr(self.config, "export_onnx_int8", False):
                    int8_path = save_dir / "model_int8.onnx"
                    # Usa val set como calibração se disponível
                    calib_data = None
                    if hasattr(self, "_test_data"):
                        X_test, _ = self._test_data
                        # Pega até 100 amostras para calibração estática
                        calib_data = X_test[:100].astype(np.float32)
                    int8_result = quantize_int8(result, int8_path, calib_data)
                    if int8_result is not None:
                        artifacts["onnx_int8_path"] = str(int8_result)
        except Exception as e:
            self.logger.warning(f"ONNX export falhou (não-crítico): {e}")
        return artifacts

    def _build_input_contract(
        self, model: tf.keras.Model, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Constrói contrato de entrada para garantir consistência train/inference.

        O input_contract é salvo junto ao modelo e lido pelo FeaturePreparer
        na inferência para garantir que as mesmas features/formato sejam usados.
        """
        metadata = metadata or {}
        model_input_shape = list(model.input_shape[1:]) if model.input_shape else None

        # Determinar tipo de input a partir da arquitetura ou shape
        architecture = metadata.get("architecture", "")
        input_type = metadata.get("input_type", "features")
        input_format = metadata.get("input_format", "tabular")

        # Heurística: modelos com input (N, 1) provavelmente recebem áudio raw
        if (
            model_input_shape
            and len(model_input_shape) == 2
            and model_input_shape[-1] == 1
        ):
            input_type = "audio"
            input_format = "raw"
        # Modelos com input (H, W) ou (H, W, C) provavelmente usam spectrogram
        elif (
            model_input_shape
            and len(model_input_shape) in (2, 3)
            and (model_input_shape[-1] != 1 if len(model_input_shape) == 2 else True)
        ):
            if any(dim and dim > 10 for dim in model_input_shape[:2]):
                input_type = "features"
                input_format = "spectrogram"

        # Feature types se disponíveis
        feature_types = metadata.get("feature_types", None)
        if feature_types is None:
            feature_types_attr = getattr(model, "feature_types_used", None)
            if feature_types_attr:
                feature_types = list(feature_types_attr)

        contract = {
            "type": input_type,
            "format": input_format,
            "input_shape": model_input_shape,
            "architecture": architecture,
            "feature_types": feature_types,
            "sample_rate": metadata.get("sample_rate", 16000),
            "scaler_applied": self.get_scaler() is not None
            and self.get_scaler().scaler is not None,
            # Temperatura calibrada (Sprint 1.4) — aplicada na inferência pelo Predictor
            "temperature": float(getattr(self, "_calibrated_temperature", 1.0)),
        }

        # Persiste o contrato canônico da arquitetura (frontend, crop e janela).
        # Sem esta fusão, AASIST/RawGAT-ST perdiam a política multicrop ao salvar.
        if architecture:
            try:
                from app.domain.models.architectures.registry import (
                    get_architecture_info,
                )

                architecture_info = get_architecture_info(architecture)
                requirements = (
                    dict(architecture_info.input_requirements)
                    if architecture_info is not None
                    else {}
                )
                for key in (
                    "input_type",
                    "feature_frontend",
                    "target_sequence_length",
                    "source_samples",
                    "crop_strategy",
                    "sample_rate",
                    "preprocessing",
                ):
                    if key in requirements:
                        contract[key] = requirements[key]
            except Exception as exc:
                self.logger.debug(
                    "Contrato do registry não pôde ser incorporado: %s", exc
                )

        # Sprint 2.5: OOD threshold (energy-based). None se desabilitado/falhou.
        ood_t = getattr(self, "_ood_threshold", None)
        if ood_t is not None:
            contract["ood_threshold"] = float(ood_t)

        # A inferencia precisa saber se a ULTIMA camada emite logits crus
        # (AASIST/AM-Softmax) ou ja probabilidades. Sem este campo, o Predictor
        # adivinhava pela faixa de valores enquanto o benchmark decidia pela
        # ativacao da camada — criterios diferentes, que divergem quando logits
        # caem por acaso em [0, 1] e somam ~1.
        try:
            from app.domain.services.detection.predictor import model_emits_logits

            is_logits = model_emits_logits(model)
            if is_logits is not None:
                contract["output_is_logits"] = bool(is_logits)
        except Exception as exc:  # noqa: BLE001 — contrato sem o campo ainda serve
            self.logger.debug("output_is_logits indisponivel: %s", exc)

        # Sprint 4.5: EER threshold (Equal Error Rate) — alternativa adaptativa
        # ao threshold 0.5 fixo. Predictor pode usar via flag use_eer_threshold.
        eer_t = getattr(self, "_eer_threshold", None)
        eer_v = getattr(self, "_eer_value", None)
        if eer_t is not None:
            contract["eer_threshold"] = float(eer_t)
        if eer_v is not None:
            contract["eer_value"] = float(eer_v)
        return contract

    def predict_with_tta(
        self,
        model: tf.keras.Model,
        X: np.ndarray,
        n_augmentations: int = 5,
        noise_std: float = 0.005,
        shift_factor: float = 0.02,
        volume_range: Tuple = (0.95, 1.05),
    ) -> np.ndarray:
        """Test-Time Augmentation: run multiple augmented copies and average predictions.

        Typically improves accuracy by 1-3% without retraining.
        Uses 5 versions: original + noise + neg_noise + time_shift + volume_change.
        """
        predictions = []

        # Original prediction
        predictions.append(
            model.predict(X, batch_size=self.config.batch_size, verbose=0)
        )

        if n_augmentations >= 2:
            # Positive noise
            X_noise = X + np.random.normal(0, noise_std, X.shape).astype(np.float32)
            predictions.append(
                model.predict(X_noise, batch_size=self.config.batch_size, verbose=0)
            )

        if n_augmentations >= 3:
            # Negative noise
            X_noise_neg = X - np.random.normal(0, noise_std, X.shape).astype(np.float32)
            predictions.append(
                model.predict(
                    X_noise_neg, batch_size=self.config.batch_size, verbose=0
                )
            )

        if n_augmentations >= 4:
            # Time shift (small circular shift along first feature axis)
            shift_amount = (
                max(1, int(X.shape[1] * shift_factor)) if len(X.shape) > 1 else 0
            )
            if shift_amount > 0:
                X_shifted = np.roll(X, shift_amount, axis=1)
                predictions.append(
                    model.predict(
                        X_shifted, batch_size=self.config.batch_size, verbose=0
                    )
                )

        if n_augmentations >= 5:
            # Volume change
            vol_factor = np.random.uniform(volume_range[0], volume_range[1])
            X_vol = X * vol_factor
            predictions.append(
                model.predict(X_vol, batch_size=self.config.batch_size, verbose=0)
            )

        # Average all predictions
        avg_prediction = np.mean(predictions, axis=0)
        self.logger.info(f"TTA applied with {len(predictions)} augmentations")
        return avg_prediction

    def _get_model_summary(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Retorna resumo do modelo."""
        return {
            "total_params": model.count_params(),
            "trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            ),
            "non_trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
            ),
            "layers_count": len(model.layers),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
        }
