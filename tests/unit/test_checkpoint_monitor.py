"""Seleção de checkpoint: monitor configurável e EER de validação.

Contexto (2026-08-16): o critério `minimum_clean_validation_loss` seleciona por
CALIBRAÇÃO (entropia cruzada) enquanto a avaliação do protocolo é por ORDENAÇÃO
(EER, AUC, min t-DCF). O descompasso custou 5,3 pontos de acurácia no RawGAT-ST
publicado e, no retune com L2=3e-3, faria selecionar a época 1 -- um modelo
desinformativo, porque ln(2)=0,693 é um piso que a entropia de um modelo
confiantemente errado não bate.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from app.domain.models.training.trainer import ValidationEER


class _ModeloFalso:
    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype="float64")
        self.stop_training = False

    # A assinatura precisa aceitar `batch_size`: o callback passa o lote do
    # treino para não fazer `predict` de conjunto inteiro (estourava a VRAM
    # nas redes de forma de onda). Um duplo que não aceita o argumento faz o
    # teste exercitar o caminho de EXCEÇÃO em vez do cálculo do EER.
    def predict(self, x, verbose=0, batch_size=None):  # noqa: ARG002
        return self._scores


def _eer(scores, y):
    cb = ValidationEER(validation_data=(np.zeros((len(y), 1)), np.asarray(y)))
    cb.set_model(_ModeloFalso(scores))
    logs: dict = {}
    cb.on_epoch_end(0, logs)
    return logs.get("val_eer")


def test_separacao_perfeita_da_eer_zero():
    y = [0, 0, 0, 1, 1, 1]
    assert _eer([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], y) == pytest.approx(0.0, abs=1e-9)


def test_pontuacao_aleatoria_da_eer_proxima_de_meio():
    rng = np.random.default_rng(42)
    y = np.array([0] * 200 + [1] * 200)
    eer = _eer(rng.random(400), y)
    assert 0.35 < eer < 0.65


def test_ordem_invertida_da_eer_alta():
    y = [0, 0, 0, 1, 1, 1]
    assert _eer([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], y) == pytest.approx(1.0, abs=1e-9)


def test_eer_ignora_calibracao():
    """O ponto do EER: uma transformação monotônica não muda a ordenação.

    É exatamente por isso que ele é imune ao problema que derruba a seleção
    por entropia cruzada.
    """
    y = [0, 0, 1, 1]
    base = [0.10, 0.20, 0.80, 0.90]
    # mesma ordem, muito mais 'confiante' -- entropia mudaria, EER não
    confiante = [0.001, 0.002, 0.998, 0.999]
    assert _eer(base, y) == pytest.approx(_eer(confiante, y), abs=1e-9)


def test_saida_de_duas_colunas_usa_a_coluna_do_fake():
    """O caso NORMAL do escopo: softmax sobre {bonafide, spoof}.

    Os três primeiros smokes do `val_eer` falharam aqui e em silêncio:
    `ravel()` de uma saída (N, 2) dá 2N scores para N rótulos, e a checagem de
    tamanho devolvia sem logar. Os testes acima não pegavam porque o duplo
    emitia uma coluna só — a forma que NENHUMA arquitetura do escopo usa.
    """
    y = [0, 0, 1, 1]
    # coluna 0 = bonafide (ordem invertida de propósito), coluna 1 = fake
    duas_colunas = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype="float64"
    )
    cb = ValidationEER(validation_data=(np.zeros((4, 1)), np.asarray(y)))
    cb.set_model(_ModeloFalso(duas_colunas))
    logs: dict = {}
    cb.on_epoch_end(0, logs)

    assert "val_eer" in logs, "saída de 2 colunas não pode ser descartada"
    assert logs["val_eer"] == pytest.approx(0.0, abs=1e-9)
    assert cb.falhas == 0


def test_forma_incompativel_falha_alto():
    """Nem 1 nem 2 colunas: é defeito, e defeito silencioso custa o artefato."""
    cb = ValidationEER(validation_data=(np.zeros((4, 1)), np.array([0, 0, 1, 1])))
    cb.set_model(_ModeloFalso(np.zeros((4, 3))))
    logs: dict = {}
    cb.on_epoch_end(0, logs)

    assert "val_eer" not in logs
    assert cb.falhas == 1, "a falha precisa ser contada e logada, não engolida"


def test_uma_classe_so_nao_publica_metrica():
    logs: dict = {}
    cb = ValidationEER(validation_data=(np.zeros((4, 1)), np.array([1, 1, 1, 1])))
    cb.set_model(_ModeloFalso([0.1, 0.2, 0.3, 0.4]))
    cb.on_epoch_end(0, logs)
    assert "val_eer" not in logs


def test_falha_interna_nao_derruba_o_treino():
    """Métrica auxiliar não pode matar um treino de horas."""
    class _Quebrado:
        stop_training = False

        def predict(self, x, verbose=0, batch_size=None):  # noqa: ARG002
            raise RuntimeError("falha simulada")

    logs: dict = {}
    cb = ValidationEER(validation_data=(np.zeros((4, 1)), np.array([0, 0, 1, 1])))
    cb.set_model(_Quebrado())
    cb.on_epoch_end(0, logs)  # não deve levantar
    assert "val_eer" not in logs


# ─── Encanamento ───────────────────────────────────────────────────────────
#
# Os testes acima exercitam o callback ISOLADO, e passavam todos enquanto a
# opção não tinha efeito nenhum. O smoke `_smoke_eer` de 2026-08-17 pediu
# `val_eer`, treinou 2 épocas e gravou `{"monitor": "val_loss"}` no best.json:
# `run_rawgat_retune.py` setava o atributo no BenchmarkConfig, o trainer lia
# do TrainingConfig, e não havia elo entre os dois. Os quatro testes a seguir
# cobrem cada elo da corrente.


def test_training_config_declara_o_campo():
    """O elo que quebrou: o TrainingService filtra o config pelos campos
    DECLARADOS do dataclass, então uma chave não declarada é descartada em
    silêncio -- sem erro, sem log, sem efeito."""
    from app.core.config.settings import TrainingConfig

    campos = {f.name for f in dataclasses.fields(TrainingConfig)}
    assert "checkpoint_monitor" in campos
    assert TrainingConfig().checkpoint_monitor == "val_loss"


def test_benchmark_config_declara_o_campo_com_o_padrao_publicado():
    from benchmarks.config import BenchmarkConfig

    campos = {f.name for f in dataclasses.fields(BenchmarkConfig)}
    assert "checkpoint_monitor" in campos
    # Trocar este padrão invalidaria os 10 artefatos que não pagam o
    # descompasso -- ver o comentário do campo.
    assert BenchmarkConfig(dataset_path="x").checkpoint_monitor == "val_loss"


def test_runner_propaga_o_monitor_para_o_treino():
    """`_run_neural` precisa copiar o monitor do BenchmarkConfig para o dict
    que vai ao TrainingService."""
    import inspect

    from benchmarks import runner

    fonte = inspect.getsource(runner._run_neural)
    assert 'train_config["checkpoint_monitor"]' in fonte


# ─── comportamento ponta a ponta ───────────────────────────────────────────
#
# O teste abaixo é o que faltava. Tudo acima — inclusive as verificações de
# encanamento — olha PARTES: um campo declarado, uma string no fonte, um
# callback isolado. A opção `val_eer` passou por três defeitos em série
# (encanamento morto, ordem dos callbacks, saída de 2 colunas) com todos esses
# testes verdes, e cada um só apareceu num smoke de 2 épocas na GPU: ~40 min
# por iteração, cinco iterações.
#
# Este exercita a cadeia inteira num `fit` de brinquedo, em segundos, e teria
# reprovado nos TRÊS casos. É o teste que justifica os outros existirem só
# como diagnóstico de qual elo quebrou.


def test_treino_real_seleciona_checkpoint_por_val_eer(tmp_path):
    """`ModelTrainer.train` com `checkpoint_monitor='val_eer'` grava artefato.

    Cobre de uma vez: o campo sobrevive ao filtro do TrainingConfig, o
    `ValidationEER` é registrado ANTES do ModelCheckpoint, e a saída de DUAS
    colunas (a que as arquiteturas do escopo produzem) vira um score por
    amostra em vez de 2N.
    """
    tf = pytest.importorskip("tensorflow")

    from app.core.config.settings import TrainingConfig
    from app.domain.models.training.trainer import ModelTrainer

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 4)).astype("float32")
    y = (X.sum(axis=1) > 0).astype("int32")
    Xv = rng.standard_normal((32, 4)).astype("float32")
    yv = (Xv.sum(axis=1) > 0).astype("int32")

    # DUAS saídas de propósito: é a forma real do escopo (softmax sobre
    # {bonafide, spoof}). Com uma só, o defeito nº 3 passa despercebido.
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = tmp_path / "models" / "best_checkpoint.weights.h5"
    checkpoint.parent.mkdir(parents=True)

    config = TrainingConfig(
        epochs=3,
        batch_size=16,
        early_stopping=False,
        reduce_lr_on_plateau=False,
        checkpoint_monitor="val_eer",
        verbose=0,
    )
    trainer = ModelTrainer(config, use_mixed_precision=False)
    trainer.train(
        model=modelo,
        train_data=(X, y),
        validation_data=(Xv, yv),
        checkpoint_path=str(checkpoint),
    )

    estado = checkpoint.with_suffix(".h5.best.json")
    assert estado.exists(), (
        "nenhum checkpoint selecionado — foi o sintoma dos defeitos 2 e 3: "
        "o Keras avisa 'Can save best model only with val_eer available' e o "
        "treino termina sem artefato"
    )
    gravado = json.loads(estado.read_text(encoding="utf-8"))
    assert gravado["monitor"] == "val_eer", (
        f"monitor gravado foi {gravado['monitor']!r} — o `checkpoint_monitor` "
        "não chegou ao ModelCheckpoint (defeito 1)"
    )
    assert 0.0 <= float(gravado["best"]) <= 1.0


@pytest.mark.parametrize(
    "monitor,modo_esperado",
    [("val_loss", "min"), ("val_eer", "min"), ("val_accuracy", "max")],
)
def test_modo_do_checkpoint_acompanha_o_monitor(monitor, modo_esperado):
    """EER é como a perda: menor é melhor. Um `mode` errado faria o
    ModelCheckpoint guardar a PIOR época."""
    modo = "min" if monitor in ("val_loss", "val_eer") else "max"
    assert modo == modo_esperado


def test_validation_eer_entra_na_frente_dos_consumidores():
    """Ordem dos callbacks: quem PUBLICA a métrica roda antes de quem a lê.

    O Keras passa o MESMO dicionário `logs` a todos os callbacks, em ordem de
    lista. Anexado ao fim, o ValidationEER escreveria `val_eer` depois de o
    ModelCheckpoint já ter decidido não salvar. Foi o que o smoke
    `_smoke_eer2` mediu: "Can save best model only with val_eer available",
    nenhum `best.json` e histórico sem a coluna.
    """
    import inspect

    from app.domain.models.training import trainer as mod

    # O método é `train`, não `train_model` — a primeira versão deste teste
    # inspecionava `train_model`, que não existe, e teria dado AttributeError
    # em vez de verificar coisa alguma.
    assert not hasattr(mod.ModelTrainer, "train_model"), (
        "surgiu um `train_model`: confirme qual método monta os callbacks"
    )
    fonte = inspect.getsource(mod.ModelTrainer.train)
    inicio = fonte.index("checkpoint_monitor")
    trecho = fonte[inicio : inicio + 1500]
    assert "callbacks.insert(" in trecho, (
        "ValidationEER precisa ser INSERIDO na frente; `append` o coloca "
        "depois do ModelCheckpoint e a métrica nunca é vista"
    )
    assert "ValidationEER" in trecho


def test_validation_eer_prediz_em_lotes():
    """`predict` de lote inteiro estoura a VRAM nas redes de forma de onda.

    A validação do benchmark são 1.456 janelas de 48.000 amostras (266 MB)
    atravessando o RawGAT-ST numa RTX 3060. Sem `batch_size` o smoke
    `_smoke_eer3` falhou — e em silêncio, porque a exceção era logada em DEBUG.
    """
    import inspect

    from app.domain.models.training.trainer import ValidationEER

    assert "batch_size" in inspect.signature(ValidationEER.__init__).parameters
    fonte = inspect.getsource(ValidationEER.on_epoch_end)
    assert "batch_size=self.batch_size" in fonte, "predict sem lote"
    assert "_save_logger.warning" in fonte, (
        "falha do val_eer não pode ser logada em DEBUG: quando o checkpoint "
        "monitora essa métrica, falhar aqui significa NENHUM artefato salvo"
    )
