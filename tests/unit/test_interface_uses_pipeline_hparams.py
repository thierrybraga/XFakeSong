"""A interface tem de propor a configuração que o benchmark treina.

O CLAUDE.md já documenta que os hiperparâmetros vivem em três lugares
(`registry.default_params`, o `create_model` de cada arquitetura e
`benchmarks/planning.py`) e alerta para o drift. A interface era um QUARTO:
`load_defaults` caía em literais próprios — `batch_size=32, epochs=10,
learning_rate=0.001` — para TODA arquitetura. AASIST, Conformer e Sonic Sleuth
apareciam idênticas, e nenhuma batia com o pipeline.

Pior que os literais: as linhas semeadas em `architecture_configs` eram
tratadas como customização do usuário e venciam o plano. A do
SpectrogramTransformer trazia `pretrained=False`, então a interface propunha
treinar do zero um modelo que o benchmark treina a partir dos pesos AudioSet.
"""

from __future__ import annotations

import pytest

from app.interfaces.gradio.utils.hyperparameters import (
    get_interface_hyperparameters,
)
from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS, _compact

# Chaves do plano que representam decisão de TREINO e precisam chegar à UI.
# (`notes`, `model_family` e afins são metadados, não hiperparâmetros.)
_CHAVES_DE_TREINO = (
    "batch_size",
    "learning_rate",
    "dropout_rate",
    "l2_reg_strength",
    "weight_decay",
    "optimizer",
    "scheduler",
    "use_augmentation",
    "use_mixed_precision",
    "pretrained",
    "epochs",
)


@pytest.mark.parametrize("arquitetura", [
    "AASIST", "Conformer", "RawNet2", "RawGAT-ST", "MultiscaleCNN",
    "Hybrid CNN-Transformer", "SpectrogramTransformer",
])
def test_ui_propoe_a_configuracao_do_pipeline(arquitetura):
    plano = NEURAL_BENCHMARK_HPARAMS[_compact(arquitetura)]
    ui = get_interface_hyperparameters(arquitetura, None)

    for chave in _CHAVES_DE_TREINO:
        if plano.get(chave) is None:
            continue
        assert ui.get(chave) == plano[chave], (
            f"{arquitetura}: a UI propõe {chave}={ui.get(chave)!r} e o "
            f"benchmark treina com {plano[chave]!r}"
        )


def test_plano_vence_a_linha_semeada_do_banco():
    """`architecture_configs` é semeado pela app, não escrito pelo usuário."""
    semeado = {
        "learning_rate": 5e-05,   # o plano usa 1e-05
        "pretrained": False,      # o plano parte dos pesos AudioSet
        "patch_size": [16, 16],   # estrutura: o plano NÃO define
        "embed_dim": 768,
    }
    ui = get_interface_hyperparameters("SpectrogramTransformer", semeado)

    assert ui["learning_rate"] == 1e-05
    assert ui["pretrained"] is True, (
        "a UI propunha treinar do zero um modelo que o pipeline treina por "
        "transferencia"
    )
    # o que o plano não define continua vindo do banco/registry
    assert tuple(ui["patch_size"]) == (16, 16)
    assert ui["embed_dim"] == 768


def test_escopo_estendido_nao_quebra_sem_entrada_no_plano():
    """Sonic Sleuth, EfficientNet-LSTM, Ensemble, WavLM e HuBERT."""
    for arquitetura in ("Sonic Sleuth", "EfficientNet-LSTM", "Ensemble",
                        "WavLM", "HuBERT"):
        ui = get_interface_hyperparameters(arquitetura, None)
        assert ui["batch_size"] > 0
        assert ui["epochs"] > 0
        assert ui["learning_rate"] > 0


def test_arquiteturas_deixam_de_ter_defaults_identicos():
    """O sintoma original: tudo com 32/10/0.001."""
    vistos = {
        arq: (
            get_interface_hyperparameters(arq, None)["batch_size"],
            get_interface_hyperparameters(arq, None)["learning_rate"],
        )
        for arq in ("AASIST", "Conformer", "RawNet2", "SpectrogramTransformer")
    }
    assert len(set(vistos.values())) > 1, (
        f"todas as arquiteturas ainda propõem a mesma configuracao: {vistos}"
    )
