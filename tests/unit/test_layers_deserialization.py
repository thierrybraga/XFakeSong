"""`architectures/layers.py`: porta de entrada e registro das camadas no Keras.

SUJEITO: o que os artefatos `.keras` já gravados exigem deste módulo. Eles citam
`app.domain.models.architectures.layers` no campo `module`, mas o Keras 3 resolve
classe própria SÓ pelo registro (`registered_name`), então é o `import layers` que
as torna carregáveis. Dividir as definições em módulos temáticos não pode mudar
isso: as sete classes abaixo estão nos modelos em produção, e duas foram gravadas
sem pacote, dependendo das chaves sem prefixo. `AudioNormalizationLayer` é a única
que `layers` não registra sozinho: a chave vem de `rawnet2.py`, que o carregador
sempre importa antes de abrir um `.keras`.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")

from app.domain.models.architectures import layers as porta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODULO_GRAVADO = "app.domain.models.architectures.layers"

# (classe, registered_name) como estão no config.json dos .keras em data/models.
GRAVADAS_EM_PRODUCAO = (
    ("ASTInputNormalization", "XFakeSong>ASTInputNormalization"),
    ("AudioFeatureNormalization", "XFakeSong>AudioFeatureNormalization"),
    ("AxisMaxAbsLayer", "XFakeSong>AxisMaxAbsLayer"),
    ("MagnitudeLayer", "XFakeSong>MagnitudeLayer"),
    ("PreEmphasisLayer", "PreEmphasisLayer"),
)

_PROCESSO_LIMPO = """
import json, sys
import keras
from app.domain.models.architectures import layers

ARGUMENTOS = {"AxisMaxAbsLayer": {"axis": 1}}

def abre(classe, registrado):
    original = getattr(layers, classe)
    obj = keras.saving.deserialize_keras_object({
        "module": "%s",
        "class_name": classe,
        "config": original(**ARGUMENTOS.get(classe, {})).get_config(),
        "registered_name": registrado,
    })
    assert type(obj) is original, (classe, type(obj))

for classe, registrado in json.loads(sys.argv[1]):
    abre(classe, registrado)
# A única que depende do rawnet2 registrar, como no carregador.
from app.domain.models.architectures import rawnet2  # noqa: F401
abre("AudioNormalizationLayer", "AudioNormalizationLayer")
print("ok")
""" % MODULO_GRAVADO


def test_config_gravada_desserializa_num_processo_que_so_importa_a_porta():
    """Sem depender de `rawnet2.py` nem de outro módulo ter registrado antes."""
    saida = subprocess.run(
        [sys.executable, "-c", _PROCESSO_LIMPO, json.dumps(GRAVADAS_EM_PRODUCAO)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
    )
    assert saida.returncode == 0, saida.stderr[-1500:]
    assert saida.stdout.strip().endswith("ok")


def test_perda_gravada_no_conformer_tambem_desserializa():
    import keras

    objeto = keras.saving.deserialize_keras_object(
        {
            "module": MODULO_GRAVADO,
            "class_name": "SparseLabelSmoothingCrossEntropy",
            "config": porta.SparseLabelSmoothingCrossEntropy().get_config(),
            "registered_name": "XFakeSong>SparseLabelSmoothingCrossEntropy",
        }
    )
    assert type(objeto) is porta.SparseLabelSmoothingCrossEntropy


def test_a_porta_expoe_toda_definicao_dos_modulos_tematicos():
    """Quem importa de `layers` (as arquiteturas, o carregador) enxerga tudo."""
    from app.domain.models.architectures import (
        block_layers,
        graph_layers,
        head_layers,
        plumbing_layers,
        sinc_layers,
        spectral_layers,
    )

    definidos = {}
    for modulo in (
        block_layers,
        graph_layers,
        head_layers,
        plumbing_layers,
        sinc_layers,
        spectral_layers,
    ):
        for nome, valor in vars(modulo).items():
            if (
                (inspect.isclass(valor) or inspect.isfunction(valor))
                and valor.__module__ == modulo.__name__
            ):
                assert nome not in definidos, f"{nome} definido em dois módulos"
                definidos[nome] = valor

    assert set(porta.__all__) == set(definidos)
    for nome, valor in definidos.items():
        assert getattr(porta, nome) is valor, nome


def test_classe_legada_sem_uso_no_codigo_segue_registrada():
    """As de grafo só existem para artefatos antigos: nada as importa, e sem o
    registro o carregamento desses arquivos falharia só em quem precisa deles."""
    registro = tf.keras.utils.get_custom_objects()
    for chave, classe in (
        ("XFakeSong>GraphPoolLayer", porta.GraphPoolLayer),
        ("XFakeSong>HSGALLayer", porta.HSGALLayer),
        ("XFakeSong>MasterNodeSeed", porta.MasterNodeSeed),
        ("XFakeSong>ResidualBlock2D", porta.ResidualBlock2D),
    ):
        assert registro[chave] is classe, chave
