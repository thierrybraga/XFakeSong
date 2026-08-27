"""O split da interface não pode repetir amostra nem falante entre partições.

A aba de Dataset chamava `create_splits(train, val, test)` sem
`speaker_disjoint`, e o default é `False` — split estratificado apenas por
classe. O mesmo locutor caía em treino, validação e teste, e a métrica passava
a medir memorização de timbre em vez de detecção de síntese. É exatamente o
vazamento que o protocolo do benchmark existe para impedir
(`benchmarks/runner._audit_split_overlap`), e que motivou o descarte do dataset
anterior.

Aqui a verificação acontece no momento da CRIAÇÃO, quando ainda dá para refazer.
"""

from __future__ import annotations

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")


def _escrever_wav(caminho, semente: int, duracao_s: float = 0.4, sr: int = 16000):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semente)
    onda = (0.05 * rng.normal(size=int(sr * duracao_s))).astype("float32")
    soundfile.write(str(caminho), onda, sr)
    return onda


def _montar_splits(raiz, mapa):
    """mapa: {split: {classe: [(nome, semente), ...]}}"""
    for split, classes in mapa.items():
        for classe, itens in classes.items():
            for nome, semente in itens:
                _escrever_wav(raiz / split / classe / f"{nome}.wav", semente)


def test_auditoria_detecta_a_mesma_amostra_em_dois_splits(tmp_path, monkeypatch):
    """Repetição literal de áudio — o caso mais grave."""
    import scripts.dataset.preprocess_dataset as pp

    # a MESMA semente gera o mesmo PCM: amostra duplicada entre treino e teste
    _montar_splits(tmp_path, {
        "train": {"real": [("a", 1)], "fake": [("b", 2)]},
        "val": {"real": [("c", 3)], "fake": [("d", 4)]},
        "test": {"real": [("e", 1)], "fake": [("f", 6)]},  # "e" == "a"
    })

    relatorio = pp.audit_splits(tmp_path)

    assert relatorio["available"] is True
    assert relatorio["content_sha256"]["overlap"]["train_test"] == 1
    assert relatorio["passed"] is False, (
        "a auditoria aceitou o mesmo audio em treino e teste"
    )


def test_auditoria_aprova_particoes_realmente_disjuntas(tmp_path):
    import scripts.dataset.preprocess_dataset as pp

    _montar_splits(tmp_path, {
        "train": {"real": [("a", 1)], "fake": [("b", 2)]},
        "val": {"real": [("c", 3)], "fake": [("d", 4)]},
        "test": {"real": [("e", 5)], "fake": [("f", 6)]},
    })

    relatorio = pp.audit_splits(tmp_path)

    assert relatorio["passed"] is True
    assert all(v == 0 for v in relatorio["content_sha256"]["overlap"].values())
    assert relatorio["counts"] == {"train": 2, "val": 2, "test": 2}


def test_auditoria_reporta_ausencia_de_splits(tmp_path):
    """Sem partição alguma, o relatório informa em vez de levantar."""
    import scripts.dataset.preprocess_dataset as pp

    relatorio = pp.audit_splits(tmp_path / "inexistente")
    assert relatorio["available"] is False
    assert relatorio["passed"] is False
    assert "reason" in relatorio


def test_interface_pede_disjuncao_por_falante_por_padrao():
    """A aba chamava `create_splits` sem o parâmetro — default `False`."""
    import inspect

    from app.interfaces.gradio.tabs import dataset_management

    fonte = inspect.getsource(dataset_management)

    assert "speaker_disjoint=bool(speaker_disjoint)" in fonte, (
        "a interface voltou a criar splits sem passar speaker_disjoint"
    )
    assert 'label="Disjunção por falante"' in fonte
    # o controle precisa vir LIGADO: o caminho de menor esforco tem de ser o
    # correto, nao o que vaza
    trecho = fonte[fonte.index("pp_speaker_disjoint = gr.Checkbox"):][:200]
    assert "value=True" in trecho, "o controle precisa vir marcado por padrao"


def test_interface_audita_o_split_depois_de_criar():
    """Criar sem verificar deixa o vazamento passar até o treino."""
    import inspect

    from app.interfaces.gradio.tabs import dataset_management

    fonte = inspect.getsource(dataset_management)
    assert fonte.count("_render_auditoria(pp.audit_splits())") >= 2, (
        "a auditoria precisa rodar tanto em 'Criar Splits' quanto no "
        "'Pipeline Completo'"
    )


def test_create_splits_expoe_o_parametro_de_disjuncao():
    import inspect

    import scripts.dataset.preprocess_dataset as pp

    assinatura = inspect.signature(pp.create_splits)
    assert "speaker_disjoint" in assinatura.parameters
