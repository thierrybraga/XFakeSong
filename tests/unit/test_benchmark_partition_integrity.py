"""O benchmark nao pode repetir amostras entre treino, validacao e teste.

A checagem de indices sozinha so prova que as particoes nao compartilham LINHAS.
Estes testes exercitam a verificacao de CONTEUDO — amostra, enunciado, locutor e
texto — que `BenchmarkData` aplica a todo split efetivamente usado, incluindo os
produzidos por protocolos alternativos.

Ver docs/data/dataset-protocol.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.data import BenchmarkData

SPEAKERS, SENTENCES = 24, 10


def _corpus(
    predefined: dict[str, np.ndarray] | None = None,
    *,
    texto_por_locutor: bool = True,
) -> BenchmarkData:
    """Grid pareado: cada locutor le as mesmas frases nas duas classes.

    Com `texto_por_locutor=True` o bloco de frases acompanha o bloco de
    locutores — a disjuncao dupla do artefato real. Com `False`, todos os
    locutores compartilham as mesmas frases, que e o caso em que particionar so
    por locutor deixa o texto vazar.
    """
    speakers, texts, labels = [], [], []
    for s in range(SPEAKERS):
        for t in range(SENTENCES):
            for label in (0, 1):
                speakers.append(f"ptpair:S{s:03d}")
                bloco = s // (SPEAKERS // 3) if texto_por_locutor else 0
                texts.append(f"ptpair:t{bloco}_{t:03d}")
                labels.append(label)
    y = np.asarray(labels)
    n = len(y)
    return BenchmarkData(
        X=np.zeros((n, 50, 1), dtype="float32"),
        y=y,
        groups=np.asarray(["ptpair"] * n),
        speakers=np.asarray(speakers),
        speaker_known=np.ones(n, dtype=bool),
        texts=np.asarray(texts),
        generators=np.asarray(["bonafide" if v == 0 else "xtts_v2" for v in y]),
        generator_known=np.ones(n, dtype=bool),
        sample_paths=np.asarray([f"s{i}.wav" for i in range(n)]),
        predefined_split_indices=predefined,
    )


def _blocos() -> dict[str, np.ndarray]:
    """Particao alinhada aos blocos de locutor e de frase."""
    por_locutor = SENTENCES * 2
    corte = SPEAKERS // 3
    return {
        "train": np.arange(0, corte * por_locutor),
        "val": np.arange(corte * por_locutor, 2 * corte * por_locutor),
        "test": np.arange(2 * corte * por_locutor, SPEAKERS * por_locutor),
    }


# ---------------------------------------------------------------------------
# A particao selada passa; violacoes reprovam
# ---------------------------------------------------------------------------


def test_particao_disjunta_e_aceita() -> None:
    data = _corpus(_blocos())
    _, ytr, _, yva, _, yte = data.stratified_split()
    assert len(ytr) and len(yva) and len(yte)
    assert set(np.bincount(yte)) == {len(yte) // 2}


def test_locutor_repetido_entre_particoes_reprova() -> None:
    blocos = _blocos()
    # Move um enunciado inteiro (real+clone) do teste para o treino: o locutor
    # passa a existir nos dois lados.
    blocos["train"] = np.concatenate([blocos["train"], blocos["test"][:2]])
    data = _corpus(blocos)
    with pytest.raises(ValueError, match="Vazamento de locutor|sobrepostas"):
        data.stratified_split()


def test_texto_repetido_entre_particoes_reprova() -> None:
    """Locutores disjuntos nao bastam: a mesma frase nao pode atravessar."""
    data = _corpus(_blocos(), texto_por_locutor=False)
    with pytest.raises(ValueError, match="Vazamento de texto"):
        data.stratified_split()


def test_amostra_repetida_entre_particoes_reprova() -> None:
    data = _corpus(_blocos())
    caminhos = np.asarray(data.sample_paths).copy()
    # A primeira amostra do teste passa a ter a identidade de uma do treino.
    caminhos[_blocos()["test"][0]] = caminhos[0]
    data.sample_paths = caminhos
    with pytest.raises(ValueError, match="Vazamento de amostra"):
        data.stratified_split()


# ---------------------------------------------------------------------------
# Protocolos alternativos nao podem descartar a particao selada em silencio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speaker_split": True},
        {"group_split": True},
        {"holdout_speaker": "ptpair:S001"},
    ],
)
def test_protocolo_alternativo_recusa_sobrescrever_particao_selada(kwargs) -> None:
    data = _corpus(_blocos())
    with pytest.raises(ValueError, match="Protocolo alternativo recusado"):
        data.stratified_split(**kwargs)


def test_protocolo_alternativo_aceito_quando_explicito() -> None:
    data = _corpus(_blocos())
    _, ytr, _, yva, _, yte = data.stratified_split(
        speaker_split=True, preserve_predefined=False
    )
    assert len(ytr) and len(yva) and len(yte)


# ---------------------------------------------------------------------------
# Cross-generator
# ---------------------------------------------------------------------------


def test_cross_generator_resolve_contra_generators_nao_contra_source() -> None:
    """Regressao: o holdout era resolvido contra `groups` (a fonte).

    Num corpus pareado as duas classes compartilham a fonte de proposito, entao
    o protocolo simplesmente nao encontrava o gerador.
    """
    data = _corpus()
    with pytest.raises(ValueError, match="Gerador holdout inexistente"):
        data.stratified_split(
            holdout_generator="griffin_lim", preserve_predefined=False
        )


def test_cross_generator_recusa_corpus_de_gerador_unico() -> None:
    data = _corpus()
    with pytest.raises(ValueError, match="cross-generator inaplicavel"):
        data.stratified_split(holdout_generator="xtts_v2", preserve_predefined=False)


def test_cross_generator_funciona_com_dois_geradores() -> None:
    data = _corpus()
    # dtype largo: o array original e `<U8` e truncaria o nome do gerador.
    generators = np.asarray(data.generators, dtype="U32").copy()
    falsas = np.flatnonzero(data.y == 1)
    generators[falsas[: len(falsas) // 2]] = "griffin_lim"
    data.generators = generators

    _, ytr, _, _, _, yte = data.stratified_split(
        holdout_generator="griffin_lim", preserve_predefined=False
    )
    assert set(np.unique(yte).tolist()) == {0, 1}
    assert len(ytr)


# ---------------------------------------------------------------------------
# Holdout de falante
# ---------------------------------------------------------------------------


def test_unseen_speaker_mantem_teste_balanceado() -> None:
    """No corpus pareado o locutor segurado ja traz as duas classes.

    Completar com reais do restante desbalanceava o teste (medido: 30 reais
    para 10 falsas).
    """
    data = _corpus()
    _, _, _, _, _, yte = data.stratified_split(
        holdout_speaker="ptpair:S001", preserve_predefined=False
    )
    reais, falsas = int((yte == 0).sum()), int((yte == 1).sum())
    assert reais == falsas, (reais, falsas)


def test_unseen_speaker_nao_deixa_o_locutor_no_treino() -> None:
    data = _corpus()
    data.stratified_split(holdout_speaker="ptpair:S001", preserve_predefined=False)
    indices = data.last_split_indices
    speakers = np.asarray(data.speakers)
    assert "ptpair:S001" not in set(speakers[indices["train"]].tolist())
    assert "ptpair:S001" in set(speakers[indices["test"]].tolist())
