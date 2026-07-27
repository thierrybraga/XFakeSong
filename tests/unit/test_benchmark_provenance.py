"""Proveniência e ambiente registrados junto com os resultados.

Duas lacunas fechadas em 2026-07-27:

- os rótulos `variant`/`family`/`runner` do manifesto existiam em
  `benchmarks/config.py` mas **não chegavam a nenhum artefato** — e vários
  estavam defasados após mudanças de arquitetura (ex.: `ast_vit_base_scratch`
  quando o AST já partia de pesos AudioSet);
- o snapshot de ambiente não gravava o **commit**, então os números não eram
  rastreáveis até a revisão do código que os produziu.
"""

from __future__ import annotations

import pytest

from benchmarks.config import OFFICIAL_TCC_MODEL_MANIFEST
from benchmarks.runner import (
    _architecture_provenance,
    _env_snapshot,
    _git_provenance,
    _library_versions,
    _pretrained_checkpoints,
)


def test_every_official_model_declares_a_variant():
    for item in OFFICIAL_TCC_MODEL_MANIFEST:
        assert item.get("variant"), item["benchmark_name"]


def test_variant_labels_are_not_stale():
    """Rótulos que contradizem a arquitetura efetivamente treinada."""
    variants = {i["result_key"]: i["variant"] for i in OFFICIAL_TCC_MODEL_MANIFEST}

    # O AST parte de pesos AudioSet — não é mais "scratch".
    assert "scratch" not in variants["AST"]
    assert "audioset" in variants["AST"].lower()

    # `temporal_pool_stride` só vale nas variantes LEGADAS do RawGAT-ST.
    assert "stride" not in variants["RawGAT-ST"]

    # RawNet2 tem duas configurações homônimas (verificação de locutor vs.
    # baseline anti-spoofing): o rótulo precisa dizer qual.
    assert any(
        token in variants["RawNet2"].lower()
        for token in ("speakerverification", "antispoofing")
    ), variants["RawNet2"]


def test_provenance_reaches_results_for_official_and_extended():
    official = _architecture_provenance("SpectrogramTransformer")
    assert official["result_key"] == "AST"
    assert official["scope"] == "official"
    assert official["variant"]

    extended = _architecture_provenance("Ensemble")
    assert extended["scope"] == "extended"

    assert _architecture_provenance("Arquitetura Inexistente") == {}


def test_git_provenance_records_commit_and_dirty_flag():
    git = _git_provenance()
    if not git.get("available"):
        pytest.skip("repositório git indisponível neste ambiente")
    assert len(git["commit"]) == 40
    assert git["commit_short"] == git["commit"][:12]
    # `dirty` precisa existir: um run com árvore suja não é reproduzível a
    # partir do commit sozinho, e o artefato tem de dizer isso.
    assert isinstance(git["dirty"], bool)


def test_library_versions_cover_numeric_stack():
    versions = _library_versions()
    for library in ("tensorflow", "numpy", "sklearn"):
        assert library in versions


def test_pretrained_checkpoints_are_declared():
    """AST/WavLM/HuBERT partem de pesos externos: o id é parte do experimento."""
    checkpoints = _pretrained_checkpoints()
    assert "MIT/ast-finetuned-audioset" in (
        checkpoints.get("SpectrogramTransformer") or ""
    )
    assert (checkpoints.get("WavLM") or "").startswith("microsoft/wavlm")
    assert (checkpoints.get("HuBERT") or "").startswith("facebook/hubert")


def test_env_snapshot_carries_provenance_blocks():
    env = _env_snapshot()
    for key in ("git", "libraries", "pretrained_checkpoints", "python", "platform"):
        assert key in env
