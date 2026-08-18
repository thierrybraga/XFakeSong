"""Escopos do benchmark: oficial x estendido, e o que cada um pode alegar.

SUJEITO: `benchmarks/config.py` — que os dois escopos sejam disjuntos e
completos, que o estendido se declare NÃO acadêmico, e que a política de limiar
e o perfil de latência sejam reportados separadamente por escopo.
"""

from argparse import Namespace

import numpy as np

from benchmarks.config import (
    DOCKER_TRAINING_ARCHITECTURES,
    EXTENDED_MODEL_MANIFEST,
    MODEL_FAMILIES,
    OFFICIAL_TCC_MODEL_MANIFEST,
)
from benchmarks.efficiency import measure_latency_profile
from benchmarks.evaluate import evaluate_scores
from scripts.training.train_by_family import build_command


def _family_args(family: str) -> Namespace:
    return Namespace(
        family=family, config=None, dataset=None, models=None, out=None,
        epochs=None, batch_size=None, device_profile=None, latency_runs=None,
        timeout_min=None, snr=None, resume=False, seeds=[42, 43],
        test_lock="sealed.json", plan_only=True, api=False,
        no_optimize_hparams=False, verbose=False,
    )


def test_official_and_extended_scopes_are_disjoint_and_complete():
    official = {item["benchmark_name"] for item in OFFICIAL_TCC_MODEL_MANIFEST}
    extended = {item["benchmark_name"] for item in EXTENDED_MODEL_MANIFEST}
    assert official == set(DOCKER_TRAINING_ARCHITECTURES)
    assert official.isdisjoint(extended)
    # As duas ajustadas (front-end destravado + grafo AASIST) SAIRAM do escopo
    # oficial em 2026-08-11: os sistemas de topo do ASVspoof 5 usam SSL
    # CONGELADO, e o resultado de referencia daquela receita usa wav2vec2
    # XLS-R, nao WavLM/HuBERT base. As entradas `Original` ja sao a
    # configuracao documentada. Ver benchmarks/config.py.
    assert set(MODEL_FAMILIES["ssl-pretrained"]) == {
        "WavLM Original", "HuBERT Original"
    }


def test_family_wrapper_uses_canonical_dataset_and_protocol_controls():
    # O dataset canonico passou a ser o do Protocolo de Dataset (CETUC pareado com clones
    # XTTS-v2, disjuncao dupla locutor x frase) em 26/07/2026; os NPZ v2 foram
    # apagados. Ver docs/data/dataset-protocol.md.
    cmd = build_command(_family_args("ssl-pretrained"))
    joined = " ".join(str(value) for value in cmd)
    assert "benchmark_dataset.npz" in joined
    assert "WavLM Original" in joined and "HuBERT Original" in joined
    assert "--academic-protocol" in cmd
    assert cmd[cmd.index("--scope") + 1] == "official"
    assert cmd[cmd.index("--seeds") + 1:cmd.index("--test-lock")] == ["42", "43"]


def test_extended_family_is_explicitly_non_academic():
    cmd = build_command(_family_args("extended"))
    assert "--no-academic-protocol" in cmd
    assert "--no-optimize-hparams" in cmd
    assert cmd[cmd.index("--scope") + 1] == "extended"


def test_threshold_policies_are_reported_separately():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    result = evaluate_scores(y, scores, calibrated_threshold=0.7)
    assert result["decision_threshold"] == 0.5
    assert result["accuracy_at_fixed_threshold"] == result["accuracy"]
    assert "accuracy_at_eer_oracle" in result
    assert result["calibrated_threshold"] == 0.7
    assert "accuracy_at_calibrated_threshold" in result


def test_latency_profile_exposes_protocol_and_distribution():
    profile = measure_latency_profile(lambda batch: batch + 1, np.zeros(8), runs=4)
    assert profile["status"] == "ok"
    assert profile["component"] == "model_forward_only"
    assert profile["measured_runs"] == 4
    assert profile["p95_ms"] >= 0
    assert profile["includes_frontend"] is False
