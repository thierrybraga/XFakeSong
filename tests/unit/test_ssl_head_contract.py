"""Testes do contrato de embedding SSL (2026-07-15) e do augmentation dinâmico.

Cobre:
- construção da cabeça por contrato (weighted vs last/legado);
- paridade save/load do state_dict entre runner e wrapper (mesma fonte);
- resolve_embedding_config com fallback legado;
- pool_hidden_states (shapes weighted/meanstd e last/mean);
- protocolo do benchmark: AASIST/RawGAT-ST usam augmenter dinâmico no lugar
  da cópia AWGN estática.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.domain.models.inference.ssl_head import (  # noqa: E402
    build_ssl_classifier,
    pool_hidden_states,
    resolve_embedding_config,
)

WEIGHTED_CFG = {
    "target_samples": 64000,
    "layer_pooling": "weighted",
    "time_pooling": "meanstd",
    "num_layers": 13,
    "feature_dim": 1536,
}


class _FakeOutputs:
    def __init__(self, hidden_states=None, last_hidden_state=None):
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state


def test_weighted_head_forward_and_state_dict_roundtrip():
    head = build_ssl_classifier(WEIGHTED_CFG, dropout=0.2)
    x = torch.randn(4, 13, 1536)
    out = head(x)
    assert tuple(out.shape) == (4, 2)

    # Paridade treino↔inferência: reconstruir pela MESMA função e carregar
    # o state_dict (é o que o TorchSSLOriginalModel faz com o .pt).
    clone = build_ssl_classifier(WEIGHTED_CFG, dropout=0.2)
    clone.load_state_dict(head.state_dict())
    head.eval(); clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(head(x), clone(x))


def test_legacy_head_matches_old_sequential_layout():
    cfg = resolve_embedding_config({}, backbone_hidden_size=768)
    assert cfg["layer_pooling"] == "last"
    assert cfg["target_samples"] == 16000
    assert cfg["feature_dim"] == 768
    head = build_ssl_classifier(cfg, dropout=0.2)
    # Layout legado: Dropout/Linear(768,256)/ReLU/Dropout/Linear(256,2) —
    # nomes de parâmetros compatíveis com checkpoints antigos ("1.weight"...).
    names = {name for name, _ in head.named_parameters()}
    assert names == {"1.weight", "1.bias", "4.weight", "4.bias"}


def test_pool_hidden_states_shapes():
    hidden = tuple(torch.randn(2, 50, 768) for _ in range(13))
    out = pool_hidden_states(_FakeOutputs(hidden_states=hidden), WEIGHTED_CFG)
    assert tuple(out.shape) == (2, 13, 1536)

    legacy = resolve_embedding_config({}, backbone_hidden_size=768)
    out2 = pool_hidden_states(
        _FakeOutputs(last_hidden_state=torch.randn(2, 50, 768)), legacy
    )
    assert tuple(out2.shape) == (2, 768)


def test_resolve_embedding_config_reads_checkpoint_contract():
    cfg = resolve_embedding_config(
        {"embedding_config": dict(WEIGHTED_CFG)}, backbone_hidden_size=768
    )
    assert cfg["layer_pooling"] == "weighted"
    assert cfg["num_layers"] == 13
    assert cfg["feature_dim"] == 1536
    assert cfg["target_samples"] == 64000


def test_protocol_swaps_static_copy_for_dynamic_augmenter():
    """AASIST/RawGAT-ST com `architecture_specific_augmentation=True`:
    sem cópia AWGN estática, augmenter dinâmico ligado. Este é o regime dos
    resultados promovidos (retrain_weak4_20260715); desde o protocolo v2 ele
    é OPT-IN — o default comparável usa cópia estática uniforme p/ todas as
    arquiteturas e o regime dinâmico deve ser reportado como ablação."""
    pytest.importorskip("tensorflow")
    from benchmarks.config import BenchmarkConfig
    from benchmarks.runner import _prepare_protocol_splits

    rng = np.random.default_rng(0)
    def _split(n):
        return (rng.standard_normal((n, 16000)) * 0.1).astype("float32")

    raw = (_split(8), np.array([0, 1] * 4), _split(4),
           np.array([0, 1] * 2), _split(4), np.array([0, 1] * 2))
    cfg = BenchmarkConfig(
        bootstrap_ci_samples=0, architecture_specific_augmentation=True
    )

    splits = _prepare_protocol_splits("AASIST", cfg, raw)
    protocol = splits[7]
    assert protocol["training_augmentation_domain"] == "waveform_dynamic_augmenter"
    # Sem cópia estática: fit == treino limpo.
    assert protocol["fit_train_samples"] == protocol["clean_train_samples"]

    # Mesmo com o flag ligado, arquiteturas fora do conjunto dinâmico seguem
    # o protocolo padrão (cópia AWGN estática).
    splits2 = _prepare_protocol_splits("Conformer", cfg, raw)
    protocol2 = splits2[7]
    assert protocol2["training_augmentation_domain"] == "waveform"
    assert protocol2["fit_train_samples"] == 2 * protocol2["clean_train_samples"]

    # Default (flag desligado): AASIST também usa a cópia estática uniforme.
    cfg_default = BenchmarkConfig(bootstrap_ci_samples=0)
    protocol3 = _prepare_protocol_splits("AASIST", cfg_default, raw)[7]
    assert protocol3["training_augmentation_domain"] == "waveform"
