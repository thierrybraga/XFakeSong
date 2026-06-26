"""Testes dos ajustes de retreino (P0 split por fonte + P2.0 ruído SNR).

Cobre:
- Split disjunto por grupo (anti-vazamento de fonte).
- Protocolo cross-generator (segurar gerador fora do treino).
- Ruído de augmentation calibrado por SNR alvo (casa treino↔teste).
"""

import numpy as np
import pytest


def _toy_dataset(n=600, seed=0):
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8, 4)).astype("float32")
    groups, y = [], []
    for i in range(n):
        r = i % 3
        if r == 0:
            groups.append("brspeech"); y.append(i % 2)
        elif r == 1:
            groups.append("cvpt"); y.append(0)
        else:
            groups.append("fkvoice"); y.append(1)
    return BenchmarkData(
        X=X, y=np.array(y), name="toy", groups=np.array(groups)
    )


def test_group_split_is_disjoint_by_source():
    """Nenhuma fonte/gerador aparece em mais de um conjunto."""
    bd = _toy_dataset()
    # Reconstrói índices por igualdade de linha para mapear grupos.
    Xtr, ytr, Xv, yv, Xte, yte = bd.stratified_split(42, group_split=True)
    assert len(ytr) and len(yv) and len(yte)

    def groups_of(Xsub):
        keys = {row.tobytes() for row in Xsub}
        return {
            bd.groups[i]
            for i, row in enumerate(bd.X)
            if row.tobytes() in keys
        }

    g_tr, g_v, g_te = groups_of(Xtr), groups_of(Xv), groups_of(Xte)
    assert g_tr.isdisjoint(g_te)
    assert g_tr.isdisjoint(g_v)
    assert g_v.isdisjoint(g_te)


def test_cross_generator_excludes_holdout_from_train():
    """O gerador segurado não aparece no treino e o teste tem ambas as classes."""
    bd = _toy_dataset()
    Xtr, ytr, Xv, yv, Xte, yte = bd.stratified_split(
        42, holdout_generator="fkvoice"
    )
    train_keys = {row.tobytes() for row in Xtr}
    held_in_train = any(
        bd.groups[i] == "fkvoice"
        for i, row in enumerate(bd.X)
        if row.tobytes() in train_keys
    )
    assert not held_in_train
    # Teste contém reais e fakes (classe inédita coberta).
    assert set(np.unique(yte).tolist()) == {0, 1}


def test_add_noise_hits_target_snr():
    """O ruído aditivo de treino bate o SNR alvo (mesma def. do benchmark)."""
    import tensorflow as tf

    from app.domain.models.training.augmentation import AudioAugmenter

    aug = AudioAugmenter({"snr_range_db": (15.0, 15.0)})
    rng = np.random.default_rng(1)
    x = tf.constant(rng.standard_normal(16000).astype("float32"))
    out, _ = aug._add_noise(x, tf.constant(1))
    noise = (out - x).numpy()
    sig_p = float(np.mean(x.numpy() ** 2))
    noi_p = float(np.mean(noise ** 2))
    snr = 10.0 * np.log10(sig_p / noi_p)
    assert snr == pytest.approx(15.0, abs=0.5)


def test_add_noise_matches_benchmark_awgn_definition():
    """A escala de ruído do treino coincide com benchmarks.data.add_awgn."""
    import tensorflow as tf

    from app.domain.models.training.augmentation import AudioAugmenter
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 16000)).astype("float32")
    snr = 10.0

    awgn = BenchmarkData.add_awgn(x, snr, seed=0)
    bench_noise_power = float(np.mean((awgn - x) ** 2))

    aug = AudioAugmenter({"snr_range_db": (snr, snr)})
    out, _ = aug._add_noise(tf.constant(x[0]), tf.constant(1))
    train_noise_power = float(np.mean((out.numpy() - x[0]) ** 2))

    # Mesma fórmula (sig_power/snr_lin) → potências de ruído ~iguais.
    assert train_noise_power == pytest.approx(bench_noise_power, rel=0.15)


# ----------------------- Follow-ups (P2/P3) -----------------------

def test_classical_features_include_rasta_plp():
    """O vetor tabular clássico passa a incluir estatísticas RASTA-PLP."""
    from benchmarks.data import _rasta_plp_stats, _to_tabular_features

    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 16000)).astype("float32")  # raw-audio-like
    feats = _to_tabular_features(X)
    rp = _rasta_plp_stats(X, n_plp=13)
    # RASTA-PLP contribui 2*n_plp colunas (média+desvio por coeficiente).
    assert rp.shape == (26, 4)
    assert np.isfinite(feats).all()
    # 11 stats + 26 MFCC + 26 RASTA-PLP = 63 (MFCC presente se librosa ok).
    assert feats.shape[1] >= 11 + 26


def test_robustness_fusion_weights_favor_robust_models():
    from app.domain.services.detection_service import (
        robustness_fusion_weights,
    )

    names = ["Conformer", "SVM", "AASIST", "RawNet2"]
    w = robustness_fusion_weights(names)
    assert sum(w) == pytest.approx(1.0)
    wmap = dict(zip(names, w))
    assert wmap["Conformer"] > wmap["SVM"]
    assert wmap["AASIST"] > wmap["RawNet2"]


def test_robustness_weights_unknown_model_is_neutral():
    from app.domain.services.detection_service import (
        robustness_fusion_weights,
    )

    w = robustness_fusion_weights(["FooBar", "BazQux"])
    assert w == pytest.approx([0.5, 0.5])  # ambos neutros → uniforme


def test_pruning_degrades_gracefully_without_tfmot():
    """Sem tfmot, apply_pruning retorna None sem levantar exceção."""
    from app.domain.models.training import magnitude_pruning as mp

    if mp.is_tfmot_available():
        pytest.skip("tfmot instalado — caminho de degradação não se aplica")
    assert mp.apply_pruning(object(), target_sparsity=0.5) is None
    assert mp.pruning_callbacks() == []
    with pytest.raises(ValueError):
        mp.apply_pruning(object(), target_sparsity=1.5)
