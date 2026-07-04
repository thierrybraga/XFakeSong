"""Testes do módulo XAI (app/core/xai): Grad-CAM, SHAP e contrato tabular."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.xai.tabular import (
    N_FEATURES,
    extract_sklearn_estimator,
    feature_group,
    split_sklearn_pipeline,
    tabular_feature_names,
)


class TestTabularContract:
    def test_feature_names_have_canonical_length_and_order(self):
        names = tabular_feature_names()
        assert len(names) == N_FEATURES == 63
        # blocos na ordem: 11 temporais, 26 MFCC, 26 RASTA-PLP
        assert names[0] == "média"
        assert names[10] == "ZCR"
        assert names[11].startswith("MFCC1 ")
        assert names[37].startswith("RASTA-PLP1 ")

    def test_feature_group_families(self):
        groups = {feature_group(n) for n in tabular_feature_names()}
        assert groups == {"Temporal", "MFCC", "RASTA-PLP"}

    def test_extract_estimator_from_pipeline_and_dict(self):
        sklearn = pytest.importorskip("sklearn")
        del sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5))]
        ).fit(X, y)

        est = extract_sklearn_estimator(pipe)
        assert isinstance(est, RandomForestClassifier)
        est_dict = extract_sklearn_estimator({"model": pipe, "meta": 1})
        assert isinstance(est_dict, RandomForestClassifier)
        assert extract_sklearn_estimator({"meta": 1}) is None

    def test_split_pipeline_applies_preprocessing(self):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(1)
        X = rng.normal(loc=10.0, scale=3.0, size=(60, 4))
        y = (X[:, 1] > 10).astype(int)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=5))]
        ).fit(X, y)

        transform, est = split_sklearn_pipeline(pipe)
        Xt = transform(X)
        assert Xt.shape == X.shape
        assert abs(float(Xt.mean())) < 1e-6  # StandardScaler aplicado
        assert isinstance(est, RandomForestClassifier)

        # estimador direto: transformação identidade
        ident, est2 = split_sklearn_pipeline(est)
        np.testing.assert_array_equal(ident(X), X)
        assert est2 is est


class TestGradCAM:
    @pytest.fixture(scope="class")
    def tiny_conv_model(self):
        tf = pytest.importorskip("tensorflow")
        inputs = tf.keras.Input(shape=(16, 12, 1))
        x = tf.keras.layers.Conv2D(4, 3, padding="same", name="conv_a")(inputs)
        x = tf.keras.layers.MaxPooling2D(2, name="pool")(x)
        x = tf.keras.layers.Conv2D(8, 3, padding="same", name="conv_b")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="head")(x)
        return tf.keras.Model(inputs, outputs)

    def test_find_last_conv_layer(self, tiny_conv_model):
        from app.core.xai.gradcam import find_last_conv_layer

        assert find_last_conv_layer(tiny_conv_model) == "conv_b"

    def test_gradcam_shapes_and_range(self, tiny_conv_model):
        from app.core.xai.gradcam import compute_gradcam, resize_heatmap

        rng = np.random.default_rng(42)
        batch = rng.normal(size=(3, 16, 12, 1)).astype("float32")
        heat = compute_gradcam(tiny_conv_model, batch)
        assert heat.shape == (3, 8, 6)  # resolução do mapa de conv_b (pós-pool)
        assert float(heat.min()) >= 0.0
        assert float(heat.max()) <= 1.0

        resized = resize_heatmap(heat, 16, 12)
        assert resized.shape == (3, 16, 12)
        assert float(resized.max()) <= 1.0

    def test_gradcam_without_conv_raises(self):
        tf = pytest.importorskip("tensorflow")
        from app.core.xai.gradcam import find_last_conv_layer

        inputs = tf.keras.Input(shape=(10,))
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
        dense_only = tf.keras.Model(inputs, outputs)
        with pytest.raises(ValueError, match="3D/4D"):
            find_last_conv_layer(dense_only)

    def test_gradcam_auto_on_conv1d_model(self):
        """Modelos Conv1D (Conformer-like) produzem CAM 1D valido."""
        tf = pytest.importorskip("tensorflow")
        from app.core.xai.gradcam import compute_gradcam_auto, heatmap_to_input_grid

        inputs = tf.keras.Input(shape=(20, 8))
        x = tf.keras.layers.Conv1D(4, 3, padding="same", name="c1d")(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)

        rng = np.random.default_rng(3)
        batch = rng.normal(size=(2, 20, 8)).astype("float32")
        cam, layer = compute_gradcam_auto(model, batch)
        assert layer == "c1d"
        assert cam.shape == (2, 20)
        grid = heatmap_to_input_grid(cam, 16, 12)
        assert grid.shape == (2, 16, 12)
        assert float(grid.max()) <= 1.0

    def test_heatmap_grid_reconstruction_from_tokens(self):
        """CAM de tokens (B, n) reconstroi grade quando n fatora bem."""
        pytest.importorskip("tensorflow")
        from app.core.xai.gradcam import _best_grid, heatmap_to_input_grid

        assert _best_grid(18, 16 / 12) == (6, 3)
        assert _best_grid(13, 1.0) is None  # primo -> faixa temporal

        cam = np.linspace(0, 1, 2 * 18, dtype="float32").reshape(2, 18)
        grid = heatmap_to_input_grid(cam, 16, 12)
        assert grid.shape == (2, 16, 12)


class TestShapWrapper:
    def test_shap_available_is_bool(self):
        from app.core.xai.shap_explainer import shap_available

        assert isinstance(shap_available(), bool)

    def test_tree_shap_matrix(self):
        pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestClassifier

        from app.core.xai.shap_explainer import explain_with_tree_shap

        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 6))
        y = (X[:, 2] > 0).astype(int)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

        matrix = explain_with_tree_shap(rf, X[:12])
        assert matrix.shape == (12, 6)
        assert np.isfinite(matrix).all()
        # feature 2 é a geradora do rótulo — deve dominar em média |SHAP|
        mean_abs = np.abs(matrix).mean(axis=0)
        assert int(np.argmax(mean_abs)) == 2

    def test_kernel_shap_matrix(self):
        pytest.importorskip("shap")
        from sklearn.linear_model import LogisticRegression

        from app.core.xai.shap_explainer import explain_with_kernel_shap

        rng = np.random.default_rng(11)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + 0.1 * X[:, 1] > 0).astype(int)
        clf = LogisticRegression().fit(X, y)

        matrix = explain_with_kernel_shap(
            lambda A: clf.predict_proba(A)[:, 1],
            X_background=X[:30],
            X_explain=X[:4],
            background_clusters=8,
            nsamples=64,
        )
        assert matrix.shape == (4, 4)
        assert np.isfinite(matrix).all()
