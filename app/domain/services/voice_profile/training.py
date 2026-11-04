"""Treino dos modelos de perfil de voz: dados, rede Keras ou clássico, EER.

``prepare_training_split`` monta o conjunto (split estratificado, pesos de
classe, scaler) a partir das features reais e fakes; as duas funções de treino
gravam o modelo e o ``training_config.json`` em ``model_dir`` e devolvem
``{"success", "metrics"}``.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from app.domain.services.voice_profile.models import build_profile_model

logger = logging.getLogger(__name__)


@dataclass
class ProfileTrainingSet:
    """Conjunto de treino do perfil, já normalizado pelo ``scaler``."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    class_weight: Dict[int, float]
    scaler: Any
    n_real: int
    n_fake: int


def prepare_training_split(
    real_features: List[np.ndarray], fake_features: List[np.ndarray]
) -> ProfileTrainingSet:
    """Split estratificado 80/20, pesos de classe balanceados e StandardScaler."""
    X_real = np.array(real_features)
    X_fake = np.array(fake_features)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate(
        [
            np.zeros(len(X_real)),
            np.ones(len(X_fake)),
        ]
    )

    # Stratified split (mantém proporção real:fake em train/val)
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Calcular class weights para balanceamento
    from sklearn.utils.class_weight import compute_class_weight

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_train
    )
    class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}

    # Normalização
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return ProfileTrainingSet(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        class_weight=class_weight_dict,
        scaler=scaler,
        n_real=len(X_real),
        n_fake=len(X_fake),
    )


def train_keras_profile_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    architecture: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    class_weight: dict,
    model_dir: Path,
    progress_callback=None,
) -> Dict[str, Any]:
    """Treina modelo DL com arquitetura do registry + class weight."""
    try:
        import tensorflow as tf

        input_dim = X_train.shape[1]

        # Construir modelo usando o registry
        model = build_profile_model(input_dim, architecture)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
            ),
        ]

        # Callback de progresso personalizado
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_callback and logs:
                    acc = logs.get("accuracy", 0)
                    val_acc = logs.get("val_accuracy", 0)
                    loss = logs.get("loss", 0)
                    precision = logs.get("val_precision", 0)
                    recall = logs.get("val_recall", 0)
                    progress_callback(
                        f"Época {epoch + 1}/{epochs} — "
                        f"loss: {loss:.4f}, acc: {acc:.4f}, "
                        f"val_acc: {val_acc:.4f}, "
                        f"P: {precision:.3f}, R: {recall:.3f}"
                    )

        callbacks.append(ProgressCallback())

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        # Salvar modelo
        model_path = model_dir / "profile_model.h5"
        model.save(str(model_path))

        # Calcular métricas avançadas
        y_pred_proba = model.predict(X_val, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        precision_val = precision_score(y_val, y_pred, zero_division=0)
        recall_val = recall_score(y_val, y_pred, zero_division=0)
        f1_val = f1_score(y_val, y_pred, zero_division=0)

        # EER (Equal Error Rate)
        eer = compute_eer(y_val, y_pred_proba)

        cm = confusion_matrix(y_val, y_pred)

        # Salvar config de treinamento
        config_path = model_dir / "training_config.json"
        training_info = {
            "architecture": architecture,
            "model_type": "dl",
            "input_dim": input_dim,
            "epochs_trained": len(history.history["loss"]),
            "model_params": model.count_params(),
            "created_at": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(training_info, f, indent=2)

        # Métricas finais
        final_metrics = {
            "accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4),
            "f1_score": round(f1_val, 4),
            "eer": round(eer, 4),
            "confusion_matrix": cm.tolist(),
            "epochs_trained": len(history.history["loss"]),
            "model_params": model.count_params(),
            "history": {
                k: [float(v) for v in vals] for k, vals in history.history.items()
            },
        }

        return {"success": True, "metrics": final_metrics}

    except Exception as e:
        logger.error(f"Erro no treinamento DL: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def train_classical_profile_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    architecture: str,
    class_weight: dict,
    model_dir: Path,
    progress_callback=None,
) -> Dict[str, Any]:
    """Treina modelo ML clássico (SVM ou Random Forest)."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        if progress_callback:
            progress_callback(f"Treinando {architecture.upper()}...")

        if architecture == "svm":
            from sklearn.svm import SVC

            model = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )
        elif architecture == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            return {
                "success": False,
                "error": f"Arquitetura ML '{architecture}' desconhecida",
            }

        model.fit(X_train, y_train)

        # Predições
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        precision_val = precision_score(y_val, y_pred, zero_division=0)
        recall_val = recall_score(y_val, y_pred, zero_division=0)
        f1_val = f1_score(y_val, y_pred, zero_division=0)
        eer = compute_eer(y_val, y_pred_proba)
        cm = confusion_matrix(y_val, y_pred)

        # Salvar modelo
        model_path = model_dir / "profile_model.pkl"
        joblib.dump(model, model_path)

        # Salvar config
        config_path = model_dir / "training_config.json"
        training_info = {
            "architecture": architecture,
            "model_type": "classical_ml",
            "input_dim": X_train.shape[1],
            "created_at": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(training_info, f, indent=2)

        if progress_callback:
            progress_callback(
                f"✅ {architecture.upper()} treinado — "
                f"Acc: {accuracy:.4f}, F1: {f1_val:.4f}"
            )

        final_metrics = {
            "accuracy": round(accuracy, 4),
            "val_accuracy": round(accuracy, 4),
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4),
            "f1_score": round(f1_val, 4),
            "eer": round(eer, 4),
            "confusion_matrix": cm.tolist(),
            "loss": 0.0,
            "val_loss": 0.0,
            "epochs_trained": 0,
        }

        return {"success": True, "metrics": final_metrics}

    except Exception as e:
        logger.error(f"Erro no treinamento ML clássico: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calcula Equal Error Rate (EER)."""
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        # Encontrar ponto onde FPR ≈ FNR
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        return float(eer)
    except Exception:
        return 0.0
