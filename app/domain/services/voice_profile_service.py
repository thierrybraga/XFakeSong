"""Serviço de gerenciamento de Perfis de Voz personalizados.

Permite criar perfis com dados pessoais, adicionar amostras de voz,
treinar modelos específicos por perfil e usar o modelo para verificação.

Melhorias v2:
- Suporte a TODAS as 12 arquiteturas via registry + SVM/RF
- Name mapping layer (snake_case ↔ display names)
- Class weight balancing para compensar imbalance real:fake
- Stratified train/val split
- Métricas avançadas (precision, recall, F1, EER)
- Validação mínima de dataset antes do treino
- Geração de fakes mais realista (vocoder artifacts, formant warp, etc.)
- feature_config populado no perfil
- Instância única gerida por app.dependencies.get_voice_profile_service
- Context manager para sessões DB (previne leaks)
- Cache de durações em metadata JSON
- Timeout guards no librosa.load
"""

import enum
import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np

from app.core.bootstrap import OperationalPaths
from app.core.db.models.voice_profile import VoiceProfile
from app.core.db.session import SessionLocal
from app.domain.services.voice_profile.features import (
    AUDIO_EXTENSIONS,
    DEFAULT_FEATURE_CONFIG,
    extract_profile_features,
    generate_synthetic_fakes,
)
from app.domain.services.voice_profile.training import (
    prepare_training_split,
    train_classical_profile_model,
    train_keras_profile_model,
)
from app.utils.file_utils import (
    resolve_within_directory,
    sanitize_filename,
    validate_path_segment,
)

logger = logging.getLogger(__name__)

def _default_voice_profiles_dir() -> Path:
    """Resolve storage after application settings/environment are initialized."""
    return OperationalPaths.resolve().data / "voice_profiles"

# Duração máxima de áudio aceita (guard contra arquivos muito longos)
MAX_AUDIO_DURATION = 300.0  # 5 minutos

# Requisitos mínimos para treinamento
MIN_SAMPLES = 3
MIN_DURATION_SECONDS = 5.0
RECOMMENDED_SAMPLES = 10
RECOMMENDED_DURATION = 60.0

# Arquiteturas ML clássicas (não usam registry de DL)
CLASSICAL_ML_ARCHITECTURES = {"svm", "random_forest"}

class ProfileStatus(str, enum.Enum):
    """Status possíveis de um perfil de voz."""
    CREATED = "created"
    COLLECTING = "collecting"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"


class VoiceProfileService:
    """Gerenciador de perfis de voz com dataset customizado e modelo treinado."""

    def __init__(self, profiles_dir: Optional[str] = None):
        # Sem singleton na classe: um `__new__` que devolvia sempre a
        # primeira instância ignorava em silêncio o `profiles_dir` das
        # seguintes. A instância compartilhada vem de
        # `app.dependencies.get_voice_profile_service`.
        self.profiles_dir = (
            Path(profiles_dir) if profiles_dir else _default_voice_profiles_dir()
        )
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_db(self):
        """Context manager para sessão DB — garante close em qualquer cenário."""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _profile_dir(self, profile_id: int) -> Path:
        return self.profiles_dir / f"profile_{profile_id}"

    def _samples_dir(self, profile_id: int) -> Path:
        return self._profile_dir(profile_id) / "samples"

    def _model_dir(self, profile_id: int) -> Path:
        return self._profile_dir(profile_id) / "model"

    # ------------------------------------------------------------------ #
    #  CRUD de Perfis                                                      #
    # ------------------------------------------------------------------ #

    def create_profile(
        self,
        name: str,
        telegram_id: str = None,
        phone: str = None,
        email: str = None,
        description: str = None,
        architecture: str = "sonic_sleuth",
    ) -> VoiceProfile:
        """Cria um novo perfil de voz e inicializa diretórios."""
        with self._get_db() as db:
            try:
                profile = VoiceProfile(
                    name=name,
                    telegram_id=telegram_id,
                    phone=phone,
                    email=email,
                    description=description,
                    architecture=architecture,
                    feature_config=DEFAULT_FEATURE_CONFIG,
                    dataset_dir="",  # placeholder
                    status=ProfileStatus.CREATED.value,
                )
                db.add(profile)
                db.flush()  # Gera o ID sem commit

                # Atualizar dataset_dir com o id real
                profile.dataset_dir = str(self._samples_dir(profile.id))

                # Commit único — transação atômica
                db.commit()
                db.refresh(profile)

                # Criar diretórios no filesystem
                self._samples_dir(profile.id).mkdir(parents=True, exist_ok=True)
                self._model_dir(profile.id).mkdir(parents=True, exist_ok=True)

                logger.info(f"Perfil de voz criado: {profile.id} - {name}")
                return profile
            except Exception as e:
                db.rollback()
                logger.error(f"Erro ao criar perfil: {e}")
                raise

    def get_profile(self, profile_id: int) -> Optional[VoiceProfile]:
        with self._get_db() as db:
            return db.query(VoiceProfile).filter_by(id=profile_id).first()

    def list_profiles(self) -> List[VoiceProfile]:
        with self._get_db() as db:
            return db.query(VoiceProfile).order_by(VoiceProfile.created_at.desc()).all()

    def update_profile(self, profile_id: int, **kwargs) -> Optional[VoiceProfile]:
        """Atualiza dados do perfil (nome, telegram_id, phone, email, description)."""
        allowed_fields = {"name", "telegram_id", "phone", "email", "description", "architecture"}
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return None
                for key, value in kwargs.items():
                    if key in allowed_fields:
                        setattr(profile, key, value)
                db.commit()
                db.refresh(profile)
                return profile
            except Exception as e:
                db.rollback()
                logger.error(f"Erro ao atualizar perfil {profile_id}: {e}")
                raise

    def delete_profile(self, profile_id: int) -> bool:
        """Remove perfil, dataset e modelo do filesystem e banco."""
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return False

                # Remover diretório no filesystem
                profile_dir = self._profile_dir(profile_id)
                if profile_dir.exists():
                    shutil.rmtree(profile_dir, ignore_errors=True)

                db.delete(profile)
                db.commit()
                logger.info(f"Perfil {profile_id} removido com sucesso.")
                return True
            except Exception as e:
                db.rollback()
                logger.error(f"Erro ao deletar perfil {profile_id}: {e}")
                return False

    # ------------------------------------------------------------------ #
    #  Gerenciamento de Amostras de Áudio                                  #
    # ------------------------------------------------------------------ #

    def add_audio_samples(
        self, profile_id: int, audio_files: List[Tuple[str, bytes]]
    ) -> Dict[str, Any]:
        """Adiciona amostras de áudio ao dataset do perfil."""
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return {"success": False, "error": "Perfil não encontrado"}

                samples_dir = self._samples_dir(profile_id)
                samples_dir.mkdir(parents=True, exist_ok=True)

                added = []
                errors = []
                total_new_duration = 0.0

                for filename, file_bytes in audio_files:
                    try:
                        safe_name = sanitize_filename(filename)
                        dest_path = samples_dir / safe_name

                        # Evitar sobrescrita
                        if dest_path.exists():
                            base, ext = os.path.splitext(safe_name)
                            counter = 1
                            while dest_path.exists():
                                safe_name = f"{base}_{counter}{ext}"
                                dest_path = samples_dir / safe_name
                                counter += 1

                        with open(dest_path, "wb") as f:
                            f.write(file_bytes)

                        # Calcular duração com timeout guard
                        try:
                            y, sr = librosa.load(
                                str(dest_path), sr=None,
                                duration=MAX_AUDIO_DURATION,
                            )
                            duration = len(y) / sr
                            total_new_duration += duration
                        except Exception:
                            duration = 0.0

                        added.append({"filename": safe_name, "duration": duration})

                    except Exception as e:
                        errors.append({"filename": filename, "error": str(e)})

                # Atualizar contagens no perfil
                profile.num_samples = sum(
                    1 for f in samples_dir.iterdir()
                    if f.suffix.lower() in AUDIO_EXTENSIONS
                )
                profile.total_duration_seconds += total_new_duration
                if profile.status == ProfileStatus.CREATED.value:
                    profile.status = ProfileStatus.COLLECTING.value
                db.commit()

                # Salvar cache de durações
                self._save_duration_cache(profile_id, added)

                return {
                    "success": True,
                    "added": added,
                    "errors": errors,
                    "total_samples": profile.num_samples,
                    "total_duration": profile.total_duration_seconds,
                }
            except Exception as e:
                db.rollback()
                logger.error(f"Erro ao adicionar amostras ao perfil {profile_id}: {e}")
                return {"success": False, "error": str(e)}


    def remove_audio_sample(self, profile_id: int, filename: str) -> bool:
        """Remove uma amostra do dataset do perfil."""
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return False

                try:
                    clean_name = validate_path_segment(
                        filename, label="nome da amostra"
                    )
                    samples_dir = self._samples_dir(profile_id)
                    raw_path = samples_dir / clean_name
                    if raw_path.is_symlink():
                        return False
                    file_path = resolve_within_directory(
                        samples_dir, clean_name, must_exist=True
                    )
                except (ValueError, FileNotFoundError):
                    return False

                if file_path.is_file():
                    # Usar cache de duração se disponível
                    duration = self._get_cached_duration(profile_id, clean_name)
                    if duration is None:
                        try:
                            y, sr = librosa.load(
                                str(file_path), sr=None,
                                duration=MAX_AUDIO_DURATION,
                            )
                            duration = len(y) / sr
                        except Exception:
                            duration = 0.0

                    profile.total_duration_seconds = max(
                        0, profile.total_duration_seconds - duration
                    )

                    file_path.unlink()
                    profile.num_samples = max(0, profile.num_samples - 1)
                    db.commit()

                    # Atualizar cache
                    self._remove_from_duration_cache(profile_id, clean_name)
                    return True
                return False
            except Exception as e:
                db.rollback()
                logger.error(f"Erro ao remover amostra {filename}: {e}")
                return False

    def get_dataset_info(self, profile_id: int) -> Dict[str, Any]:
        """Retorna informações sobre o dataset de um perfil."""
        samples_dir = self._samples_dir(profile_id)
        if not samples_dir.exists():
            return {"files": [], "total_samples": 0, "total_duration": 0.0}

        files = []
        total_duration = 0.0
        duration_cache = self._load_duration_cache(profile_id)
        cache_updated = False

        for fpath in sorted(samples_dir.iterdir()):
            if fpath.suffix.lower() in AUDIO_EXTENSIONS:
                cached = duration_cache.get(fpath.name)
                if cached:
                    duration = cached.get("duration", 0.0)
                    sr = cached.get("sample_rate", 0)
                else:
                    try:
                        y, sr = librosa.load(
                            str(fpath), sr=None,
                            duration=MAX_AUDIO_DURATION,
                        )
                        duration = len(y) / sr
                        duration_cache[fpath.name] = {
                            "duration": round(duration, 2),
                            "sample_rate": sr,
                        }
                        cache_updated = True
                    except Exception:
                        duration = 0.0
                        sr = 0

                files.append({
                    "filename": fpath.name,
                    "size_bytes": fpath.stat().st_size,
                    "duration": round(duration, 2),
                    "sample_rate": sr,
                })
                total_duration += duration

        if cache_updated:
            self._write_duration_cache(profile_id, duration_cache)

        return {
            "files": files,
            "total_samples": len(files),
            "total_duration": round(total_duration, 2),
        }

    # ------------------------------------------------------------------ #
    #  Validação de Dataset                                                #
    # ------------------------------------------------------------------ #

    def validate_dataset_for_training(self, profile_id: int) -> Dict[str, Any]:
        """Valida se o dataset atende requisitos mínimos para treinamento.

        Returns:
            Dict com 'valid' (bool), 'errors' (list), 'warnings' (list)
        """
        info = self.get_dataset_info(profile_id)
        errors = []
        warnings = []

        num_samples = info["total_samples"]
        total_dur = info["total_duration"]

        if num_samples < MIN_SAMPLES:
            errors.append(
                f"Mínimo de {MIN_SAMPLES} amostras necessárias "
                f"(atual: {num_samples})."
            )

        if total_dur < MIN_DURATION_SECONDS:
            errors.append(
                f"Duração mínima de {MIN_DURATION_SECONDS}s necessária "
                f"(atual: {total_dur:.1f}s)."
            )

        if num_samples < RECOMMENDED_SAMPLES:
            warnings.append(
                f"Recomendado ao menos {RECOMMENDED_SAMPLES} amostras "
                f"para bom desempenho (atual: {num_samples})."
            )

        if total_dur < RECOMMENDED_DURATION:
            warnings.append(
                f"Recomendado ao menos {RECOMMENDED_DURATION}s de áudio "
                f"(atual: {total_dur:.1f}s)."
            )

        # Verificar variedade de durações
        durations = [f["duration"] for f in info["files"] if f["duration"] > 0]
        if durations and max(durations) - min(durations) < 1.0:
            warnings.append(
                "Amostras com durações muito similares. "
                "Varie entre frases curtas e longas para melhor generalização."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "num_samples": num_samples,
            "total_duration": total_dur,
        }

    # ------------------------------------------------------------------ #
    #  Treinamento do Modelo do Perfil                                     #
    # ------------------------------------------------------------------ #

    def train_profile_model(
        self,
        profile_id: int,
        architecture: str = None,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Treina um modelo de classificação binária para o perfil.

        Suporta:
        - Todas as 12 arquiteturas DL do registry (via feature-based adapter)
        - SVM e Random Forest (sklearn)
        - Class weight balancing automático
        - Stratified train/val split
        - Métricas avançadas (precision, recall, F1, EER)
        """
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return {"success": False, "error": "Perfil não encontrado"}

                # Validar dataset
                validation = self.validate_dataset_for_training(profile_id)
                if not validation["valid"]:
                    return {
                        "success": False,
                        "error": " | ".join(validation["errors"]),
                        "warnings": validation.get("warnings", []),
                    }

                arch = architecture or profile.architecture or "sonic_sleuth"
                profile.status = ProfileStatus.TRAINING.value
                profile.architecture = arch
                profile.feature_config = DEFAULT_FEATURE_CONFIG
                db.commit()

                if progress_callback:
                    # Reportar warnings
                    for w in validation.get("warnings", []):
                        progress_callback(f"⚠️ {w}")
                    progress_callback("Extraindo features das amostras...")

                samples_dir = self._samples_dir(profile_id)
                sr_target = DEFAULT_FEATURE_CONFIG["sr_target"]
                real_features = self._load_real_features(samples_dir, sr_target)

                if len(real_features) < MIN_SAMPLES:
                    profile.status = ProfileStatus.ERROR.value
                    db.commit()
                    return {
                        "success": False,
                        "error": (
                            f"Apenas {len(real_features)} amostras válidas. "
                            f"Mínimo: {MIN_SAMPLES}."
                        ),
                    }

                if progress_callback:
                    progress_callback("Gerando amostras sintéticas (fakes)...")

                fake_features = generate_synthetic_fakes(
                    real_features, samples_dir, sr_target
                )

                if progress_callback:
                    progress_callback("Preparando dataset com stratified split...")

                conjunto = prepare_training_split(real_features, fake_features)

                if progress_callback:
                    progress_callback(
                        f"Treinando modelo ({arch}, {epochs} épocas, "
                        f"{len(conjunto.X_train)} train / "
                        f"{len(conjunto.X_val)} val)..."
                    )

                # Dispatch: ML clássico vs DL
                model_dir = self._model_dir(profile_id)
                if arch in CLASSICAL_ML_ARCHITECTURES:
                    metrics = train_classical_profile_model(
                        conjunto.X_train, conjunto.y_train,
                        conjunto.X_val, conjunto.y_val,
                        arch, conjunto.class_weight,
                        model_dir, progress_callback,
                    )
                else:
                    metrics = train_keras_profile_model(
                        conjunto.X_train, conjunto.y_train,
                        conjunto.X_val, conjunto.y_val,
                        arch, epochs, batch_size, learning_rate,
                        conjunto.class_weight,
                        model_dir, progress_callback,
                    )

                if not metrics.get("success", False):
                    profile.status = ProfileStatus.ERROR.value
                    db.commit()
                    return metrics

                scaler_path = model_dir / "scaler.pkl"
                joblib.dump(conjunto.scaler, scaler_path)

                model_ext = ".pkl" if arch in CLASSICAL_ML_ARCHITECTURES else ".h5"
                model_path = model_dir / f"profile_model{model_ext}"
                profile.model_path = str(model_path)
                profile.scaler_path = str(scaler_path)
                profile.status = ProfileStatus.READY.value
                profile.feature_config = DEFAULT_FEATURE_CONFIG
                profile.training_config = {
                    "architecture": arch,
                    "epochs": epochs if arch not in CLASSICAL_ML_ARCHITECTURES else 0,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_real_samples": conjunto.n_real,
                    "num_fake_samples": conjunto.n_fake,
                    "feature_dim": conjunto.X_train.shape[1],
                    "class_weights": {
                        str(k): round(v, 4)
                        for k, v in conjunto.class_weight.items()
                    },
                    "stratified_split": True,
                    "train_size": len(conjunto.X_train),
                    "val_size": len(conjunto.X_val),
                }
                profile.training_metrics = metrics.get("metrics", {})
                db.commit()

                logger.info(
                    f"Modelo treinado para perfil {profile_id}: "
                    f"acc={metrics.get('metrics', {}).get('val_accuracy', 'N/A')}"
                )
                return metrics

            except Exception as e:
                logger.error(
                    f"Erro no treinamento do perfil {profile_id}: {e}",
                    exc_info=True,
                )
                try:
                    profile = (
                        db.query(VoiceProfile).filter_by(id=profile_id).first()
                    )
                    if profile:
                        profile.status = ProfileStatus.ERROR.value
                        db.commit()
                except Exception:
                    db.rollback()
                return {"success": False, "error": str(e)}

    def _load_real_features(
        self, samples_dir: Path, sr_target: int
    ) -> List[np.ndarray]:
        """Features das amostras do perfil (as REAIS do treino)."""
        real_features = []
        for audio_file in sorted(samples_dir.iterdir()):
            if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
                try:
                    y, sr = librosa.load(str(audio_file), sr=sr_target, mono=True)
                    feats = extract_profile_features(y, sr)
                    if feats is not None:
                        real_features.append(feats)
                except Exception as e:
                    logger.warning(f"Erro ao processar {audio_file.name}: {e}")
        return real_features

    # ------------------------------------------------------------------ #
    #  Inferência com Modelo do Perfil                                     #
    # ------------------------------------------------------------------ #

    def detect_with_profile(
        self, profile_id: int, audio_path: str
    ) -> Dict[str, Any]:
        """Verifica se um áudio é da pessoa do perfil ou não."""
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return {"success": False, "error": "Perfil não encontrado"}

                if profile.status != ProfileStatus.READY.value or not profile.model_path:
                    return {
                        "success": False,
                        "error": "Modelo do perfil ainda não foi treinado ou não está pronto.",
                    }

                model_path = Path(profile.model_path)
                scaler_path = Path(profile.scaler_path) if profile.scaler_path else None

                if not model_path.exists():
                    return {"success": False, "error": "Arquivo do modelo não encontrado."}

                # Detectar tipo de modelo
                arch = profile.architecture or "sonic_sleuth"
                is_classical = arch in CLASSICAL_ML_ARCHITECTURES

                if is_classical:
                    model = joblib.load(str(model_path))
                else:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(str(model_path))

                scaler = (
                    joblib.load(str(scaler_path))
                    if scaler_path and scaler_path.exists() else None
                )

                # Extrair features do áudio
                y, sr = librosa.load(
                    audio_path, sr=16000, mono=True,
                    duration=MAX_AUDIO_DURATION,
                )
                features = extract_profile_features(y, sr)

                if features is None:
                    return {"success": False, "error": "Falha na extração de features."}

                # Normalizar
                features_2d = features.reshape(1, -1)
                if scaler is not None:
                    features_2d = scaler.transform(features_2d)

                # Predição
                if is_classical:
                    prediction = model.predict_proba(features_2d)[0][1]
                else:
                    prediction = model.predict(features_2d, verbose=0)[0][0]

                is_fake = float(prediction) > 0.5
                confidence = float(prediction) if is_fake else 1.0 - float(prediction)

                return {
                    "success": True,
                    "is_authentic": not is_fake,
                    "is_fake": is_fake,
                    "confidence": round(confidence * 100, 2),
                    "raw_score": float(prediction),
                    "profile_name": profile.name,
                    "profile_id": profile_id,
                    "details": {
                        "model_architecture": profile.architecture,
                        "model_type": "classical_ml" if is_classical else "deep_learning",
                        "num_training_samples": profile.training_config.get(
                            "num_real_samples", 0
                        ) if profile.training_config else 0,
                        "feature_dim": len(features),
                        "training_metrics": {
                            k: v for k, v in (profile.training_metrics or {}).items()
                            if k in ("f1_score", "eer", "precision", "recall")
                        },
                    },
                }

            except Exception as e:
                logger.error(f"Erro na detecção com perfil {profile_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    #  Utilitários                                                         #
    # ------------------------------------------------------------------ #

    # ── Duration Cache ─────────────────────────────────────────────────

    def _duration_cache_path(self, profile_id: int) -> Path:
        return self._profile_dir(profile_id) / ".duration_cache.json"

    def _load_duration_cache(self, profile_id: int) -> Dict[str, Any]:
        cache_path = self._duration_cache_path(profile_id)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_duration_cache(self, profile_id: int, cache: Dict[str, Any]):
        cache_path = self._duration_cache_path(profile_id)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache de durações: {e}")

    def _save_duration_cache(self, profile_id: int, added_files: List[Dict]):
        cache = self._load_duration_cache(profile_id)
        for item in added_files:
            fname = item.get("filename", "")
            dur = item.get("duration", 0.0)
            if fname:
                cache[fname] = {"duration": dur, "sample_rate": 0}
        self._write_duration_cache(profile_id, cache)

    def _get_cached_duration(self, profile_id: int, filename: str) -> Optional[float]:
        cache = self._load_duration_cache(profile_id)
        entry = cache.get(filename)
        return entry.get("duration") if entry else None

    def _remove_from_duration_cache(self, profile_id: int, filename: str):
        cache = self._load_duration_cache(profile_id)
        if filename in cache:
            del cache[filename]
            self._write_duration_cache(profile_id, cache)
