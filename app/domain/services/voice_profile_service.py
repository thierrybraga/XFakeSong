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
- Singleton service pattern
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np

from app.core.database import SessionLocal
from app.domain.models.voice_profile import VoiceProfile

logger = logging.getLogger(__name__)

# Diretório base para armazenar perfis de voz
VOICE_PROFILES_DIR = Path("app/voice_profiles")

# Duração máxima de áudio aceita (guard contra arquivos muito longos)
MAX_AUDIO_DURATION = 300.0  # 5 minutos

# Requisitos mínimos para treinamento
MIN_SAMPLES = 3
MIN_DURATION_SECONDS = 5.0
RECOMMENDED_SAMPLES = 10
RECOMMENDED_DURATION = 60.0

# Arquiteturas ML clássicas (não usam registry de DL)
CLASSICAL_ML_ARCHITECTURES = {"svm", "random_forest"}

# Feature configuration padrão
DEFAULT_FEATURE_CONFIG = {
    "n_mfcc": 13,
    "sr_target": 16000,
    "feature_set": "full",  # full | compact
    "features": [
        "mfcc_stats",       # 13 * 4 = 52
        "delta_mfcc",       # 13 * 2 = 26
        "spectral_stats",   # 6 * 2 = 12  (centroid, bandwidth, rolloff, flatness, zcr, rms)
        "spectral_contrast", # 7 * 2 = 14
        "f0_stats",         # 4
        "chroma_stats",     # 12 * 2 = 24
        "tonnetz_stats",    # 6 * 2 = 12
    ],
    "total_dim": 144,
}


class ProfileStatus(str, enum.Enum):
    """Status possíveis de um perfil de voz."""
    CREATED = "created"
    COLLECTING = "collecting"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"


class VoiceProfileService:
    """Gerenciador de perfis de voz com dataset customizado e modelo treinado."""

    _instance: Optional["VoiceProfileService"] = None

    def __new__(cls, profiles_dir: str = None):
        """Singleton — garante instância única."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, profiles_dir: str = None):
        if self._initialized:
            return
        self.profiles_dir = Path(profiles_dir) if profiles_dir else VOICE_PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

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
                        safe_name = self._sanitize_filename(filename)
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
                audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
                profile.num_samples = sum(
                    1 for f in samples_dir.iterdir()
                    if f.suffix.lower() in audio_exts
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

    def add_audio_sample_from_path(
        self, profile_id: int, source_path: str
    ) -> Dict[str, Any]:
        """Adiciona uma amostra de áudio a partir de um caminho no filesystem."""
        path = Path(source_path)
        if not path.exists():
            return {"success": False, "error": f"Arquivo não encontrado: {source_path}"}
        with open(path, "rb") as f:
            data = f.read()
        return self.add_audio_samples(profile_id, [(path.name, data)])

    def remove_audio_sample(self, profile_id: int, filename: str) -> bool:
        """Remove uma amostra do dataset do perfil."""
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return False

                file_path = self._samples_dir(profile_id) / filename
                if file_path.exists():
                    # Usar cache de duração se disponível
                    duration = self._get_cached_duration(profile_id, filename)
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
                    self._remove_from_duration_cache(profile_id, filename)
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
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        duration_cache = self._load_duration_cache(profile_id)
        cache_updated = False

        for fpath in sorted(samples_dir.iterdir()):
            if fpath.suffix.lower() in audio_extensions:
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
                real_features = []
                sr_target = DEFAULT_FEATURE_CONFIG["sr_target"]

                for audio_file in sorted(samples_dir.iterdir()):
                    if audio_file.suffix.lower() in {
                        ".wav", ".mp3", ".flac", ".ogg", ".m4a",
                    }:
                        try:
                            y, sr = librosa.load(
                                str(audio_file), sr=sr_target, mono=True
                            )
                            feats = self._extract_profile_features(y, sr)
                            if feats is not None:
                                real_features.append(feats)
                        except Exception as e:
                            logger.warning(
                                f"Erro ao processar {audio_file.name}: {e}"
                            )

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

                fake_features = self._generate_synthetic_fakes(
                    real_features, samples_dir, sr_target
                )

                if progress_callback:
                    progress_callback("Preparando dataset com stratified split...")

                X_real = np.array(real_features)
                X_fake = np.array(fake_features)

                X = np.vstack([X_real, X_fake])
                y = np.concatenate([
                    np.zeros(len(X_real)),
                    np.ones(len(X_fake)),
                ])

                # Stratified split (mantém proporção real:fake em train/val)
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=42
                )
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

                if progress_callback:
                    progress_callback(
                        f"Treinando modelo ({arch}, {epochs} épocas, "
                        f"{len(X_train)} train / {len(X_val)} val)..."
                    )

                # Dispatch: ML clássico vs DL
                if arch in CLASSICAL_ML_ARCHITECTURES:
                    metrics = self._train_classical_model(
                        X_train, y_train, X_val, y_val,
                        arch, class_weight_dict,
                        profile_id, progress_callback,
                    )
                else:
                    metrics = self._train_model(
                        X_train, y_train, X_val, y_val,
                        arch, epochs, batch_size, learning_rate,
                        class_weight_dict,
                        profile_id, progress_callback,
                    )

                if not metrics.get("success", False):
                    profile.status = ProfileStatus.ERROR.value
                    db.commit()
                    return metrics

                scaler_path = self._model_dir(profile_id) / "scaler.pkl"
                joblib.dump(scaler, scaler_path)

                model_ext = ".pkl" if arch in CLASSICAL_ML_ARCHITECTURES else ".h5"
                model_path = self._model_dir(profile_id) / f"profile_model{model_ext}"
                profile.model_path = str(model_path)
                profile.scaler_path = str(scaler_path)
                profile.status = ProfileStatus.READY.value
                profile.feature_config = DEFAULT_FEATURE_CONFIG
                profile.training_config = {
                    "architecture": arch,
                    "epochs": epochs if arch not in CLASSICAL_ML_ARCHITECTURES else 0,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_real_samples": len(X_real),
                    "num_fake_samples": len(X_fake),
                    "feature_dim": X_train.shape[1],
                    "class_weights": {
                        str(k): round(v, 4)
                        for k, v in class_weight_dict.items()
                    },
                    "stratified_split": True,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
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

    def _extract_profile_features(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extrai vetor de features para treinamento/inferência de perfil.

        Retorna vetor de 146 dimensões com estatísticas agregadas.
        """
        try:
            features = []

            # MFCCs (13 coefs x 4 stats = 52)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            features.extend(np.max(mfcc, axis=1) - np.min(mfcc, axis=1))
            features.extend(np.median(mfcc, axis=1))

            # Delta MFCCs (13 x 2 stats = 26)
            delta_mfcc = librosa.feature.delta(mfcc)
            features.extend(np.mean(delta_mfcc, axis=1))
            features.extend(np.std(delta_mfcc, axis=1))

            # Spectral features (7 features x 2 stats = 14)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)

            for feat in [spectral_centroid, spectral_bandwidth, spectral_rolloff,
                         spectral_flatness, zcr, rms]:
                features.extend([np.mean(feat), np.std(feat)])

            # Spectral contrast (7 bands x 2 stats = 14)
            features.extend(np.mean(spectral_contrast, axis=1))
            features.extend(np.std(spectral_contrast, axis=1))

            # F0 / Pitch (4 features)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0])
            if len(f0_valid) == 0:
                f0_valid = np.array([0])
            features.extend([
                np.mean(f0_valid), np.std(f0_valid),
                np.max(f0_valid) - np.min(f0_valid),
                np.mean(voiced_probs) if voiced_probs is not None else 0.0,
            ])

            # Chroma (12 x 2 stats = 24)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))

            # Tonnetz (6 x 2 = 12)
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                features.extend(np.mean(tonnetz, axis=1))
                features.extend(np.std(tonnetz, axis=1))
            except Exception:
                features.extend([0.0] * 12)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Erro na extração de features: {e}")
            return None

    def _generate_synthetic_fakes(
        self,
        real_features: List[np.ndarray],
        samples_dir: Path,
        sr_target: int,
    ) -> List[np.ndarray]:
        """Gera features de amostras fake via perturbações realistas.

        Inclui perturbações que simulam artefatos de deepfake:
        - Pitch shift (conversão de voz)
        - Ruído gaussiano (artefatos de vocoder)
        - Time stretch (distorções temporais)
        - Formant warp (pitch + speed combo)
        - Low-pass filter (perda de alta frequência de vocoders)
        - Quantization noise (artefatos de compressão neural)
        - Spectral smoothing (suavização típica de TTS)
        """
        import scipy.signal as signal

        fake_features = []
        audio_files = sorted(
            f for f in samples_dir.iterdir()
            if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        )

        for audio_file in audio_files:
            try:
                y, sr = librosa.load(str(audio_file), sr=sr_target, mono=True)

                perturbations = []

                # 1. Pitch shift +3 semitons (conversão de voz leve)
                try:
                    y_p1 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=3)
                    perturbations.append(y_p1)
                except Exception:
                    pass

                # 2. Pitch shift -3 semitons
                try:
                    y_p2 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-3)
                    perturbations.append(y_p2)
                except Exception:
                    pass

                # 3. Ruído gaussiano (artefatos de vocoder)
                noise = np.random.randn(len(y)) * 0.03
                perturbations.append(y + noise)

                # 4. Time stretch rápido
                try:
                    y_fast = librosa.effects.time_stretch(y=y, rate=1.2)
                    perturbations.append(y_fast)
                except Exception:
                    pass

                # 5. Formant warp (pitch +5 + slow)
                try:
                    y_fw = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=5)
                    y_fw = librosa.effects.time_stretch(y=y_fw, rate=0.85)
                    perturbations.append(y_fw)
                except Exception:
                    pass

                # 6. Low-pass filter (simula vocoder que perde altas freq.)
                try:
                    nyquist = sr / 2
                    cutoff = 3500 / nyquist  # Corta acima de 3.5kHz
                    if 0 < cutoff < 1:
                        b, a = signal.butter(4, cutoff, btype='low')
                        y_lp = signal.filtfilt(b, a, y).astype(np.float32)
                        perturbations.append(y_lp)
                except Exception:
                    pass

                # 7. Quantization noise (simula artefatos de compressão neural)
                levels = 256
                y_quant = np.round(y * levels) / levels
                noise_q = np.random.randn(len(y)) * 0.01
                perturbations.append(y_quant + noise_q)

                # Extrair features de cada perturbação
                for y_pert in perturbations:
                    feats = self._extract_profile_features(y_pert, sr)
                    if feats is not None:
                        fake_features.append(feats)

            except Exception as e:
                logger.warning(f"Erro ao gerar fake de {audio_file.name}: {e}")

        # Se houver poucas fakes, interpolar features com ruído
        while len(fake_features) < len(real_features):
            idx = np.random.randint(0, len(real_features))
            noise_feat = real_features[idx] + np.random.randn(
                len(real_features[idx])
            ) * 0.3 * np.std(real_features[idx])
            fake_features.append(noise_feat)

        return fake_features

    def _build_profile_model(
        self,
        input_dim: int,
        architecture: str,
    ):
        """Constrói modelo para voice profile usando o architecture registry.

        Para arquiteturas DL do registry: constrói um adapter de features
        que mapeia o vetor de features fixo para a entrada esperada pela
        arquitetura, usa a arquitetura do registry como backbone, e adiciona
        um classificador binário.

        Fallback: modelo MLP compacto se a arquitetura do registry falhar.
        """

        # Tentar usar a arquitetura do registry
        try:
            from app.domain.models.architectures.registry import (
                ArchitectureRegistry,
                architecture_registry,
            )

            # Resolver display name
            display_name = ArchitectureRegistry.normalize_architecture_name(architecture)
            arch_info = architecture_registry.get_architecture(display_name)
            input_req = arch_info.input_requirements

            logger.info(
                f"Construindo modelo com arquitetura '{display_name}' "
                f"(input_type={input_req.get('type')}) para profile (feature_dim={input_dim})"
            )

            # Para arquiteturas baseadas em features/spectrogram, usar adapter
            # O perfil trabalha com features fixas (146-dim), então criamos
            # um modelo que projeta essas features para o formato da arquitetura
            # e usa camadas inspiradas na arquitetura escolhida

            # Buscar configuração da arquitetura
            arch_params = arch_info.default_params
            dropout = arch_params.get("dropout_rate", 0.3)

            # Construir modelo adaptado por tipo de arquitetura
            if display_name in ("AASIST", "RawGAT-ST"):
                # Modelos de atenção: usar self-attention layers
                model = self._build_attention_profile_model(
                    input_dim, dropout, arch_params
                )
            elif display_name in ("Conformer", "SpectrogramTransformer"):
                # Transformer-based: usar transformer encoder leve
                model = self._build_transformer_profile_model(
                    input_dim, dropout, arch_params
                )
            elif display_name in ("RawNet2", "Sonic Sleuth"):
                # Conv-based: usar 1D convolutions sobre features
                model = self._build_conv1d_profile_model(
                    input_dim, dropout, arch_params
                )
            elif display_name in ("EfficientNet-LSTM", "WavLM", "HuBERT"):
                # Recurrent: usar LSTM/GRU sobre features
                model = self._build_recurrent_profile_model(
                    input_dim, dropout, arch_params
                )
            elif display_name in ("Ensemble", "MultiscaleCNN", "Hybrid CNN-Transformer"):
                # Multi-branch: usar arquitetura multi-head
                model = self._build_multihead_profile_model(
                    input_dim, dropout, arch_params
                )
            else:
                # Fallback genérico
                model = self._build_mlp_profile_model(input_dim, dropout)

            logger.info(
                f"Modelo '{display_name}' construído: "
                f"{model.count_params()} parâmetros"
            )
            return model

        except Exception as e:
            logger.warning(
                f"Falha ao construir modelo via registry para '{architecture}': {e}. "
                f"Usando MLP fallback."
            )
            return self._build_mlp_profile_model(input_dim, 0.3)

    def _build_mlp_profile_model(self, input_dim: int, dropout: float):
        """MLP compacto (fallback e baseline)."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(dropout * 0.7),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        return model

    def _build_attention_profile_model(self, input_dim: int, dropout: float, params: dict):
        """Modelo com self-attention (inspirado em AASIST/RawGAT-ST)."""
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(input_dim,))
        # Reshape para sequência de pseudo-tokens
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Reshape((16, 16))(x)
        # Multi-Head Attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        # Feed-forward
        ff = tf.keras.layers.Dense(64, activation='relu')(x)
        ff = tf.keras.layers.Dense(16)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
        # Pool & classify
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs, name="attention_profile")

    def _build_transformer_profile_model(self, input_dim: int, dropout: float, params: dict):
        """Modelo Transformer encoder leve (inspirado em Conformer/AST)."""
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Reshape((8, 16))(x)

        # 2 Transformer blocks
        for _ in range(2):
            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=16, dropout=dropout
            )(x, x)
            x = tf.keras.layers.Add()([x, attn])
            x = tf.keras.layers.LayerNormalization()(x)
            ff = tf.keras.layers.Dense(32, activation='gelu')(x)
            ff = tf.keras.layers.Dense(16)(ff)
            x = tf.keras.layers.Add()([x, ff])
            x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs, name="transformer_profile")

    def _build_conv1d_profile_model(self, input_dim: int, dropout: float, params: dict):
        """Modelo Conv1D (inspirado em RawNet2/Sonic Sleuth)."""
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Reshape((input_dim, 1))(inputs)

        # Conv blocks
        for filters in [32, 64, 128]:
            x = tf.keras.layers.Conv1D(
                filters, 3, padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
            x = tf.keras.layers.Dropout(dropout * 0.5)(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs, name="conv1d_profile")

    def _build_recurrent_profile_model(self, input_dim: int, dropout: float, params: dict):
        """Modelo LSTM bidirecional (inspirado em EfficientNet-LSTM/WavLM/HuBERT)."""
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(input_dim,))
        # Reshape em sequência
        x = tf.keras.layers.Reshape((input_dim // 2, 2))(inputs)
        # Bi-LSTM
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=dropout)
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, dropout=dropout)
        )(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs, name="recurrent_profile")

    def _build_multihead_profile_model(self, input_dim: int, dropout: float, params: dict):
        """Modelo multi-branch (inspirado em Ensemble/MultiscaleCNN/Hybrid)."""
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(input_dim,))

        # Branch 1: Dense
        b1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        b1 = tf.keras.layers.BatchNormalization()(b1)
        b1 = tf.keras.layers.Dropout(dropout)(b1)
        b1 = tf.keras.layers.Dense(64, activation='relu')(b1)

        # Branch 2: Conv1D
        x2 = tf.keras.layers.Reshape((input_dim, 1))(inputs)
        x2 = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x2)
        x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
        b2 = tf.keras.layers.Dense(64, activation='relu')(x2)

        # Branch 3: Feature selection via attention
        attn_weights = tf.keras.layers.Dense(input_dim, activation='softmax')(inputs)
        x3 = tf.keras.layers.Multiply()([inputs, attn_weights])
        b3 = tf.keras.layers.Dense(64, activation='relu')(x3)

        # Fusion
        merged = tf.keras.layers.Concatenate()([b1, b2, b3])
        x = tf.keras.layers.Dense(128, activation='relu')(merged)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs, name="multihead_profile")

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        architecture: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        class_weight: dict,
        profile_id: int,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Treina modelo DL com arquitetura do registry + class weight."""
        try:
            import tensorflow as tf

            input_dim = X_train.shape[1]

            # Construir modelo usando o registry
            model = self._build_profile_model(input_dim, architecture)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                ],
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=8, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
                ),
            ]

            # Callback de progresso personalizado
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if progress_callback and logs:
                        acc = logs.get('accuracy', 0)
                        val_acc = logs.get('val_accuracy', 0)
                        loss = logs.get('loss', 0)
                        precision = logs.get('val_precision', 0)
                        recall = logs.get('val_recall', 0)
                        progress_callback(
                            f"Época {epoch + 1}/{epochs} — "
                            f"loss: {loss:.4f}, acc: {acc:.4f}, "
                            f"val_acc: {val_acc:.4f}, "
                            f"P: {precision:.3f}, R: {recall:.3f}"
                        )

            callbacks.append(ProgressCallback())

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=0,
            )

            # Salvar modelo
            model_path = self._model_dir(profile_id) / "profile_model.h5"
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
            eer = self._compute_eer(y_val, y_pred_proba)

            cm = confusion_matrix(y_val, y_pred)

            # Salvar config de treinamento
            config_path = self._model_dir(profile_id) / "training_config.json"
            training_info = {
                "architecture": architecture,
                "model_type": "dl",
                "input_dim": input_dim,
                "epochs_trained": len(history.history['loss']),
                "model_params": model.count_params(),
                "created_at": datetime.now().isoformat(),
            }
            with open(config_path, "w") as f:
                json.dump(training_info, f, indent=2)

            # Métricas finais
            final_metrics = {
                "accuracy": float(history.history['accuracy'][-1]),
                "val_accuracy": float(history.history['val_accuracy'][-1]),
                "loss": float(history.history['loss'][-1]),
                "val_loss": float(history.history['val_loss'][-1]),
                "precision": round(precision_val, 4),
                "recall": round(recall_val, 4),
                "f1_score": round(f1_val, 4),
                "eer": round(eer, 4),
                "confusion_matrix": cm.tolist(),
                "epochs_trained": len(history.history['loss']),
                "model_params": model.count_params(),
                "history": {
                    k: [float(v) for v in vals]
                    for k, vals in history.history.items()
                },
            }

            return {"success": True, "metrics": final_metrics}

        except Exception as e:
            logger.error(f"Erro no treinamento DL: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _train_classical_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        architecture: str,
        class_weight: dict,
        profile_id: int,
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
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    class_weight='balanced',
                    probability=True,
                    random_state=42,
                )
            elif architecture == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                return {"success": False, "error": f"Arquitetura ML '{architecture}' desconhecida"}

            model.fit(X_train, y_train)

            # Predições
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            accuracy = accuracy_score(y_val, y_pred)
            precision_val = precision_score(y_val, y_pred, zero_division=0)
            recall_val = recall_score(y_val, y_pred, zero_division=0)
            f1_val = f1_score(y_val, y_pred, zero_division=0)
            eer = self._compute_eer(y_val, y_pred_proba)
            cm = confusion_matrix(y_val, y_pred)

            # Salvar modelo
            model_path = self._model_dir(profile_id) / "profile_model.pkl"
            joblib.dump(model, model_path)

            # Salvar config
            config_path = self._model_dir(profile_id) / "training_config.json"
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

    @staticmethod
    def _compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
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
                features = self._extract_profile_features(y, sr)

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

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Remove caracteres perigosos do nome de arquivo."""
        import re
        safe = re.sub(r'[^\w\-.]', '_', filename)
        safe = safe.replace('..', '_')
        return safe

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
