"""Serviço de gerenciamento de Perfis de Voz personalizados.

Permite criar perfis com dados pessoais, adicionar amostras de voz,
treinar modelos específicos por perfil e usar o modelo para verificação.

Melhorias:
- Context manager para sessões DB (previne leaks)
- Cache de durações em metadata JSON
- Timeout guards no librosa.load
- Transação única no create_profile
"""

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
from app.core.interfaces.base import ProcessingStatus
from app.domain.models.voice_profile import VoiceProfile

logger = logging.getLogger(__name__)

# Diretório base para armazenar perfis de voz
VOICE_PROFILES_DIR = Path("app/voice_profiles")

# Duração máxima de áudio aceita (guard contra arquivos muito longos)
MAX_AUDIO_DURATION = 300.0  # 5 minutos


class VoiceProfileService:
    """Gerenciador de perfis de voz com dataset customizado e modelo treinado."""

    def __init__(self, profiles_dir: str = None):
        self.profiles_dir = Path(profiles_dir) if profiles_dir else VOICE_PROFILES_DIR
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
        """Cria um novo perfil de voz e inicializa diretórios.

        Usa flush + update + commit único para transação atômica.
        """
        with self._get_db() as db:
            try:
                profile = VoiceProfile(
                    name=name,
                    telegram_id=telegram_id,
                    phone=phone,
                    email=email,
                    description=description,
                    architecture=architecture,
                    dataset_dir="",  # placeholder
                    status="created",
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
        """
        Adiciona amostras de áudio ao dataset do perfil.

        Args:
            profile_id: ID do perfil
            audio_files: Lista de (filename, file_bytes)

        Returns:
            Dict com resultado da operação
        """
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
                if profile.status == "created":
                    profile.status = "collecting"
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
        """Retorna informações sobre o dataset de um perfil.

        Usa cache de durações para evitar recarregar cada áudio via librosa.
        """
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
                # Tentar cache primeiro
                cached = duration_cache.get(fpath.name)
                if cached:
                    duration = cached.get("duration", 0.0)
                    sr = cached.get("sample_rate", 0)
                else:
                    # Carregar apenas se não estiver no cache
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

        # Salvar cache atualizado
        if cache_updated:
            self._write_duration_cache(profile_id, duration_cache)

        return {
            "files": files,
            "total_samples": len(files),
            "total_duration": round(total_duration, 2),
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
        """
        Treina um modelo de classificação binária para o perfil.

        As amostras de voz do perfil são usadas como classe 'real' (0).
        Fakes sintéticos são gerados via perturbações (pitch shift, noise, etc).
        """
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return {"success": False, "error": "Perfil não encontrado"}

                if profile.num_samples < 3:
                    return {
                        "success": False,
                        "error": "Mínimo de 3 amostras necessárias para treinamento.",
                    }

                arch = architecture or profile.architecture or "sonic_sleuth"
                profile.status = "training"
                profile.architecture = arch
                db.commit()
                if progress_callback:
                    progress_callback("Extraindo features das amostras...")

                samples_dir = self._samples_dir(profile_id)
                real_features = []
                sr_target = 16000

                for audio_file in sorted(samples_dir.iterdir()):
                    if audio_file.suffix.lower() in {
                        ".wav",
                        ".mp3",
                        ".flac",
                        ".ogg",
                        ".m4a",
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

                if len(real_features) < 3:
                    profile.status = "error"
                    db.commit()
                    return {
                        "success": False,
                        "error": (
                            f"Apenas {len(real_features)} amostras válidas. "
                            "Mínimo: 3."
                        ),
                    }

                if progress_callback:
                    progress_callback("Gerando amostras sintéticas (fakes)...")

                fake_features = self._generate_synthetic_fakes(
                    real_features, samples_dir, sr_target
                )

                if progress_callback:
                    progress_callback("Preparando dataset...")

                X_real = np.array(real_features)
                X_fake = np.array(fake_features)

                X = np.vstack([X_real, X_fake])
                y = np.concatenate(
                    [
                        np.zeros(len(X_real)),
                        np.ones(len(X_fake)),
                    ]
                )

                indices = np.random.permutation(len(X))
                X = X[indices]
                y = y[indices]

                split_idx = int(0.8 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                if progress_callback:
                    progress_callback(
                        f"Treinando modelo ({arch}, {epochs} épocas)..."
                    )

                metrics = self._train_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    arch,
                    epochs,
                    batch_size,
                    learning_rate,
                    profile_id,
                    progress_callback,
                )

                if not metrics.get("success", False):
                    profile.status = "error"
                    db.commit()
                    return metrics

                scaler_path = self._model_dir(profile_id) / "scaler.pkl"
                joblib.dump(scaler, scaler_path)

                model_path = self._model_dir(profile_id) / "profile_model.h5"
                profile.model_path = str(model_path)
                profile.scaler_path = str(scaler_path)
                profile.status = "ready"
                profile.training_config = {
                    "architecture": arch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_real_samples": len(X_real),
                    "num_fake_samples": len(X_fake),
                    "feature_dim": X_train.shape[1],
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
                        profile.status = "error"
                        db.commit()
                except Exception:
                    db.rollback()
                return {"success": False, "error": str(e)}

    def _extract_profile_features(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extrai vetor de features para treinamento/inferência de perfil."""
        try:
            features = []

            # MFCCs (13 coefs x stats = 52)
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
        """Gera features de amostras fake via perturbações dos áudios reais."""
        fake_features = []
        audio_files = sorted(
            f for f in samples_dir.iterdir()
            if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        )

        for audio_file in audio_files:
            try:
                y, sr = librosa.load(str(audio_file), sr=sr_target, mono=True)

                # Perturbação 1: Pitch shift (+4 semitons)
                y_pitch_up = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=4)
                feats = self._extract_profile_features(y_pitch_up, sr)
                if feats is not None:
                    fake_features.append(feats)

                # Perturbação 2: Pitch shift (-4 semitons)
                y_pitch_down = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-4)
                feats = self._extract_profile_features(y_pitch_down, sr)
                if feats is not None:
                    fake_features.append(feats)

                # Perturbação 3: Ruído gaussiano forte
                noise = np.random.randn(len(y)) * 0.05
                y_noisy = y + noise
                feats = self._extract_profile_features(y_noisy, sr)
                if feats is not None:
                    fake_features.append(feats)

                # Perturbação 4: Time stretch (mais rápido)
                y_fast = librosa.effects.time_stretch(y=y, rate=1.3)
                feats = self._extract_profile_features(y_fast, sr)
                if feats is not None:
                    fake_features.append(feats)

                # Perturbação 5: Formant shift (pitch + speed combo)
                y_formant = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=6)
                y_formant = librosa.effects.time_stretch(y=y_formant, rate=0.8)
                feats = self._extract_profile_features(y_formant, sr)
                if feats is not None:
                    fake_features.append(feats)

            except Exception as e:
                logger.warning(f"Erro ao gerar fake de {audio_file.name}: {e}")

        # Se houver poucas fakes, adicionar perturbações das features diretamente
        while len(fake_features) < len(real_features):
            idx = np.random.randint(0, len(real_features))
            noise_feat = real_features[idx] + np.random.randn(
                len(real_features[idx])
            ) * 0.3 * np.std(real_features[idx])
            fake_features.append(noise_feat)

        return fake_features

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
        profile_id: int,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Treina um modelo de classificação binária."""
        try:
            import tensorflow as tf

            input_dim = X_train.shape[1]

            # Modelo compacto de classificação binária
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'],
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
                        progress_callback(
                            f"Época {epoch + 1}/{epochs} — "
                            f"loss: {loss:.4f}, acc: {acc:.4f}, "
                            f"val_acc: {val_acc:.4f}"
                        )

            callbacks.append(ProgressCallback())

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            # Salvar modelo
            model_path = self._model_dir(profile_id) / "profile_model.h5"
            model.save(str(model_path))

            # Salvar config de treinamento
            config_path = self._model_dir(profile_id) / "training_config.json"
            training_info = {
                "architecture": architecture,
                "input_dim": input_dim,
                "epochs_trained": len(history.history['loss']),
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
                "epochs_trained": len(history.history['loss']),
                "history": {
                    k: [float(v) for v in vals]
                    for k, vals in history.history.items()
                },
            }

            return {"success": True, "metrics": final_metrics}

        except Exception as e:
            logger.error(f"Erro no treinamento: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    #  Inferência com Modelo do Perfil                                     #
    # ------------------------------------------------------------------ #

    def detect_with_profile(
        self, profile_id: int, audio_path: str
    ) -> Dict[str, Any]:
        """
        Verifica se um áudio é da pessoa do perfil ou não.

        Returns:
            Dict com is_authentic (bool), confidence (float), details (dict)
        """
        with self._get_db() as db:
            try:
                profile = db.query(VoiceProfile).filter_by(id=profile_id).first()
                if not profile:
                    return {"success": False, "error": "Perfil não encontrado"}

                if profile.status != "ready" or not profile.model_path:
                    return {
                        "success": False,
                        "error": "Modelo do perfil ainda não foi treinado ou não está pronto.",
                    }

                # Carregar modelo e scaler
                import tensorflow as tf

                model_path = Path(profile.model_path)
                scaler_path = Path(profile.scaler_path) if profile.scaler_path else None

                if not model_path.exists():
                    return {"success": False, "error": "Arquivo do modelo não encontrado."}

                model = tf.keras.models.load_model(str(model_path))
                scaler = (
                    joblib.load(str(scaler_path))
                    if scaler_path and scaler_path.exists() else None
                )

                # Extrair features do áudio com timeout guard
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
                        "num_training_samples": profile.training_config.get(
                            "num_real_samples", 0
                        ) if profile.training_config else 0,
                        "feature_dim": len(features),
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
        """Carrega cache de durações do disco."""
        cache_path = self._duration_cache_path(profile_id)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_duration_cache(self, profile_id: int, cache: Dict[str, Any]):
        """Salva cache de durações no disco."""
        cache_path = self._duration_cache_path(profile_id)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache de durações: {e}")

    def _save_duration_cache(self, profile_id: int, added_files: List[Dict]):
        """Atualiza cache com arquivos recém-adicionados."""
        cache = self._load_duration_cache(profile_id)
        for item in added_files:
            fname = item.get("filename", "")
            dur = item.get("duration", 0.0)
            if fname:
                cache[fname] = {"duration": dur, "sample_rate": 0}
        self._write_duration_cache(profile_id, cache)

    def _get_cached_duration(self, profile_id: int, filename: str) -> Optional[float]:
        """Retorna duração de uma amostra do cache, ou None."""
        cache = self._load_duration_cache(profile_id)
        entry = cache.get(filename)
        return entry.get("duration") if entry else None

    def _remove_from_duration_cache(self, profile_id: int, filename: str):
        """Remove entrada do cache de durações."""
        cache = self._load_duration_cache(profile_id)
        if filename in cache:
            del cache[filename]
            self._write_duration_cache(profile_id, cache)
