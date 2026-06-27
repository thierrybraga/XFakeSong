import logging
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from app.domain.models.architectures.layers import (
    AttentionLayer,
    AudioFeatureNormalization,
    GraphAttentionLayer,
    SliceLayer,
)

# Importar custom layers para carregar modelos
from app.domain.models.architectures.rawnet2 import (
    AudioNormalizationLayer,
    AudioResamplingLayer,
    MultiScaleConv1DBlock,
)
from app.domain.models.architectures.registry import (
    create_model_by_name,
    get_architecture_info,
    get_available_architectures,
)
from app.domain.models.architectures.safe_normalization import SafeInstanceNormalization

logger = logging.getLogger(__name__)


def _load_custom_architecture_modules() -> None:
    """Import custom architecture modules so Keras registrations are available."""
    for module_name in (
        "app.domain.models.architectures.aasist",
        "app.domain.models.architectures.conformer",
        "app.domain.models.architectures.efficientnet_lstm",
        "app.domain.models.architectures.hybrid_cnn_transformer",
        "app.domain.models.architectures.multiscale_cnn",
        "app.domain.models.architectures.rawnet2",
        "app.domain.models.architectures.rawgat_st",
        "app.domain.models.architectures.wavlm",
        "app.domain.models.architectures.hubert",
        "app.domain.models.architectures.sonic_sleuth",
        "app.domain.models.architectures.spectrogram_transformer",
        "app.domain.models.architectures.ensemble",
        "app.domain.models.training.optimization",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - best-effort registry hydration
            logger.debug("Falha ao importar %s para custom_objects: %s", module_name, exc)


@dataclass
class ModelInfo:
    """Informações sobre um modelo carregado."""
    name: str
    architecture: str
    model: Any
    scaler: Optional[StandardScaler]
    input_shape: tuple
    model_type: str  # 'tensorflow' ou 'sklearn'
    input_contract: Optional[Dict[str, Any]] = None  # Contrato train/inference
    # Sprint 1.4: temperatura calibrada (post-hoc temperature scaling).
    # T=1.0 significa sem calibração. T>1.0 reduz a confiança (modelo overconfident),
    # T<1.0 aumenta a confiança (modelo underconfident).
    temperature: float = 1.0
    # Sprint 3.1: função tf.function(jit_compile=True) cacheada para predict
    # rápida. Construída sob demanda pelo Predictor (lazy).
    jit_predict_fn: Optional[Any] = None
    # Sprint 3.3: flag indicando se warm-up já foi feito (1 forward pass de
    # dummy data para forçar JIT/compile no load)
    warmed_up: bool = False
    # Sprint 4.5: EER threshold calibrado no val set durante o treino.
    # Se != 0.5, predictor pode usar via flag use_eer_threshold para
    # classificação adaptativa (FAR=FRR ótimo). None = não calibrado.
    eer_threshold: Optional[float] = None
    eer_value: Optional[float] = None
    # Tier-1 perf: sessão ONNX Runtime, criada quando há um `<name>.onnx` ao lado
    # do modelo e `onnxruntime` está instalado. Quando presente, o Predictor a
    # usa para inferência (FP32, mais rápida em CPU; mesmos pesos → mesma saída),
    # com fallback automático para o modelo Keras.
    onnx_session: Optional[Any] = None


class TorchSSLOriginalModel:
    """Lazy PyTorch inference wrapper for original WavLM/HuBERT benchmark models."""

    def __init__(self, artifact_path: Path, metadata: Dict[str, Any]):
        self.artifact_path = Path(artifact_path)
        self.metadata = metadata or {}
        self.backbone_dir = self._resolve_backbone_dir()
        self.device = None
        self.torch = None
        self.backbone = None
        self.classifier = None
        self._loaded = False

    def _resolve_backbone_dir(self) -> Path:
        architecture = str(self.metadata.get("architecture", "")).lower()
        if "hubert" in architecture or "hubert" in self.artifact_path.stem.lower():
            default_name = "hubert_backbone"
        else:
            default_name = "wavlm_backbone"

        candidates = []
        raw_artifact = self.metadata.get("backbone_artifact")
        if raw_artifact:
            candidates.append(Path(raw_artifact))
        candidates.append(self.artifact_path.parent / default_name)
        candidates.append(
            self.artifact_path.parent
            / "benchmark_final"
            / self.artifact_path.stem.replace("bench_", "")
            / default_name
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _normalize_wave_batch(features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        flat = x.reshape(len(x), -1)
        target_len = 16000
        if flat.shape[1] > target_len:
            start = max(0, (flat.shape[1] - target_len) // 2)
            flat = flat[:, start:start + target_len]
        elif flat.shape[1] < target_len:
            repeats = int(np.ceil(target_len / max(1, flat.shape[1])))
            flat = np.tile(flat, (1, repeats))[:, :target_len]
        mean = flat.mean(axis=1, keepdims=True)
        std = flat.std(axis=1, keepdims=True)
        flat = (flat - mean) / np.maximum(std, 1e-6)
        return np.clip(flat, -5.0, 5.0).astype(np.float32)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import HubertModel, WavLMModel
        except Exception as exc:  # noqa: BLE001 - dependency surfaced to UI
            raise RuntimeError(
                "Modelos WavLM/HuBERT originais exigem torch e transformers "
                "instalados no ambiente de inferência."
            ) from exc

        checkpoint = torch.load(
            self.artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        model_class = checkpoint.get(
            "model_class",
            self.metadata.get("model_class", "WavLMModel"),
        )
        backbone_cls = HubertModel if "Hubert" in str(model_class) else WavLMModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone_cls.from_pretrained(str(self.backbone_dir)).to(
            self.device
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = int(
            checkpoint.get(
                "hidden_size",
                getattr(self.backbone.config, "hidden_size", 768),
            )
        )
        dropout = float(
            self.metadata.get(
                "dropout",
                checkpoint.get("training_config", {}).get("dropout", 0.2),
            )
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.classifier.eval()
        self.torch = torch
        self._loaded = True

    def predict(self, features: np.ndarray, batch_size: int = 8) -> np.ndarray:
        self._ensure_loaded()
        assert self.torch is not None
        assert self.backbone is not None
        assert self.classifier is not None

        x = self._normalize_wave_batch(features)
        outputs = []
        with self.torch.no_grad():
            for start in range(0, len(x), max(1, int(batch_size))):
                xb = self.torch.from_numpy(x[start:start + batch_size]).to(
                    self.device
                )
                backbone_out = self.backbone(
                    xb,
                    output_hidden_states=False,
                    return_dict=True,
                )
                pooled = backbone_out.last_hidden_state.mean(dim=1)
                logits = self.classifier(pooled)
                probs = self.torch.softmax(logits, dim=1)
                outputs.append(probs.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


class ModelLoader:
    """Responsável por carregar e gerenciar modelos."""

    def __init__(
        self,
        models_dir: Union[str, Path],
        create_default_models: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.available_model_names: List[str] = []
        self.default_model = None
        self.create_default_models = create_default_models

    def _discover_model_files(self) -> List[Path]:
        """Return supported model artifacts in deterministic preference order.

        Cobre tanto artefatos na raiz de ``models_dir`` quanto os modelos
        PROMOVIDOS do benchmark em ``benchmark_final/<arch>/bench_<arch>.*``.
        Sem a busca em ``benchmark_final`` (P0), a descoberta nao-recursiva nao
        encontrava os modelos treinados e o servico caia nos modelos de
        demonstracao (`_create_default_models`). Restringe ao prefixo ``bench_``
        sob ``benchmark_final/<arch>/`` para nao capturar checkpoints
        intermediarios (ex.: ``results/.../models/best_checkpoint.keras``).
        """
        exts = ("*.keras", "*.h5", "*.pkl", "*.pt")
        files: List[Path] = []
        for ext in exts:
            files.extend(self.models_dir.glob(ext))
        bench_final = self.models_dir / "benchmark_final"
        if bench_final.is_dir():
            for ext in ("bench_*.keras", "bench_*.h5", "bench_*.pkl", "bench_*.pt"):
                files.extend(bench_final.glob(f"*/{ext}"))
        seen: set = set()
        unique: List[Path] = []
        for fp in files:
            if "_scaler" in fp.stem:
                continue
            key = fp.resolve()
            if key in seen:
                continue
            seen.add(key)
            unique.append(fp)
        return sorted(unique, key=lambda fp: fp.stem)

    def load_available_models(self):
        """Descobre modelos disponíveis sem carregar todos os pesos no startup."""
        logger.info("Descobrindo modelos disponíveis em %s...", self.models_dir)

        model_files = self._discover_model_files()
        self.available_model_names = [p.stem for p in model_files]

        # Se não há artefatos, criar modelos padrão leves para demonstração.
        if not self.available_model_names and self.create_default_models:
            logger.info(
                "Nenhum modelo salvo encontrado. Criando modelos padrão...")
            self._create_default_models()
            self.available_model_names = sorted(self.loaded_models.keys())
        elif not self.available_model_names:
            logger.info(
                "Nenhum modelo salvo encontrado. Criação de modelos padrão "
                "desativada."
            )

        # Definir modelo padrão
        if self.available_model_names:
            self.default_model = self.available_model_names[0]
            logger.info(f"Modelo padrão definido: {self.default_model}")

    def _load_single_model(self, model_path: Path, warmup: bool = True):
        """Carrega um único modelo.

        Args:
            warmup: se True, faz o warm-up (1 forward pass) — apropriado quando o
                modelo está sendo carregado sob demanda (vai ser usado já já).
                `load_available_models` passa False e aquece só o default, para
                não pagar N forward-passes no startup.
        """
        model_name = model_path.stem
        metadata = {}

        # Tentar carregar metadados se existirem
        config_path = model_path.parent / f"{model_name}_config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar config para {model_name}: {e}")

        try:
            if model_path.suffix in ('.h5', '.keras'):
                # Modelo TensorFlow/Keras — suporta ambos os formatos
                _load_custom_architecture_modules()
                custom_objects = {
                    'AudioResamplingLayer': AudioResamplingLayer,
                    'AudioNormalizationLayer': AudioNormalizationLayer,
                    'MultiScaleConv1DBlock': MultiScaleConv1DBlock,
                    'AudioFeatureNormalization': AudioFeatureNormalization,
                    'AttentionLayer': AttentionLayer,
                    'GraphAttentionLayer': GraphAttentionLayer,
                    'SliceLayer': SliceLayer,
                    'SafeInstanceNormalization': SafeInstanceNormalization
                }
                try:
                    from app.domain.models.architectures.wavlm import (
                        WavLMFeatureExtractor,
                    )

                    custom_objects['WavLMFeatureExtractor'] = WavLMFeatureExtractor
                except Exception as exc:  # noqa: BLE001 - optional SSL loader
                    logger.debug(
                        "WavLMFeatureExtractor indisponível para load: %s", exc
                    )
                try:
                    from app.domain.models.architectures.hubert import (
                        HuBERTFeatureExtractor,
                    )

                    custom_objects['HuBERTFeatureExtractor'] = HuBERTFeatureExtractor
                except Exception as exc:  # noqa: BLE001 - optional SSL loader
                    logger.debug(
                        "HuBERTFeatureExtractor indisponível para load: %s", exc
                    )
                custom_objects.update(tf.keras.utils.get_custom_objects())

                try:
                    model = tf.keras.models.load_model(
                        str(model_path),
                        custom_objects=custom_objects,
                        safe_mode=False,
                        compile=False,
                    )
                except TypeError:
                    # Fallback sem custom objects se não forem necessários
                    model = tf.keras.models.load_model(
                        str(model_path),
                        safe_mode=False,
                        compile=False,
                    )

                model_type = 'tensorflow'

                # Tentar carregar scaler correspondente
                scaler_path = model_path.parent / f"{model_name}_scaler.pkl"
                scaler = None
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)

                # Inferir input_shape do modelo
                input_shape = model.input_shape[1:]  # Remove batch dimension

            elif model_path.suffix == '.pkl':
                # Modelo sklearn
                if '_scaler' in model_name:
                    return  # Skip scaler files

                model = joblib.load(model_path)
                model_type = 'sklearn'

                # Carregar scaler correspondente
                scaler_path = model_path.parent / f"{model_name}_scaler.pkl"
                scaler = None
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)

                # Para modelos sklearn, input_shape será definido durante a
                # predição
                input_shape = None

            elif model_path.suffix == '.pt':
                model = TorchSSLOriginalModel(model_path, metadata)
                model_type = 'pytorch_transformers'
                scaler = None
                raw_shape = metadata.get('input_shape', [16000, 1])
                input_shape = tuple(raw_shape)

            else:
                logger.warning(
                    f"Formato de arquivo não suportado: {model_path}")
                return

            # Determinar arquitetura: via metadados ou inferência
            if 'architecture' in metadata:
                architecture = metadata['architecture']
            else:
                architecture = self._infer_architecture_from_name(model_name)

            # Sobrescrever input_shape se definido nos metadados
            if 'input_shape' in metadata:
                input_shape = tuple(metadata['input_shape'])

            # Extrair input_contract dos metadados (salvo pelo trainer)
            input_contract = metadata.get('input_contract', None)

            # Sprint 1.4: extrair temperatura calibrada do input_contract.
            # Default 1.0 (sem calibração) se o modelo é legado ou não foi calibrado.
            temperature = 1.0
            # Sprint 4.5: EER threshold calibrado (None se modelo legado)
            eer_threshold: Optional[float] = None
            eer_value: Optional[float] = None
            if input_contract and isinstance(input_contract, dict):
                try:
                    temperature = float(input_contract.get('temperature', 1.0))
                except (TypeError, ValueError):
                    temperature = 1.0
                # EER fields (Sprint 4.5)
                eer_t_raw = input_contract.get('eer_threshold')
                if eer_t_raw is not None:
                    try:
                        eer_threshold = float(eer_t_raw)
                    except (TypeError, ValueError):
                        pass
                eer_v_raw = input_contract.get('eer_value')
                if eer_v_raw is not None:
                    try:
                        eer_value = float(eer_v_raw)
                    except (TypeError, ValueError):
                        pass

            model_info = ModelInfo(
                name=model_name,
                architecture=architecture,
                model=model,
                scaler=scaler,
                input_shape=input_shape,
                model_type=model_type,
                input_contract=input_contract,
                temperature=temperature,
                eer_threshold=eer_threshold,
                eer_value=eer_value,
            )

            # Tier-1 perf: se houver um `<name>.onnx` ao lado e onnxruntime
            # instalado, prepara uma sessão ONNX Runtime para inferência (FP32,
            # mesmos pesos → mesma saída, porém mais rápida em CPU). Degrada
            # graciosamente: sem .onnx ou sem onnxruntime → segue com Keras/TF.
            if model_type == 'tensorflow':
                onnx_path = model_path.parent / f"{model_name}.onnx"
                if onnx_path.exists():
                    try:
                        from app.domain.models.inference.onnx_export import (
                            OnnxInferenceSession,
                        )
                        model_info.onnx_session = OnnxInferenceSession(str(onnx_path))
                        logger.info(
                            f"ONNX session ativa para {model_name} "
                            f"(inferência acelerada, fallback TF disponível)")
                    except Exception as e:
                        logger.debug(
                            f"ONNX indisponível para {model_name} "
                            f"(usando TF): {e}")
                        model_info.onnx_session = None

            # Sprint 3.3: warm-up do modelo (1 forward pass com zeros) para
            # forçar JIT compile / layer init / GPU memory allocation.
            # Primeira inferência fica ~10× mais rápida.
            # Tier-1 perf: condicional — no startup (load_available_models)
            # aquecemos só o modelo default; os demais aquecem no 1º uso.
            if warmup and model_type == 'tensorflow':
                self._warmup_model(model_info)

            self.loaded_models[model_name] = model_info
            calib_str = (
                f" | T={temperature:.3f} (calibrado)" if temperature != 1.0 else ""
            )
            warmup_str = " | warmed-up" if model_info.warmed_up else ""
            logger.info(
                f"Modelo {model_name} carregado com sucesso "
                f"({model_type}){calib_str}{warmup_str}")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_path}: {e}")
            raise

    def _warmup_model(self, model_info: 'ModelInfo') -> None:
        """Sprint 3.3: warm-up com 1 forward pass de zeros.

        Força:
        - Alocação de memória GPU (lazy em TF)
        - Compilação inicial do grafo
        - JIT compile (se tf.function for usado depois)
        - Initialização de layers preguiçosas

        Resultado: primeira inferência real fica ~5–10× mais rápida.
        Falhas são silenciosas (apenas warning) — não afetam o carregamento.
        """
        if model_info.input_shape is None:
            return
        try:
            import numpy as np
            # Cria tensor de zeros no shape esperado pelo modelo
            shape = (1,) + tuple(
                int(d) if d is not None else 1 for d in model_info.input_shape
            )
            dummy = np.zeros(shape, dtype=np.float32)
            # Forward pass silencioso
            _ = model_info.model.predict(dummy, verbose=0, batch_size=1)
            model_info.warmed_up = True
            logger.debug(f"Warm-up OK para {model_info.name} (shape={shape})")
        except Exception as e:
            # Não-crítico: warm-up é otimização, não correctness
            logger.debug(f"Warm-up falhou para {model_info.name}: {e}")
            model_info.warmed_up = False

    def _infer_architecture_from_name(self, model_name: str) -> str:
        """Infere a arquitetura baseada no nome do modelo."""
        name_lower = model_name.lower()

        if 'aasist' in name_lower:
            return 'AASIST'
        elif 'rawgat' in name_lower:
            return 'RawGAT-ST'
        elif 'efficientnet' in name_lower:
            return 'EfficientNet-LSTM'
        elif 'multiscale' in name_lower:
            return 'MultiscaleCNN'
        elif 'conformer' in name_lower:
            return 'Conformer'
        elif 'hybrid' in name_lower:
            return 'Hybrid CNN-Transformer'
        elif 'spectrogram' in name_lower or 'transformer' in name_lower:
            return 'SpectrogramTransformer'
        elif 'ensemble' in name_lower:
            return 'Ensemble'
        elif 'rawnet2' in name_lower:
            return 'RawNet2'
        elif 'wavlm' in name_lower:
            return 'WavLM'
        elif 'hubert' in name_lower:
            return 'HuBERT'
        elif 'sonic' in name_lower or 'sleuth' in name_lower:
            return 'Sonic Sleuth'
        elif 'svm' in name_lower:
            return 'SVM'
        elif 'random_forest' in name_lower or 'randomforest' in name_lower or 'rf' in name_lower:
            return 'RandomForest'
        elif 'neural_network' in name_lower:
            return 'SimpleNN'
        else:
            return 'Unknown'

    def _create_default_models(self):
        """Cria modelos padrão para demonstração."""
        logger.info("Criando modelos padrão...")

        input_shape = (100, 80)  # Formato padrão

        # Criar alguns modelos leves para demonstração
        lightweight_architectures = ['MultiscaleCNN', 'EfficientNet-LSTM']

        for arch_name in lightweight_architectures:
            try:
                logger.info(f"Criando modelo {arch_name}...")

                # Criar modelo usando variant lite se disponível
                arch_info = get_architecture_info(arch_name)
                variant = None
                if 'lite' in [
                        v for v in arch_info.supported_variants if 'lite' in v
                ]:
                    variant = [
                        v for v in arch_info.supported_variants if 'lite' in v
                    ][0]

                model = create_model_by_name(
                    arch_name,
                    input_shape,
                    num_classes=2,
                    variant=variant
                )

                model_info = ModelInfo(
                    name=f"{arch_name.lower()}_default",
                    architecture=arch_name,
                    model=model,
                    scaler=StandardScaler(),
                    input_shape=input_shape,
                    model_type='tensorflow'
                )

                self.loaded_models[model_info.name] = model_info
                logger.info(f"Modelo {arch_name} criado com sucesso")

            except Exception as e:
                logger.warning(f"Erro ao criar modelo {arch_name}: {e}")

    def get_model(self, model_name: str) -> Optional['ModelInfo']:
        """Retorna ModelInfo por nome com carregamento lazy.

        Primeiro verifica o cache de modelos carregados. Se não encontrado,
        tenta localizar e carregar o arquivo correspondente no models_dir.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        # Lazy loading: busca nos formatos suportados (ordem de preferência)
        for suffix in ('.keras', '.h5', '.pkl', '.pt'):
            model_file = self.models_dir / f"{model_name}{suffix}"
            if model_file.exists():
                try:
                    self._load_single_model(model_file)
                    return self.loaded_models.get(model_name)
                except Exception as e:
                    logger.warning(
                        f"Falha ao carregar modelo '{model_name}' de {model_file}: {e}")

        logger.warning(f"Modelo '{model_name}' não encontrado em {self.models_dir}")
        return None

    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis (arquivos ou carregados)."""
        discovered = {p.stem for p in self._discover_model_files()}
        discovered.update(self.available_model_names)
        discovered.update(self.loaded_models.keys())
        return sorted(discovered)

    def get_available_architectures(self) -> List[str]:
        """Retorna lista de arquiteturas disponíveis."""
        return get_available_architectures()

    def find_model(self, architecture: str,
                   variant: str = None) -> Optional[str]:
        """Encontra um modelo disponível que corresponda à arquitetura."""
        arch_lower = architecture.lower()
        variant_lower = variant.lower() if variant else None

        # Verificar em todos os modelos disponíveis (arquivos + carregados)
        all_models = self.get_available_models()

        for name in all_models:
            # Precisamos inferir a arquitetura se não estiver carregado
            if name in self.loaded_models:
                model_arch = self.loaded_models[name].architecture
            else:
                model_arch = self._infer_architecture_from_name(name)

            if model_arch.lower() == arch_lower:
                if variant_lower:
                    if variant_lower in name.lower():
                        return name
                else:
                    return name
        return None
