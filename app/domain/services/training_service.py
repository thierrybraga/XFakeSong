import dataclasses
import importlib as _stdlib_importlib
import json
import logging
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from app.core.config.settings import TrainingConfig
from app.core.interfaces.base import ProcessingStatus
from app.core.interfaces.services import (
    ITrainingService,
    ModelMetadata,
    ProcessingResult,
)
from app.domain.models.architectures.registry import get_architecture_info
from app.domain.models.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)
importlib = types.SimpleNamespace(import_module=_stdlib_importlib.import_module)


class TrainingService(ITrainingService):
    """
    Serviço responsável por gerenciar processos de treinamento.
    Integra ModelTrainer com o restante do sistema.
    """

    # Chaves que NÃO são parâmetros do CONSTRUTOR do modelo. O `default_params`
    # do registry mistura args do create_model (base_width, scale, dropout…)
    # com campos de TrainingConfig (epochs, learning_rate…) e hints de pipeline
    # (patience, lr_patience, gradient_clip, augmentation_strength). Passar
    # esses ao create_model quebra builders cujo inner não aceita **kwargs
    # (ex.: _create_res2net_model → "unexpected keyword argument 'patience'").
    # Derivado dinamicamente do dataclass para não desatualizar.
    _NON_MODEL_PARAM_KEYS = (
        {f.name for f in dataclasses.fields(TrainingConfig)}
        | {
            "patience",
            "lr_patience",
            "gradient_clip",
            "augmentation_strength",
            "model_name",
            "num_classes",
            "parameters",
            "architecture",
            "dataset_path",
            "use_mfcc_branch",
            "use_cross_attention",
            "use_gated_fusion",
            "use_se_blocks",
            "aux_loss_weight",
            "use_mixed_precision",
        }
    )

    def __init__(self, models_dir: str = "app/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._label_classes: list = []   # mapping de classes detectadas

    @staticmethod
    def _normalize_labels(y_raw: np.ndarray) -> tuple:
        """Normaliza labels para [0, K-1) contíguo (BUG.Training.1/2/3).

        Datasets reais (BRSpeech-DF, ASVspoof, In-the-Wild) podem trazer:
          - Labels esparsos: {0, 3} em vez de {0, 1}
          - Labels string: ['bonafide', 'spoof']
          - Labels float: [0.0, 1.0]
          - Labels one-hot: [[1,0], [0,1]] (não toca — só ndim==1)

        Args:
            y_raw: array 1D de labels brutas

        Returns:
            (y_normalized, label_classes) onde y_normalized contém ints
            em [0, K-1) e label_classes[i] = valor ORIGINAL da classe i.
            Usar `label_classes[predicted_class]` para inverter na inferência.
        """
        y_arr = np.asarray(y_raw)

        # Se já é one-hot (N, K), não remapear
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            n_classes = y_arr.shape[-1]
            label_classes = list(range(n_classes))
            logger.info(
                f"Labels já em formato one-hot (N, {n_classes}) — sem remapeamento"
            )
            return y_arr, label_classes

        # Achata para 1D
        y_flat = y_arr.ravel()

        # Detecta valores únicos preservando ordem de aparição (estável)
        unique_vals = []
        seen = set()
        for v in y_flat:
            # Converte tipos numpy para Python nativo para hash estável
            key = v.item() if hasattr(v, "item") else v
            if key not in seen:
                seen.add(key)
                unique_vals.append(key)

        # Ordena para mapping consistente entre runs
        try:
            unique_vals_sorted = sorted(unique_vals)
        except TypeError:
            # tipos mistos — mantém ordem de aparição
            unique_vals_sorted = unique_vals

        # Mapping: valor_original → índice [0, K-1)
        value_to_idx = {v: i for i, v in enumerate(unique_vals_sorted)}

        # Verifica se já está em [0, K-1) contíguo — atalho rápido
        n_classes = len(unique_vals_sorted)
        is_already_contiguous = (
            n_classes > 0
            and all(isinstance(v, (int, np.integer)) for v in unique_vals_sorted)
            and unique_vals_sorted == list(range(n_classes))
        )

        if is_already_contiguous:
            logger.info(
                f"Labels já em [0, {n_classes}) contíguo — sem remapeamento. "
                f"Classes: {unique_vals_sorted}"
            )
            return y_flat.astype(np.int32), list(unique_vals_sorted)

        # Aplica mapping
        y_norm = np.array(
            [value_to_idx[v.item() if hasattr(v, "item") else v]
             for v in y_flat],
            dtype=np.int32,
        )
        logger.info(
            f"Labels remapeadas: {unique_vals_sorted} -> [0, {n_classes}). "
            f"Mapping: {value_to_idx}"
        )
        return y_norm, list(unique_vals_sorted)

    @staticmethod
    def _apply_label_mapping(y_raw: np.ndarray, label_classes: list) -> np.ndarray:
        """Aplica o mesmo mapping ao val/test set (BUG.Training.3).

        Levanta erro se encontrar label não-presente no mapping do treino.
        """
        if not label_classes:
            return np.asarray(y_raw, dtype=np.int32)

        y_arr = np.asarray(y_raw)
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            # one-hot já está OK
            return y_arr

        y_flat = y_arr.ravel()
        value_to_idx = {v: i for i, v in enumerate(label_classes)}

        try:
            return np.array(
                [value_to_idx[v.item() if hasattr(v, "item") else v]
                 for v in y_flat],
                dtype=np.int32,
            )
        except KeyError as e:
            unknown = e.args[0]
            raise ValueError(
                f"Label '{unknown}' no val/test set não existe no train "
                f"set. Classes do train: {label_classes}. "
                f"Verifique se train/val foram split do mesmo dataset."
            )

    def train_model(self, architecture: str, dataset_path: str,
                    config: Dict[str, Any]) -> ProcessingResult[ModelMetadata]:
        """
        Inicia o treinamento de um modelo.
        """
        try:
            logger.info(
                f"Iniciando serviço de treinamento para {architecture}")

            # 1. Validar Arquitetura
            arch_info = get_architecture_info(architecture)
            if not arch_info:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Arquitetura '{architecture}' não encontrada."]
                )

            # 2. Carregar Dados
            # Suporte inicial para arquivos .npz (padrão numpy)
            data_path = Path(dataset_path)
            if not data_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Dataset não encontrado: {dataset_path}"]
                )

            try:
                if data_path.suffix == '.npz':
                    data = np.load(data_path)
                    if 'X_train' not in data or 'y_train' not in data:
                        return ProcessingResult(
                            status=ProcessingStatus.ERROR,
                            errors=[
                                "Arquivo .npz deve conter chaves "
                                "'X_train' e 'y_train'"
                            ]
                        )

                    X_train = data['X_train']
                    y_train = data['y_train']

                    validation_data = None
                    if 'X_val' in data and 'y_val' in data:
                        validation_data = (data['X_val'], data['y_val'])

                    # BUG.Training.1/2/3: normaliza labels para [0, K-1).
                    # Datasets como BRSpeech-DF/ASVspoof podem trazer labels
                    # não-contíguos (ex: {0, 3}). sparse_categorical_crossentropy
                    # exige labels em [0, num_classes). Auto-remapeia + loga
                    # + salva mapping para inferência inversa.
                    y_train, label_classes = self._normalize_labels(y_train)
                    self._label_classes = label_classes  # salva para contract
                    if validation_data is not None:
                        # Aplica MESMO mapping ao val set para consistência
                        X_val, y_val_raw = validation_data
                        y_val = self._apply_label_mapping(
                            y_val_raw, label_classes
                        )
                        validation_data = (X_val, y_val)

                else:
                    # Apenas datasets .npz (X_train/y_train) são suportados por
                    # este caminho. CSV/tabular segmentado não é aceito aqui —
                    # gere um .npz pelo pipeline de extração de features.
                    return ProcessingResult(
                        status=ProcessingStatus.ERROR,
                        errors=[
                            "Formato não suportado: "
                            f"{data_path.suffix or '(sem extensão)'}. "
                            "Forneça um .npz com X_train/y_train."
                        ]
                    )
            except Exception as e:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Erro ao carregar dataset: {str(e)}"]
                )

            # 3. Instanciar Modelo
            try:
                if config.get("use_mixed_precision") is False:
                    try:
                        import tensorflow as tf

                        tf.keras.mixed_precision.set_global_policy("float32")
                        logger.info(
                            "Mixed precision desabilitado antes da instanciação "
                            "do modelo."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Falha ao definir política float32 antes do modelo: {e}"
                        )
                module = importlib.import_module(arch_info.module_path)
                create_model_fn = getattr(module, arch_info.function_name)

                # Mesclar parâmetros padrão com os fornecidos
                merged_params = arch_info.default_params.copy()
                if 'parameters' in config:
                    merged_params.update(config['parameters'])

                # Separa params do MODELO dos hints de treino/pipeline. Só os
                # primeiros vão ao create_model (senão builders sem **kwargs
                # quebram, ex.: Res2Net com 'patience').
                model_params = {
                    k: v for k, v in merged_params.items()
                    if k not in self._NON_MODEL_PARAM_KEYS
                }
                # Aproveita os hints de patience recomendados pelo registry como
                # DEFAULT do TrainingConfig (se o usuário não os definiu no config).
                self._recommended_training = {}
                if 'patience' in merged_params:
                    self._recommended_training['early_stopping_patience'] = (
                        merged_params['patience']
                    )
                if 'lr_patience' in merged_params:
                    self._recommended_training['reduce_lr_patience'] = (
                        merged_params['lr_patience']
                    )

                # Determinar input shape
                # Assumindo (batch, time, feats) ou (batch, feats)
                input_shape = X_train.shape[1:]

                # num_classes: usa valor do config OU usa label_classes
                # detectado por _normalize_labels (mais confiável que unique).
                detected_classes = (
                    len(getattr(self, '_label_classes', []) or [])
                    or int(np.unique(y_train).size)
                )
                num_classes = config.get('num_classes', detected_classes)
                num_classes = max(num_classes, 2)  # mínimo 2 classes

                model = create_model_fn(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **model_params)
            except Exception as e:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Erro ao instanciar modelo: {str(e)}"]
                )

            # 4. Configurar e Executar Treinamento
            # Filtra SOMENTE campos válidos do TrainingConfig — o config do
            # caller pode trazer 'parameters', 'model_name', 'num_classes',
            # 'architecture' etc., que quebrariam TrainingConfig(**dict).
            valid_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
            train_conf_dict = {
                k: v for k, v in config.items() if k in valid_fields
            }
            # Aplica os defaults recomendados pelo registry (patience etc.) sem
            # sobrescrever o que o usuário definiu explicitamente.
            for k, v in getattr(self, '_recommended_training', {}).items():
                train_conf_dict.setdefault(k, v)

            training_config = TrainingConfig(**train_conf_dict)
            trainer = ModelTrainer(
                training_config,
                use_mixed_precision=config.get("use_mixed_precision"),
            )

            # kwargs para callbacks podem ser passados via config
            train_result = trainer.train(
                model,
                (X_train, y_train),
                validation_data=validation_data
            )

            if train_result.status != ProcessingStatus.SUCCESS:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=train_result.errors
                )

            # 5. Salvar Modelo e Metadados
            model_name = config.get(
                'model_name',
                f"{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            save_path = self.models_dir / f"{model_name}.keras"

            save_result = trainer.save_model(model, save_path)
            if save_result.status != ProcessingStatus.SUCCESS:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=save_result.errors
                )

            # Salvar config do modelo para carregamento futuro.
            # BUG.Training.3: salva label_classes para inverter mapping na inferência.
            config_path = self.models_dir / f"{model_name}_config.json"
            label_classes = getattr(self, '_label_classes', []) or []
            # Garante JSON-serializable (numpy types não são por default)
            label_classes_json = [
                v.item() if hasattr(v, "item") else v for v in label_classes
            ]

            # input_contract: PRESERVA a calibração feita no train() (temperatura,
            # EER e OOD thresholds — Sprints 1.4/2.5/4.5). Sem isto, este JSON
            # sobrescreveria o config que trainer.save_model gravou e a calibração
            # seria SILENCIOSAMENTE perdida no reload (Predictor cairia em T=1.0 e
            # threshold 0.5). Reconstruído a partir do trainer (atributos setados
            # durante o treino), com a arquitetura real injetada.
            try:
                input_contract = trainer._build_input_contract(
                    model, metadata={"architecture": architecture}
                )
            except Exception as _e:
                logger.warning(f"Falha ao construir input_contract: {_e}")
                input_contract = {}
            if label_classes_json:
                # Co-loca label_classes no contrato (além do top-level) para que
                # o mapping inverso fique junto do resto do contrato de inferência.
                input_contract["label_classes"] = label_classes_json

            model_metadata = {
                "architecture": architecture,
                "input_shape": list(input_shape),
                "num_classes": num_classes,
                "label_classes": label_classes_json,
                "model_type": "tensorflow",
                "created_at": str(datetime.now()),
                "metrics": train_result.data,
                "input_contract": input_contract,
            }
            with open(config_path, 'w') as f:
                json.dump(model_metadata, f, indent=4, default=str)

            # Extract metrics
            metrics = train_result.data or {}

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=ModelMetadata(
                    name=model_name,
                    version="1.0.0",
                    architecture=architecture,
                    created_at=datetime.now(),
                    metrics=metrics,
                    file_path=Path(save_path),
                    accuracy=metrics.get("accuracy", 0.0),
                    precision=metrics.get("precision", 0.0),
                    recall=metrics.get("recall", 0.0),
                    f1_score=metrics.get("f1", 0.0),
                    training_dataset=str(dataset_path),
                    file_size=(
                        save_path.stat().st_size if save_path.exists() else 0
                    )
                ),
                metadata={"model": model},
            )

        except Exception as e:
            logger.error(
                f"Erro não tratado no TrainingService: {e}", exc_info=True)
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro interno: {str(e)}"]
            )

    def cross_validate_model(
        self,
        architecture: str,
        dataset_path: str,
        config: Dict[str, Any],
        n_folds: int = 5,
        save_fold_models: bool = False,
    ) -> ProcessingResult[Dict[str, Any]]:
        """Sprint 4.1: K-fold Cross Validation estratificado.

        Divide o dataset em K folds (preservando proporção de classes),
        treina K modelos independentes e retorna métricas agregadas
        (mean ± std). Útil para validar a estabilidade do modelo em
        datasets pequenos/médios.

        Args:
            architecture: nome da arquitetura (igual a train_model)
            dataset_path: caminho .npz com X_train, y_train
            config: config de treinamento (igual a train_model)
            n_folds: número de folds (default 5; usar 10 em datasets grandes)
            save_fold_models: se True, salva modelo de cada fold em
                models/<name>_fold{i}.keras

        Returns:
            ProcessingResult com dict:
                - per_fold: list[dict] com métricas de cada fold
                - aggregated: dict {metric: {'mean': float, 'std': float}}
                - best_fold: int (fold com melhor val_accuracy)
                - n_folds: int
        """
        try:
            from sklearn.model_selection import StratifiedKFold

            logger.info(
                f"Iniciando K-fold CV ({n_folds} folds) para {architecture}"
            )

            # 1. Valida arquitetura
            arch_info = get_architecture_info(architecture)
            if not arch_info:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Arquitetura '{architecture}' não encontrada."],
                )

            # 2. Carrega dados
            data_path = Path(dataset_path)
            if not data_path.exists() or data_path.suffix != '.npz':
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Dataset inválido: {dataset_path} (precisa ser .npz)"],
                )

            data = np.load(data_path)
            if 'X_train' not in data or 'y_train' not in data:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["NPZ precisa ter 'X_train' e 'y_train'"],
                )
            X = data['X_train']
            y = data['y_train']

            # Para StratifiedKFold precisa de labels 1D
            y_for_split = y if y.ndim == 1 else np.argmax(y, axis=-1) \
                          if y.shape[-1] > 1 else y.ravel()
            y_for_split = y_for_split.astype(int)

            # 3. K-fold splits estratificados
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42
            )

            per_fold_results: list = []
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_for_split)):
                logger.info(
                    f"=== Fold {fold_idx + 1}/{n_folds} "
                    f"(train={len(train_idx)}, val={len(val_idx)}) ==="
                )

                # Salva temporariamente em NPZ para reusar train_model
                fold_npz = data_path.parent / f".cv_fold{fold_idx}.npz"
                try:
                    np.savez(
                        fold_npz,
                        X_train=X[train_idx], y_train=y[train_idx],
                        X_val=X[val_idx], y_val=y[val_idx],
                    )

                    # Config específica do fold (nome único)
                    fold_config = dict(config)
                    fold_config['model_name'] = (
                        f"{architecture}_fold{fold_idx}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    fold_result = self.train_model(
                        architecture=architecture,
                        dataset_path=str(fold_npz),
                        config=fold_config,
                    )

                    if fold_result.status != ProcessingStatus.SUCCESS:
                        logger.warning(
                            f"Fold {fold_idx} falhou: {fold_result.errors}"
                        )
                        per_fold_results.append({
                            'fold': fold_idx,
                            'status': 'error',
                            'errors': fold_result.errors,
                        })
                        continue

                    md = fold_result.data
                    per_fold_results.append({
                        'fold': fold_idx,
                        'status': 'success',
                        'accuracy': md.accuracy,
                        'precision': md.precision,
                        'recall': md.recall,
                        'f1_score': md.f1_score,
                        'model_path': str(md.file_path) if save_fold_models else None,
                        'metrics': dict(md.metrics) if md.metrics else {},
                    })

                    # Remove modelo do fold se save_fold_models=False
                    if not save_fold_models:
                        try:
                            Path(md.file_path).unlink(missing_ok=True)
                            cfg_p = Path(str(md.file_path).replace(
                                '.keras', '_config.json').replace(
                                '.h5', '_config.json'))
                            cfg_p.unlink(missing_ok=True)
                        except Exception:
                            pass
                finally:
                    # Limpa NPZ temporário
                    try:
                        fold_npz.unlink(missing_ok=True)
                    except Exception:
                        pass

            # 4. Agrega métricas
            successful_folds = [r for r in per_fold_results if r['status'] == 'success']
            if not successful_folds:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Todos os folds falharam"],
                )

            aggregated: Dict[str, Dict[str, float]] = {}
            for metric_key in ('accuracy', 'precision', 'recall', 'f1_score'):
                values = [r.get(metric_key, 0.0) for r in successful_folds
                          if r.get(metric_key) is not None]
                if values:
                    aggregated[metric_key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }

            # Melhor fold por accuracy
            best_fold = max(
                successful_folds,
                key=lambda r: r.get('accuracy', 0.0),
            )['fold']

            logger.info(
                f"K-fold CV concluído: "
                f"accuracy={aggregated.get('accuracy', {}).get('mean', 0.0):.4f} ± "
                f"{aggregated.get('accuracy', {}).get('std', 0.0):.4f} "
                f"(best fold={best_fold})"
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    'per_fold': per_fold_results,
                    'aggregated': aggregated,
                    'best_fold': best_fold,
                    'n_folds': n_folds,
                    'n_successful': len(successful_folds),
                    'architecture': architecture,
                },
            )

        except ImportError:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=["sklearn é necessário para K-fold CV"],
            )
        except Exception as e:
            logger.error(f"Erro em cross_validate_model: {e}", exc_info=True)
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro inesperado: {str(e)}"],
            )

    def evaluate_model(
        self, model_name: str, test_dataset: str
    ) -> ProcessingResult[Dict[str, float]]:
        # Implementação futura
        return ProcessingResult(
            status=ProcessingStatus.ERROR, errors=["Not implemented"])

    def fine_tune_model(
        self, base_model: str, dataset_name: str, config: Dict[str, Any]
    ) -> ProcessingResult[ModelMetadata]:
        # Implementação futura
        return ProcessingResult(
            status=ProcessingStatus.ERROR, errors=["Not implemented"])

    def get_training_progress(
            self, training_id: str) -> ProcessingResult[Dict[str, Any]]:
        # Implementação futura
        return ProcessingResult(
            status=ProcessingStatus.ERROR, errors=["Not implemented"])

