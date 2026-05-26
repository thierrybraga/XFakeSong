"""Hyperparameter Tuning via Optuna (Sprint 4.2).

Provê busca eficiente de hiperparâmetros usando Optuna — biblioteca de
otimização Bayesiana com pruning automático via Hyperband. Permite achar
configurações que tipicamente melhoram +2–5% accuracy em arquiteturas não
ajustadas manualmente.

Dependência opcional (degradação graciosa se não instalada):
    pip install optuna

Uso típico:
    from app.domain.models.training.hyperparameter_tuning import tune_hyperparameters

    search_space = {
        'learning_rate': ('loguniform', 1e-5, 1e-2),
        'batch_size': ('categorical', [16, 32, 64]),
        'dropout_rate': ('uniform', 0.1, 0.5),
        'l2_reg_strength': ('loguniform', 1e-6, 1e-3),
    }

    result = tune_hyperparameters(
        architecture='AASIST',
        dataset_path='data.npz',
        base_config={'epochs': 20, 'optimizer': 'adamw'},
        search_space=search_space,
        n_trials=30,
        metric='val_accuracy',
        direction='maximize',
    )

    print(f"Best params: {result['best_params']}")
    print(f"Best score: {result['best_score']}")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def is_optuna_available() -> bool:
    """Verifica se optuna está instalado."""
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def _suggest_from_spec(trial, name: str, spec):
    """Converte spec ('uniform', a, b) ou ('loguniform', a, b) ou
    ('categorical', [...]) ou ('int', a, b) em trial.suggest_*.
    """
    kind = spec[0]
    if kind == 'uniform':
        return trial.suggest_float(name, spec[1], spec[2], log=False)
    elif kind == 'loguniform':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif kind == 'int':
        return trial.suggest_int(name, spec[1], spec[2], log=spec[3] if len(spec) > 3 else False)
    elif kind == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Spec inválido: {spec}. Use 'uniform'/'loguniform'/'int'/'categorical'")


def tune_hyperparameters(
    architecture: str,
    dataset_path: str,
    base_config: Dict[str, Any],
    search_space: Dict[str, tuple],
    n_trials: int = 30,
    metric: str = 'val_accuracy',
    direction: str = 'maximize',
    timeout_seconds: Optional[int] = None,
    use_pruning: bool = True,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    custom_train_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Otimização de hiperparâmetros via Optuna.

    Args:
        architecture: nome da arquitetura (AASIST, Conformer, etc.)
        dataset_path: .npz com X_train/y_train (e opcional X_val/y_val)
        base_config: config base (épocas reduzidas para velocidade — ex: 20)
        search_space: dict de {param_name: spec} onde spec é:
            ('uniform', low, high) | ('loguniform', low, high) |
            ('int', low, high[, log]) | ('categorical', [opt1, opt2, ...])
        n_trials: número de trials Optuna (default 30; usar 50-100 em produção)
        metric: 'val_accuracy', 'accuracy', 'f1_score', 'val_loss', etc.
        direction: 'maximize' (default) ou 'minimize'
        timeout_seconds: timeout total opcional
        use_pruning: se True, ativa Hyperband pruner (poda trials ruins cedo)
        study_name: nome para persistir study (com storage)
        storage: SQLite URL para persistir study (ex: 'sqlite:///optuna.db')
        custom_train_fn: função custom (trial, params) -> score. Default usa
            TrainingService.train_model.

    Returns:
        Dict com:
            - status: 'success' | 'error' | 'unavailable'
            - best_params: dict de hiperparâmetros ótimos
            - best_score: melhor valor da métrica
            - n_trials: trials completados
            - study: optuna.Study (se study_name fornecido)
            - errors: list (se status != success)
    """
    if not is_optuna_available():
        msg = ("optuna não instalado. Instale com: pip install optuna")
        logger.warning(msg)
        return {
            'status': 'unavailable',
            'errors': [msg],
            'best_params': {},
            'best_score': None,
            'n_trials': 0,
        }

    try:
        import optuna
        from optuna.pruners import HyperbandPruner, MedianPruner
        from optuna.samplers import TPESampler

        # Configura sampler e pruner
        sampler = TPESampler(seed=42)
        pruner = HyperbandPruner() if use_pruning else MedianPruner()

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(storage),
        )

        # Default training function: usa TrainingService
        if custom_train_fn is None:
            custom_train_fn = _default_train_fn

        def objective(trial: 'optuna.Trial') -> float:
            # Sample hyperparameters
            params = {
                name: _suggest_from_spec(trial, name, spec)
                for name, spec in search_space.items()
            }
            # Merge com base_config; params vão em 'parameters' do TrainingConfig
            fold_config = dict(base_config)
            arch_params = fold_config.get('parameters', {}).copy()

            # Distribui params entre top-level e parameters
            for k, v in params.items():
                if k in (
                    'learning_rate', 'batch_size', 'epochs', 'optimizer',
                    'loss_function', 'early_stopping_patience',
                    'reduce_lr_patience', 'validation_split',
                ):
                    fold_config[k] = v
                else:
                    arch_params[k] = v
            fold_config['parameters'] = arch_params
            # Nome único para evitar colisão
            fold_config['model_name'] = (
                f"{architecture}_trial{trial.number}"
            )

            try:
                score = custom_train_fn(
                    architecture=architecture,
                    dataset_path=dataset_path,
                    config=fold_config,
                    metric=metric,
                    trial=trial,
                )
                return float(score)
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} falhou: {e}")
                # Trial falho retorna pior valor possível
                return float('-inf') if direction == 'maximize' else float('inf')

        # Run optimization
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout_seconds,
            show_progress_bar=False,
        )

        best_params = dict(study.best_params)
        best_score = float(study.best_value)
        n_completed = len(study.trials)

        logger.info(
            f"Optuna tuning concluído: best {metric}={best_score:.4f} "
            f"(n_trials={n_completed})"
        )
        logger.info(f"Best params: {best_params}")

        return {
            'status': 'success',
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_completed,
            'study': study,
            'errors': [],
        }

    except Exception as e:
        logger.error(f"Erro em tune_hyperparameters: {e}", exc_info=True)
        return {
            'status': 'error',
            'errors': [str(e)],
            'best_params': {},
            'best_score': None,
            'n_trials': 0,
        }


def _default_train_fn(
    architecture: str,
    dataset_path: str,
    config: Dict[str, Any],
    metric: str,
    trial: Any,
) -> float:
    """Função default de treinamento usada pelo Optuna objective.

    Usa TrainingService.train_model e extrai a métrica desejada do
    ModelMetadata retornado. Limpa o arquivo de modelo após cada trial
    para evitar acumular arquivos em disco.
    """
    from pathlib import Path

    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.training_service import TrainingService

    service = TrainingService(models_dir="app/models/optuna_trials")
    result = service.train_model(
        architecture=architecture,
        dataset_path=dataset_path,
        config=config,
    )

    if result.status != ProcessingStatus.SUCCESS:
        raise RuntimeError(f"Train failed: {result.errors}")

    md = result.data
    # Cleanup modelo do trial (Optuna pode rodar 30+ trials)
    try:
        Path(md.file_path).unlink(missing_ok=True)
        cfg_p = Path(str(md.file_path).replace('.keras', '_config.json'))
        cfg_p.unlink(missing_ok=True)
    except Exception:
        pass

    # Extrai métrica
    if metric in ('accuracy', 'val_accuracy'):
        return float(md.accuracy)
    elif metric in ('precision', 'val_precision'):
        return float(md.precision)
    elif metric in ('recall', 'val_recall'):
        return float(md.recall)
    elif metric in ('f1_score', 'val_f1_score', 'f1'):
        return float(md.f1_score)
    elif metric == 'val_loss':
        # Lower is better — Optuna direction='minimize'
        return float(md.metrics.get('val_loss', md.metrics.get('loss', 1.0)))
    else:
        # Fallback: tenta no dict de métricas
        return float(md.metrics.get(metric, md.accuracy))


def suggest_search_space(architecture: str) -> Dict[str, tuple]:
    """Sugere search space razoável por arquitetura.

    Returns:
        Dict {param_name: spec} pronto para passar a `tune_hyperparameters`.
    """
    base = {
        'learning_rate': ('loguniform', 1e-5, 1e-2),
        'batch_size': ('categorical', [16, 32, 64]),
        'dropout_rate': ('uniform', 0.1, 0.5),
    }

    arch_lower = architecture.lower()
    if 'aasist' in arch_lower or 'rawgat' in arch_lower:
        base['l2_reg_strength'] = ('loguniform', 1e-6, 1e-3)
        base['hidden_dim'] = ('categorical', [256, 512])
    elif 'conformer' in arch_lower or 'transformer' in arch_lower:
        base['learning_rate'] = ('loguniform', 1e-5, 5e-3)  # menor para Trans.
        base['dropout_rate'] = ('uniform', 0.1, 0.3)
    elif 'efficientnet' in arch_lower:
        base['lstm_units'] = ('categorical', [128, 256, 512])
    elif 'rawnet' in arch_lower:
        base['gru_units'] = ('categorical', [128, 256, 512])

    return base
