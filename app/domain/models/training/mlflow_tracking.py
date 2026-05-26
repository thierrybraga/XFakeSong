"""MLflow Tracking (Sprint 4.3).

Provê rastreamento de experimentos via MLflow — biblioteca de MLOps para
versionamento de modelos, parâmetros e métricas. Permite comparação visual
de runs no MLflow UI (`mlflow ui`).

Dependência opcional (degradação graciosa se não instalada):
    pip install mlflow

Uso típico via context manager:
    from app.domain.models.training.mlflow_tracking import MLflowTracker

    with MLflowTracker(experiment_name='aasist_tuning') as tracker:
        tracker.log_config(config_dict)
        result = trainer.train(model, train_data, validation_data)
        tracker.log_training_result(result)
        tracker.log_model_artifacts(model_path, config_path)

Visualizar:
    mlflow ui  # abre em http://localhost:5000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Verifica se mlflow está instalado."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


class MLflowTracker:
    """Wrapper de MLflow com degradação graciosa.

    Quando mlflow não está instalado, todas as operações são no-ops
    (loggam debug mas não falham). Permite usar como `with` sem precisar
    checar disponibilidade no código de chamada.

    Args:
        experiment_name: nome do experimento (cria se não existir)
        run_name: nome do run (default: auto-gerado por MLflow)
        tracking_uri: URI do servidor MLflow (default: ./mlruns local)
        tags: dict de tags adicionais (ex: {'architecture': 'AASIST'})
        enabled: se False, faz no-op mesmo com mlflow instalado (útil em CI)
    """

    def __init__(
        self,
        experiment_name: str = "xfakesong",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        enabled: bool = True,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.enabled = enabled and is_mlflow_available()
        self._run = None
        self._mlflow = None

    def __enter__(self):
        if not self.enabled:
            logger.debug(
                "MLflowTracker desabilitado (mlflow não instalado ou enabled=False)"
            )
            return self
        try:
            import mlflow
            self._mlflow = mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)

            self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
            logger.info(
                f"MLflow run iniciado: experiment='{self.experiment_name}', "
                f"run_id={self._run.info.run_id}"
            )
        except Exception as e:
            logger.warning(f"Falha ao iniciar MLflow: {e}")
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or self._run is None:
            return False
        try:
            if exc_type is not None:
                self._mlflow.set_tag('status', 'failed')
                self._mlflow.set_tag('error', str(exc_val))
            else:
                self._mlflow.set_tag('status', 'success')
            self._mlflow.end_run()
        except Exception as e:
            logger.warning(f"Falha ao finalizar MLflow run: {e}")
        return False  # não suprime exceptions

    def log_config(self, config: Dict[str, Any]) -> None:
        """Loga config de treinamento como parâmetros.

        Flatten dicts aninhados (ex: 'parameters.dropout_rate').
        """
        if not self.enabled:
            return
        try:
            flat = self._flatten_dict(config)
            for k, v in flat.items():
                # MLflow tem limite de 250 chars em values
                v_str = str(v)
                if len(v_str) > 240:
                    v_str = v_str[:237] + '...'
                self._mlflow.log_param(k, v_str)
        except Exception as e:
            logger.warning(f"Falha ao logar config: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Loga métricas (accuracy, loss, EER, etc.)."""
        if not self.enabled:
            return
        try:
            clean_metrics = {}
            for k, v in metrics.items():
                # MLflow só aceita números
                if isinstance(v, (int, float)) and v == v:  # filtra NaN
                    clean_metrics[k] = float(v)
            if clean_metrics:
                self._mlflow.log_metrics(clean_metrics, step=step)
        except Exception as e:
            logger.warning(f"Falha ao logar métricas: {e}")

    def log_training_result(self, result_data: Dict[str, Any]) -> None:
        """Loga resultado de ModelTrainer.train (dict com history + final_metrics)."""
        if not self.enabled:
            return
        try:
            # Métricas finais (Sprint 1+2)
            final = result_data.get('final_metrics', {})
            if final:
                # Filtra valores escalares (deixa de fora arrays como confusion matrix)
                scalar_metrics = {
                    k: v for k, v in final.items()
                    if isinstance(v, (int, float)) and v == v
                }
                self.log_metrics(scalar_metrics)

            # History (por epoch) — loga last epoch values
            history = result_data.get('history', {})
            for metric_name, values in history.items():
                if isinstance(values, list) and values:
                    self._mlflow.log_metric(f"final_{metric_name}", float(values[-1]))
                    # Loga curva completa
                    for step, val in enumerate(values):
                        if isinstance(val, (int, float)) and val == val:
                            self._mlflow.log_metric(metric_name, float(val), step=step)
        except Exception as e:
            logger.warning(f"Falha ao logar resultado de treinamento: {e}")

    def log_model_artifacts(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        extra_paths: Optional[list] = None,
    ) -> None:
        """Loga modelo e arquivos relacionados como artifacts."""
        if not self.enabled:
            return
        try:
            model_path = Path(model_path)
            if model_path.exists():
                self._mlflow.log_artifact(str(model_path), artifact_path="model")

            if config_path is not None:
                config_path = Path(config_path)
                if config_path.exists():
                    self._mlflow.log_artifact(str(config_path), artifact_path="model")

            for p in (extra_paths or []):
                p = Path(p)
                if p.exists():
                    self._mlflow.log_artifact(str(p), artifact_path="model")
        except Exception as e:
            logger.warning(f"Falha ao logar artifacts: {e}")

    def log_calibration_params(
        self,
        temperature: float = 1.0,
        ood_threshold: Optional[float] = None,
        eer_threshold: Optional[float] = None,
        eer_value: Optional[float] = None,
    ) -> None:
        """Loga parâmetros de calibração (Sprint 1.4 + 2.5 + 4.5)."""
        if not self.enabled:
            return
        try:
            self._mlflow.log_param('temperature_calibrated', float(temperature))
            if ood_threshold is not None:
                self._mlflow.log_param('ood_threshold', float(ood_threshold))
            if eer_threshold is not None:
                self._mlflow.log_param('eer_threshold', float(eer_threshold))
            if eer_value is not None:
                self._mlflow.log_metric('eer', float(eer_value))
        except Exception as e:
            logger.warning(f"Falha ao logar params de calibração: {e}")

    def log_tag(self, key: str, value: str) -> None:
        """Loga uma tag arbitrária."""
        if not self.enabled:
            return
        try:
            self._mlflow.set_tag(key, str(value))
        except Exception as e:
            logger.warning(f"Falha ao logar tag {key}: {e}")

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.',
    ) -> Dict[str, Any]:
        """Achata dict aninhado: {'a': {'b': 1}} → {'a.b': 1}"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


def wrap_training_with_mlflow(
    train_callable,
    experiment_name: str = "xfakesong",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Helper para envolver função de treinamento com MLflow tracking.

    Uso:
        result = wrap_training_with_mlflow(
            lambda: service.train_model(arch, ds, cfg),
            experiment_name='aasist_runs',
            tags={'architecture': 'AASIST'},
        )
    """
    with MLflowTracker(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags or {},
    ) as tracker:
        result = train_callable()
        # Tentativa de extrair info do ProcessingResult
        try:
            from app.core.interfaces.base import ProcessingStatus
            if hasattr(result, 'status') and result.status == ProcessingStatus.SUCCESS:
                md = result.data
                if hasattr(md, 'accuracy'):
                    tracker.log_metrics({
                        'accuracy': float(md.accuracy),
                        'precision': float(md.precision),
                        'recall': float(md.recall),
                        'f1_score': float(md.f1_score),
                    })
                if hasattr(md, 'file_path'):
                    tracker.log_model_artifacts(md.file_path)
        except Exception as e:
            logger.debug(f"Não conseguiu extrair metadata para MLflow: {e}")
        return result
