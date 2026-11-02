"""Ciclo de vida dos jobs de treino e de K-fold disparados pela API.

O router de treino só valida a entrada e delega: criar o job, executá-lo em
background, atualizar progresso/estado e responder consultas de status são
responsabilidades daqui. O estado é persistido em ``training_jobs`` (via
``TrainingJobRepository``) e espelhado num cache com TTL para que o polling de
status não bata no banco a cada requisição.

O ``TrainingService`` é recebido por chamada, não guardado: a API o injeta por
requisição (``Depends(get_training_service)``) e é essa instância que deve
treinar.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.core.contracts.base import ProcessingStatus
from app.core.db.models.training_job import TrainingJob
from app.core.db.repositories.training_jobs import TrainingJobRepository
from app.core.exceptions import ValidationError

if TYPE_CHECKING:
    from app.domain.services.training_service import TrainingService

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = ("completed", "failed")


class TTLCache:
    """Cache com time-to-live para evitar divergência com o banco de dados.

    Entradas expiram após `ttl` segundos, forçando releitura do DB.
    Jobs em estado terminal (completed/failed) ficam no cache por mais tempo
    pois não mudam mais.
    """

    def __init__(self, ttl: int = 30, terminal_ttl: int = 300):
        self._data: Dict[str, Dict] = {}
        self._timestamps: Dict[str, float] = {}
        self.ttl = ttl
        self.terminal_ttl = terminal_ttl

    def get(self, key: str) -> Optional[Dict]:
        if key not in self._data:
            return None
        age = time.monotonic() - self._timestamps[key]
        entry = self._data[key]
        # Jobs terminais têm TTL mais longo
        max_ttl = (
            self.terminal_ttl if entry.get("status") in _TERMINAL_STATUSES else self.ttl
        )
        if age > max_ttl:
            del self._data[key]
            del self._timestamps[key]
            return None
        return entry

    def set(self, key: str, value: Dict):
        self._data[key] = value
        self._timestamps[key] = time.monotonic()


def _status_entry(job: TrainingJob) -> Dict[str, Any]:
    return {
        "status": job.status,
        "message": job.message,
        "progress": job.progress,
        "metrics": job.metrics or {},
    }


class TrainingJobService:
    """Cria, executa e acompanha jobs de treino e validação cruzada."""

    def __init__(
        self,
        repository: TrainingJobRepository | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        self._repository = repository or TrainingJobRepository()
        # TTL de 30s para jobs ativos, 5min para terminais
        self._cache = cache or TTLCache(ttl=30, terminal_ttl=300)

    # ── Criação ─────────────────────────────────────────────────────────

    def _create_job(self, message: str, **fields: Any) -> str:
        job_id = str(uuid.uuid4())
        job = self._repository.create(
            job_id=job_id, status="pending", message=message, progress=0, **fields
        )
        self._cache.set(job_id, _status_entry(job))
        return job_id

    def create_training_job(
        self,
        *,
        architecture: str,
        model_name: str,
        dataset_path: str,
        parameters: Dict[str, Any] | None,
    ) -> str:
        """Registra um job de treino pendente e devolve o ``job_id``.

        Falhas de persistência sobem para o chamador.
        """
        return self._create_job(
            "Job created",
            architecture=architecture,
            model_name=model_name,
            dataset_path=dataset_path,
            parameters=parameters,
        )

    def create_cross_validation_job(
        self,
        *,
        architecture: str,
        dataset_path: str,
        config: Dict[str, Any],
        n_folds: int,
    ) -> str:
        """Registra um job de K-fold pendente e devolve o ``job_id``."""
        return self._create_job(
            f"CV job created (k={n_folds})",
            architecture=architecture,
            model_name=f"cv_{architecture}",
            dataset_path=dataset_path,
            parameters={**config, "n_folds": n_folds},
        )

    # ── Consulta ────────────────────────────────────────────────────────

    def get_status(self, job_id: str) -> Dict[str, Any] | None:
        """Estado do job: primeiro o cache, depois o banco; ``None`` se não existe.

        Erros do banco sobem para o chamador.
        """
        cached = self._cache.get(job_id)
        if cached:
            return cached
        job = self._repository.get(job_id)
        if not job:
            return None
        return _status_entry(job)

    def get_cross_validation_result(self, job_id: str) -> Dict[str, Any]:
        """Resultado agregado de um K-fold concluído.

        Raises:
            ValidationError: job inexistente ou ainda não concluído.
        """
        job = self._repository.get(job_id)
        if not job:
            raise ValidationError(f"Job '{job_id}' não encontrado")
        if job.status != "completed":
            raise ValidationError(
                f"Job ainda não concluído (status={job.status}). "
                f"Use /training/status/{job_id} para acompanhar."
            )

        metrics = job.metrics or {}
        # Reconstrói aggregated a partir das métricas planas que salvamos
        aggregated: Dict[str, Dict[str, float]] = {}
        for full_key, value in metrics.items():
            if "_" in full_key:
                base, stat = full_key.rsplit("_", 1)
                if stat in ("mean", "std", "min", "max"):
                    aggregated.setdefault(base, {})[stat] = float(value)

        return {
            "architecture": job.architecture or "unknown",
            "n_folds": int(metrics.get("n_folds", 0)),
            "n_successful": int(metrics.get("per_fold_count", 0)),
            "best_fold": int(metrics.get("best_fold", -1)),
            "aggregated": aggregated,
            "per_fold": [],  # detalhes per_fold ficam apenas no log do task
        }

    # ── Execução (background) ───────────────────────────────────────────

    def _update_job(self, job_id: str, **fields: Any) -> None:
        """Atualiza job de treinamento no banco e no cache.

        O cache só muda depois do commit bem-sucedido; falhas são registradas
        e não interrompem a tarefa em background.
        """
        try:
            job = self._repository.update(job_id, **fields)
            if not job:
                return
            self._cache.set(job_id, _status_entry(job))
        except Exception as e:
            logger.error(f"Falha ao atualizar job {job_id}: {e}")

    def run_training(
        self,
        job_id: str,
        service: TrainingService,
        *,
        architecture: str,
        dataset_path: str,
        model_name: str,
        parameters: Dict[str, Any] | None,
        epochs: int | None,
        batch_size: int | None,
    ) -> None:
        """
        Executa tarefa de treinamento em background usando o serviço real.
        Atualiza progresso e status no banco para persistência.
        """
        try:
            self._update_job(
                job_id,
                status="running",
                message="Training started...",
                progress=5
            )

            config = {
                "model_name": model_name,
                "parameters": parameters,
                "epochs": epochs,
                "batch_size": batch_size
            }

            result = service.train_model(
                architecture=architecture,
                dataset_path=dataset_path,
                config=config
            )

            if result.status == ProcessingStatus.SUCCESS:
                if hasattr(result.data, "metrics"):
                    metrics = getattr(result.data, "metrics", {})
                else:
                    metrics = {}
                self._update_job(
                    job_id,
                    status="completed",
                    message="Training completed successfully.",
                    progress=100,
                    metrics=metrics
                )
            else:
                self._update_job(
                    job_id,
                    status="failed",
                    message=f"Training failed: {'; '.join(result.errors)}",
                    progress=100
                )
        except Exception as e:
            logger.error(f"Error in background training task: {e}")
            self._update_job(
                job_id,
                status="failed",
                message=f"Internal error: {str(e)}",
                progress=100
            )

    def run_cross_validation(
        self,
        job_id: str,
        service: TrainingService,
        *,
        architecture: str,
        dataset_path: str,
        config: Dict[str, Any],
        n_folds: int,
        save_fold_models: bool,
    ) -> None:
        """Background task para K-fold cross-validation."""
        try:
            self._update_job(
                job_id,
                status="running",
                message=f"K-fold CV started (k={n_folds})...",
                progress=5,
            )

            result = service.cross_validate_model(
                architecture=architecture,
                dataset_path=dataset_path,
                config=config,
                n_folds=n_folds,
                save_fold_models=save_fold_models,
            )

            if result.status == ProcessingStatus.SUCCESS:
                data = result.data or {}
                agg = data.get("aggregated", {})
                metrics = {
                    f"{k}_{stat}": v
                    for k, sub in agg.items() for stat, v in sub.items()
                }
                self._update_job(
                    job_id,
                    status="completed",
                    message=(
                        f"CV completo: {data.get('n_successful', 0)}/"
                        f"{n_folds} folds bem-sucedidos. "
                        f"Best fold={data.get('best_fold', '?')}"
                    ),
                    progress=100,
                    metrics={
                        **metrics,
                        "n_folds": data.get("n_folds", n_folds),
                        "best_fold": data.get("best_fold", -1),
                        "per_fold_count": len(data.get("per_fold", [])),
                    },
                )
            else:
                self._update_job(
                    job_id,
                    status="failed",
                    message=f"CV failed: {'; '.join(result.errors or ['unknown'])}",
                    progress=100,
                )
        except Exception as e:
            logger.error(f"Erro em CV task {job_id}: {e}", exc_info=True)
            self._update_job(
                job_id,
                status="failed",
                message=f"Internal error: {e}",
                progress=100,
            )
