"""API do benchmark do TCC: protocolo, modelos, runs e execução do artigo.

A leitura é aberta, como ``GET /detection/models``: devolve só métricas
agregadas, receitas e o estado do run, nunca dados por amostra do teste. Iniciar
e parar um run exige ``X-API-Key`` e passa pelas mesmas pré-condições da aba
(``app/domain/services/benchmark/runner.py``); o login do Gradio, que a aba
exige, aqui é a própria chave.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.core.auth.auth_handler import get_api_key
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.security import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/benchmark", tags=["Benchmark"])


class ArticleRunRequest(BaseModel):
    confirm_duration: bool = Field(
        False,
        description="Confirma que o run do artigo leva ~58 h de GPU e ocupa a máquina",
    )


class ResumeRequest(BaseModel):
    run_name: str = Field(..., description="Run iniciado pela aba ou pela API")


def _paths():
    from app.core.bootstrap import OperationalPaths

    return OperationalPaths.resolve()


def _asdict(obj: Any) -> Any:
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def _gpu() -> bool:
    try:
        from app.core.gpu import is_gpu_available

        return bool(is_gpu_available())
    except Exception:  # noqa: BLE001
        return False


@router.get("/protocol", summary="Protocolo, entradas e receitas do artigo")
@limiter.limit("30/minute")
async def protocol(request: Request) -> Dict[str, Any]:
    from app.domain.services.benchmark import article, catalog, runner

    caminhos = _paths()
    entradas = []
    for e in catalog.official_entries():
        entradas.append(
            {
                **_asdict(e),
                "kind": e.kind,
                "expected_gpu_hours": catalog.expected_hours(e),
                "hyperparameters": {
                    k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                    for k, v in catalog.article_hyperparameters(e).items()
                },
            }
        )
    info = article.article_info(caminhos.results / "paper")
    spec = runner.ArticleRunSpec.from_paths(caminhos)
    return {
        "article": _asdict(info),
        "entries": entradas,
        "flags": list(runner.ARTICLE_FLAGS),
        "command_example": runner.build_command(spec),
        "docker_example": runner.docker_command(spec, "bash"),
    }


@router.get("/models", summary="Modelos instalados com calibração e métricas")
@limiter.limit("30/minute")
async def models(request: Request) -> Dict[str, Any]:
    from app.domain.services.benchmark import catalog, results

    caminhos = _paths()
    modelos = catalog.installed_models(
        catalog.discover_model_paths(caminhos.models),
        caminhos.models,
        None,
        results.fingerprint_index(caminhos.results),
    )
    return {"models": [_asdict(m) for m in modelos]}


@router.get("/runs", summary="Runs de benchmark encontrados")
@limiter.limit("30/minute")
async def runs(request: Request) -> Dict[str, Any]:
    from app.domain.services.benchmark.results import discover_runs

    return {"runs": [_asdict(r) for r in discover_runs(_paths().results)]}


@router.get("/runs/{run_id:path}", summary="Métricas agregadas de um run")
@limiter.limit("30/minute")
async def run_detail(request: Request, run_id: str) -> Dict[str, Any]:
    from app.domain.services.benchmark.results import load_run, resolve_run

    try:
        pasta = resolve_run(_paths().results, run_id)
    except ValueError as exc:
        raise ValidationError(str(exc), field="run_id")
    except FileNotFoundError:
        raise NotFoundError("Run", run_id)
    return _asdict(load_run(pasta, run_id))


@router.get("/run/status", summary="Estado do run do artigo em andamento")
@limiter.limit("60/minute")
async def run_status(request: Request) -> Dict[str, Any]:
    from app.domain.services.benchmark.runner import run_status as status

    return status(_paths().results).to_dict()


@router.post(
    "/run",
    summary="Inicia o benchmark do artigo",
    dependencies=[Depends(get_api_key)],
)
@limiter.limit("2/minute")
async def start_run(request: Request, body: ArticleRunRequest) -> Dict[str, Any]:
    from app.domain.services.benchmark import runner

    if not body.confirm_duration:
        raise ValidationError(
            "confirme que o run leva ~58 h de GPU (confirm_duration=true)",
            field="confirm_duration",
        )
    spec = runner.ArticleRunSpec.from_paths(_paths())
    checks = runner.preflight(spec, gpu_available=_gpu, auth_enabled=True)
    if not runner.can_start(checks):
        raise ConflictError(
            "Pré-condições não atendidas: "
            + "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok)
        )
    return {"started": runner.start(spec, checks)}


@router.post(
    "/run/resume",
    summary="Retoma um run do artigo iniciado pela aba ou pela API",
    dependencies=[Depends(get_api_key)],
)
@limiter.limit("2/minute")
async def resume_run(request: Request, body: ResumeRequest) -> Dict[str, Any]:
    from app.domain.services.benchmark import runner

    try:
        spec = runner.ArticleRunSpec.from_paths(_paths(), run_name=body.run_name)
    except ValueError as exc:
        raise ValidationError(str(exc), field="run_name")
    checks = runner.preflight(spec, gpu_available=_gpu, auth_enabled=True, resume=True)
    if not runner.can_start(checks):
        raise ConflictError(
            "Pré-condições não atendidas: "
            + "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok)
        )
    return {"started": runner.start(spec, checks, resume=True)}


@router.post(
    "/run/stop",
    summary="Para o run do artigo em andamento",
    dependencies=[Depends(get_api_key)],
)
@limiter.limit("5/minute")
async def stop_run(request: Request) -> Dict[str, Any]:
    from app.domain.services.benchmark.runner import stop

    trava = stop(_paths().results)
    if trava is None:
        raise NotFoundError("Run em andamento")
    return {"stopped": trava}
