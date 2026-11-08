"""Leitura dos runs do benchmark, só com métricas agregadas.

Um run é uma pasta com ``run_summary.json`` (gravado pelo orquestrador ao fim de
cada modelo) e uma subpasta por modelo com ``results.json``. O ``results.json``
também carrega dados de TESTE por amostra (``scores_*``, ``y_test``, ids de
cluster e de locutor), e a pasta tem ``predictions_*.csv``: nada disso sai
daqui. Os leitores devolvem métricas agregadas, curvas de validação e caminhos
de figuras.

Os caminhos gravados dentro dos JSON são do container (``/app/data/...``) ou de
outra máquina; os arquivos são sempre resolvidos relativos à pasta do run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .catalog import compact, official_entries

logger = logging.getLogger(__name__)

#: Pastas de ``data/results`` que não são runs de benchmark.
NOT_RUNS = {"paper", "recalibration", "xai", "reporting", "consolidated", "families"}
#: Figuras por modelo que o relatório do runner gera (``benchmarks/report.py``).
FIGURE_NAMES = (
    "roc",
    "det",
    "robustez",
    "convergencia",
    "eficiencia",
    "score_distributions",
    "confusion_matrices",
)
SNR_ORDER = ("30", "20", "10", "5")


@dataclass
class RunInfo:
    run_id: str
    path: str
    label: str
    kind: str
    status: Optional[str]
    n_models: int
    modified: str


@dataclass
class ModelResult:
    architecture: str
    display_name: str
    folder: str
    status: str
    superseded: bool = False
    clean: Dict[str, Any] = field(default_factory=dict)
    robustness: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    codec: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    worst_speaker_accuracy: Optional[float] = None
    efficiency: Dict[str, Any] = field(default_factory=dict)
    stability: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    figures: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunView:
    info: RunInfo
    models: List[ModelResult]
    summary_models: List[Dict[str, Any]]
    protocol: Dict[str, Any]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _classify(rel: str) -> tuple[str, str]:
    nome = rel.replace("\\", "/")
    if nome == "benchmark":
        return "vigente", "Run vigente (protocolo corrigido)"
    if nome.endswith("clean_benchmark_15k"):
        return "artigo", "Run dos números do artigo (arquivado, protocolo antigo)"
    if nome.split("/")[-1].startswith("tcc_reproducao_"):
        return "reproducao", f"Reprodução pela aba ({nome.split('/')[-1]})"
    if nome.split("/")[-1].startswith("production"):
        return "producao", f"Geração de produção ({nome})"
    return "outro", nome


def _is_run(pasta: Path) -> bool:
    if (pasta / "run_summary.json").is_file():
        return True
    return any(p.is_file() for p in pasta.glob("*/results.json"))


def discover_runs(results_root: Path, max_depth: int = 3) -> List[RunInfo]:
    """Runs sob ``results_root`` (até ``max_depth`` níveis), o vigente primeiro."""
    raiz = Path(results_root)
    if not raiz.is_dir():
        return []
    achados: List[RunInfo] = []

    def visitar(pasta: Path, profundidade: int) -> None:
        for filho in sorted(p for p in pasta.iterdir() if p.is_dir()):
            if filho.name.startswith(("_", ".")) or filho.name in NOT_RUNS:
                continue
            if _is_run(filho):
                rel = filho.relative_to(raiz).as_posix()
                tipo, rotulo = _classify(rel)
                resumo = _read_json(filho / "run_summary.json") or {}
                achados.append(
                    RunInfo(
                        run_id=rel,
                        path=str(filho),
                        label=rotulo,
                        kind=tipo,
                        status=(
                            resumo.get("status") if isinstance(resumo, dict) else None
                        ),
                        n_models=sum(1 for _ in filho.glob("*/results.json")),
                        modified=datetime.fromtimestamp(
                            filho.stat().st_mtime
                        ).isoformat(timespec="minutes"),
                    )
                )
            elif profundidade < max_depth:
                visitar(filho, profundidade + 1)

    visitar(raiz, 1)
    prioridade = {"vigente": 0, "reproducao": 1, "artigo": 2, "producao": 3, "outro": 4}
    return sorted(achados, key=lambda r: (prioridade.get(r.kind, 9), r.run_id))


def resolve_run(results_root: Path, run_id: str) -> Path:
    """Pasta do run, recusando qualquer coisa fora de ``results_root``."""
    raiz = Path(results_root).resolve()
    alvo = (raiz / run_id).resolve()
    if raiz != alvo and raiz not in alvo.parents:
        raise ValueError("run fora da pasta de resultados")
    if not alvo.is_dir():
        raise FileNotFoundError(run_id)
    return alvo


_CLEAN_KEYS = (
    "eer",
    "eer_ci95_low",
    "eer_ci95_high",
    "auc_roc",
    "accuracy",
    "accuracy_at_calibrated_threshold",
    "min_tdcf",
    "ece",
    "f1",
    "eer_threshold",
    "calibrated_threshold",
    "n",
)


def _aggregate(bloco: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: bloco.get(k) for k in _CLEAN_KEYS if bloco.get(k) is not None}


def _figures(model_dir: Path) -> Dict[str, str]:
    saida: Dict[str, str] = {}
    for nome in FIGURE_NAMES:
        alvo = model_dir / "figures" / f"{nome}.png"
        if alvo.is_file():
            saida[nome] = str(alvo)
    return saida


def model_result(model_dir: Path, superseded: bool = False) -> Optional[ModelResult]:
    """O resultado agregado de um modelo, ou ``None`` se não houver ``results.json``."""
    dados = _read_json(Path(model_dir) / "results.json")
    if not isinstance(dados, dict):
        return None
    arquiteturas = dados.get("architectures") or {}
    if not isinstance(arquiteturas, dict) or not arquiteturas:
        return None
    nome, bloco = next(iter(arquiteturas.items()))
    if not isinstance(bloco, dict):
        return None
    entrada = next((e for e in official_entries() if e.matches(nome)), None)
    robustez = {}
    for snr, medida in (bloco.get("robustness") or {}).items():
        if isinstance(medida, dict):
            robustez[str(snr)] = {
                **_aggregate(medida),
                "noise_condition": medida.get("noise_condition"),
            }
    codec = {
        str(k): _aggregate(v)
        for k, v in (bloco.get("codec_robustness") or {}).items()
        if isinstance(v, dict) and v.get("status") != "error"
    }
    agrupado = (bloco.get("grouped_clean") or {}).get("speaker") or {}
    eficiencia = bloco.get("efficiency") or {}
    estabilidade = bloco.get("training_stability") or {}
    treino = bloco.get("training_config") or {}
    historico = bloco.get("history") or {}
    return ModelResult(
        architecture=str(nome),
        display_name=entrada.display_name if entrada else str(nome),
        folder=Path(model_dir).name,
        status=str(bloco.get("status") or "ok"),
        superseded=superseded,
        clean=_aggregate(bloco.get("clean") or {}),
        robustness=dict(sorted(robustez.items(), key=lambda kv: -float(kv[0]))),
        codec=codec,
        worst_speaker_accuracy=(
            agrupado.get("worst_group_accuracy") if isinstance(agrupado, dict) else None
        ),
        efficiency={
            k: eficiencia.get(k)
            for k in ("params", "size_mb", "latency_ms")
            if isinstance(eficiencia, dict) and eficiencia.get(k) is not None
        },
        stability={
            k: estabilidade.get(k)
            for k in ("status", "best_epoch", "warnings")
            if isinstance(estabilidade, dict) and estabilidade.get(k) is not None
        },
        history={
            k: [float(x) for x in v]
            for k, v in historico.items()
            if k in ("val_eer", "val_loss", "loss", "val_accuracy")
            and isinstance(v, list)
            and all(isinstance(x, (int, float)) for x in v)
        },
        training={
            k: treino.get(k)
            for k in (
                "checkpoint_monitor",
                "calibrated_temperature",
                "batch_size",
                "learning_rate",
                "epochs",
                "optimizer",
            )
            if isinstance(treino, dict) and treino.get(k) is not None
        },
        figures=_figures(Path(model_dir)),
    )


def load_run(run_dir: Path, run_id: str = "") -> RunView:
    """Métricas agregadas de todos os modelos do run.

    Uma entrada do artigo que falte nas pastas principais é procurada em
    ``_superseded/`` (é onde está o RawNet2 válido do run vigente) e marcada.
    """
    pasta = Path(run_dir)
    modelos: List[ModelResult] = []
    for sub in sorted(p for p in pasta.iterdir() if p.is_dir()):
        if sub.name.startswith(("_", ".")):
            continue
        resultado = model_result(sub)
        if resultado is not None:
            modelos.append(resultado)
    presentes = {compact(m.architecture) for m in modelos}
    substituidos = pasta / "_superseded"
    if substituidos.is_dir():
        for entrada in official_entries():
            if compact(entrada.benchmark_name) in presentes:
                continue
            candidatos = sorted(
                (p for p in substituidos.iterdir() if (p / "results.json").is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for candidato in candidatos:
                resultado = model_result(candidato, superseded=True)
                if resultado and entrada.matches(resultado.architecture):
                    if resultado.status == "ok":
                        modelos.append(resultado)
                        presentes.add(compact(entrada.benchmark_name))
                        break
    ordem = {e.benchmark_name: i for i, e in enumerate(official_entries())}
    modelos.sort(
        key=lambda m: next(
            (
                i
                for nome, i in ordem.items()
                if compact(nome) == compact(m.architecture)
            ),
            99,
        )
    )
    resumo = _read_json(pasta / "run_summary.json") or {}
    resumo_modelos = [
        {
            k: item.get(k)
            for k in ("model", "status", "elapsed_s", "timeout_min", "error")
            if isinstance(item, dict)
        }
        for item in (resumo.get("models") or [] if isinstance(resumo, dict) else [])
    ]
    protocolo = _read_json(pasta / "benchmark_protocol.json") or {}
    tipo, rotulo = _classify(run_id or pasta.name)
    return RunView(
        info=RunInfo(
            run_id=run_id or pasta.name,
            path=str(pasta),
            label=rotulo,
            kind=tipo,
            status=resumo.get("status") if isinstance(resumo, dict) else None,
            n_models=len(modelos),
            modified=datetime.fromtimestamp(pasta.stat().st_mtime).isoformat(
                timespec="minutes"
            ),
        ),
        models=modelos,
        summary_models=resumo_modelos,
        protocol={
            k: protocolo.get(k)
            for k in (
                "protocol_version",
                "standardized_controls",
                "academic_protocol",
                "experiment_scope",
            )
            if isinstance(protocolo, dict) and protocolo.get(k) is not None
        },
    )


def fingerprint_index(results_root: Path) -> Dict[str, Dict[str, Any]]:
    """Métricas de teste por sha256 do artefato que o run gravou.

    É o elo confiável entre um modelo instalado e o run que o produziu: o
    ``registry.json`` da promoção pode descrever outro artefato (o de
    2026-08-14 é de antes do retreino que gerou os modelos instalados).
    """
    indice: Dict[str, Dict[str, Any]] = {}
    for run in discover_runs(results_root):
        pasta = Path(run.path)
        candidatos = list(pasta.glob("*/results.json")) + list(
            pasta.glob("_superseded/*/results.json")
        )
        for arquivo in candidatos:
            dados = _read_json(arquivo)
            if not isinstance(dados, dict):
                continue
            for nome, bloco in (dados.get("architectures") or {}).items():
                if not isinstance(bloco, dict):
                    continue
                digital = (bloco.get("model_artifact_fingerprint") or {}).get("sha256")
                if digital and digital not in indice:
                    eficiencia = bloco.get("efficiency") or {}
                    indice[str(digital)] = {
                        "run_id": run.run_id,
                        "architecture": str(nome),
                        "clean": _aggregate(bloco.get("clean") or {}),
                        "efficiency": {
                            k: eficiencia.get(k)
                            for k in ("params", "size_mb", "latency_ms")
                            if isinstance(eficiencia, dict)
                            and eficiencia.get(k) is not None
                        },
                    }
    return indice


def article_summary(paper_dir: Path) -> List[Dict[str, Any]]:
    """Linhas da tabela do artigo (``consolidated/benchmark_summary.json``)."""
    dados = _read_json(Path(paper_dir) / "consolidated" / "benchmark_summary.json")
    if not isinstance(dados, list):
        return []
    saida = []
    for linha in dados:
        if not isinstance(linha, dict):
            continue
        robustez = linha.get("robustness") or {}
        saida.append(
            {
                "model": linha.get("model"),
                "eer": linha.get("eer"),
                "eer_ci95_low": linha.get("eer_ci95_low"),
                "eer_ci95_high": linha.get("eer_ci95_high"),
                "auc": linha.get("auc"),
                "accuracy": linha.get("accuracy"),
                "min_tdcf": linha.get("min_tdcf"),
                "eer_10db": (robustez.get("10") or {}).get("eer"),
                "eer_5db": (robustez.get("5") or {}).get("eer"),
                "worst_speaker_accuracy": linha.get("worst_speaker_accuracy"),
                "latency_ms": linha.get("latency"),
                "source": str(linha.get("path") or "").replace("\\", "/"),
            }
        )
    return saida


def article_significance(paper_dir: Path) -> List[Dict[str, Any]]:
    """Pares de modelos com McNemar e bootstrap pareado (já corrigidos por Holm)."""
    dados = _read_json(Path(paper_dir) / "consolidated" / "benchmark_significance.json")
    if not isinstance(dados, dict):
        return []
    saida = []
    for par in dados.get("pairs") or []:
        if not isinstance(par, dict):
            continue
        mcnemar = par.get("mcnemar") or {}
        boot = par.get("paired_bootstrap") or {}
        saida.append(
            {
                "model_a": par.get("model_a"),
                "model_b": par.get("model_b"),
                "mcnemar_p_holm": mcnemar.get("p_value_holm"),
                "eer_difference": boot.get("observed_difference"),
                "eer_difference_ci95": (
                    boot.get("difference_ci95_low"),
                    boot.get("difference_ci95_high"),
                ),
                "bootstrap_p_holm": boot.get("p_value_holm", boot.get("p_value")),
            }
        )
    return saida
