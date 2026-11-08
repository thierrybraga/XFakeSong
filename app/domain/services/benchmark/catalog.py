"""Catálogo do benchmark: entradas do artigo, modelos instalados e receitas.

Junta, sem carregar pesos:

- as 11 entradas oficiais do artigo (``benchmarks/config.py``);
- o sidecar ``<modelo>_config.json`` de cada artefato instalado (contrato de
  inferência com a calibração, métricas de VALIDAÇÃO, histórico de épocas,
  configuração de treino);
- o ``registry.json`` da promoção, com as métricas de TESTE do run de origem,
  a eficiência e a estabilidade de treino;
- a receita de hiperparâmetros que o benchmark do artigo usa
  (``benchmarks/planning.py`` para as neurais Keras, os grids de
  ``svm.py``/``random_forest.py`` para as clássicas e a receita SUPERB do runner
  PyTorch para as SSL).

Os sidecars variam: os clássicos não têm ``metrics``, os do assistente de treino
só têm um resumo, e o ``registry.json`` pode trazer caminhos de outra máquina.
Toda leitura aqui tolera chave ausente.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

#: Amostras de ajuste do recorte do artigo: 12.162 de treino + 1 cópia AWGN.
ARTICLE_FIT_SAMPLES = 24324

#: Receita das duas entradas SSL do artigo (runner PyTorch dedicado,
#: ``benchmarks/run_wavlm_original_benchmark.py`` e o lote que o orquestrador
#: repassa em ``run_models_sequential.py --ssl-train-batch-size``).
SSL_ARTICLE_RECIPE: Dict[str, Any] = {
    "backbone": "congelado (só a cabeça treina)",
    "camadas": "soma ponderada aprendida das 13 camadas (SUPERB)",
    "pooling temporal": "média ⊕ desvio por camada",
    "cabeça": "MLP 1536 → 256 → 2, dropout 0,2",
    "otimizador": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lote (cabeça)": 128,
    "lote (extração)": 16,
    "épocas": 100,
    "seleção de época": "menor perda de validação limpa",
    "janela": "48.000 amostras (3 s a 16 kHz)",
}
SSL_BACKBONES = {
    "wavlmoriginal": "microsoft/wavlm-base-plus",
    "hubertoriginal": "facebook/hubert-base-ls960",
}

#: Chaves do ``training_config`` do plano que a tela mostra, na ordem.
PLAN_KEYS = (
    "optimizer",
    "learning_rate",
    "min_learning_rate",
    "scheduler",
    "warmup_steps",
    "decay_steps",
    "alpha",
    "weight_decay",
    "l2_reg_strength",
    "dropout_rate",
    "label_smoothing",
    "clipnorm",
    "global_clipnorm",
    "batch_size",
    "epochs",
    "use_mixed_precision",
    "pretrained",
    "select_best_checkpoint",
)


def compact(name: Any) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def discover_model_paths(models_dir: Path) -> Dict[str, Path]:
    """Artefatos que a detecção carrega, por nome, sem importar o TensorFlow.

    Mesmas regras de ``ModelLoader._discover_model_files``: a raiz de
    ``models_dir`` e os promovidos em ``benchmark_final/<arch>/``, sem os
    ``_scaler`` dos clássicos.
    """
    raiz = Path(models_dir)
    arquivos: List[Path] = []
    for ext in ("*.keras", "*.h5", "*.pkl", "*.pt"):
        arquivos.extend(raiz.glob(ext))
    finais = raiz / "benchmark_final"
    if finais.is_dir():
        for ext in ("bench_*.keras", "bench_*.h5", "bench_*.pkl", "bench_*.pt"):
            arquivos.extend(finais.glob(f"*/{ext}"))
        arquivos.extend(finais.glob("*/results/models/bench_*_original.pt"))
    saida: Dict[str, Path] = {}
    for arquivo in sorted(arquivos, key=lambda p: p.stem):
        if "_scaler" not in arquivo.stem and arquivo.stem not in saida:
            saida[arquivo.stem] = arquivo
    return saida


@dataclass(frozen=True)
class OfficialEntry:
    """Uma das 11 entradas do escopo oficial do artigo."""

    benchmark_name: str
    display_name: str
    result_key: str
    variant: str
    family: str
    input_type: str
    runner: str

    @property
    def kind(self) -> str:
        if self.family.startswith("classical"):
            return "classical"
        if "ssl" in self.family or "original" in compact(self.benchmark_name):
            return "ssl"
        return "neural"

    def matches(self, *names: Any) -> bool:
        alvos = {compact(self.benchmark_name), compact(self.result_key)}
        return any(compact(n) in alvos for n in names if n)


def official_entries() -> List[OfficialEntry]:
    """As entradas do artigo, na ordem da tabela do TCC."""
    from benchmarks.config import OFFICIAL_TCC_MODEL_MANIFEST

    return [
        OfficialEntry(
            benchmark_name=str(item["benchmark_name"]),
            display_name=str(item.get("display_name") or item["benchmark_name"]),
            result_key=str(item.get("result_key") or item["benchmark_name"]),
            variant=str(item.get("variant") or ""),
            family=str(item.get("family") or ""),
            input_type=str(item.get("input_type") or ""),
            runner=str(item.get("runner") or ""),
        )
        for item in OFFICIAL_TCC_MODEL_MANIFEST
    ]


@dataclass
class InstalledModel:
    """Um artefato instalado, descrito pelo sidecar e pelo registry."""

    name: str
    path: str
    architecture: str
    model_type: str
    entry: Optional[OfficialEntry] = None
    is_default: bool = False
    size_mb: Optional[float] = None
    calibration: Dict[str, Any] = field(default_factory=dict)
    input: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    test: Dict[str, Any] = field(default_factory=dict)
    efficiency: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.entry.display_name if self.entry else self.architecture or self.name


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        dados = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("JSON ilegível em %s: %s", path, exc)
        return {}
    return dados if isinstance(dados, dict) else {}


def _registry_by_slug(models_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Entradas do ``registry.json`` por slug (nome do artefato sem ``bench_``)."""
    saida: Dict[str, Dict[str, Any]] = {}
    for item in _read_json(Path(models_dir) / "registry.json").get("models") or []:
        if isinstance(item, dict) and item.get("slug"):
            saida[compact(item["slug"])] = item
    return saida


def _calibration(contract: Mapping[str, Any]) -> Dict[str, Any]:
    banda = contract.get("band_correction") or {}
    agregacao = contract.get("aggregation") or {}
    return {
        "temperature": contract.get("temperature"),
        "temperature_on_logits": contract.get("temperature_on_logits"),
        "eer_threshold": contract.get("eer_threshold"),
        "eer_value": contract.get("eer_value"),
        "threshold_source": contract.get("threshold_source"),
        "ood_threshold": contract.get("ood_threshold"),
        "band_cutoff_hz": banda.get("cutoff_hz") if isinstance(banda, dict) else None,
        "aggregation_method": (
            agregacao.get("method") if isinstance(agregacao, dict) else None
        ),
        "aggregation_offset": (
            agregacao.get("offset") if isinstance(agregacao, dict) else None
        ),
        "uncertainty_margin": (
            agregacao.get("uncertainty_margin") if isinstance(agregacao, dict) else None
        ),
    }


def _input(contract: Mapping[str, Any], sidecar: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "feature_frontend": contract.get("feature_frontend"),
        "input_shape": contract.get("input_shape") or sidecar.get("input_shape"),
        "sample_rate": contract.get("sample_rate"),
        "crop_strategy": contract.get("crop_strategy"),
        "eval_num_crops": contract.get("eval_num_crops"),
    }


def _history(metrics: Mapping[str, Any]) -> Dict[str, List[float]]:
    historico = metrics.get("history") or {}
    if not isinstance(historico, dict):
        return {}
    saida: Dict[str, List[float]] = {}
    for chave in ("val_eer", "val_loss", "val_accuracy", "loss", "accuracy"):
        valores = historico.get(chave)
        if isinstance(valores, list) and valores:
            try:
                saida[chave] = [float(v) for v in valores]
            except (TypeError, ValueError):
                continue
    return saida


_SHA_CACHE: Dict[tuple, str] = {}


def file_sha256(path: Path) -> Optional[str]:
    """sha256 do artefato, em cache por caminho, tamanho e data de modificação."""
    try:
        estado = Path(path).stat()
    except OSError:
        return None
    chave = (str(Path(path).resolve()), estado.st_size, estado.st_mtime_ns)
    if chave not in _SHA_CACHE:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for bloco in iter(lambda: handle.read(1 << 20), b""):
                digest.update(bloco)
        _SHA_CACHE[chave] = digest.hexdigest()
    return _SHA_CACHE[chave]


def describe_model(
    name: str,
    path: Path,
    registry: Mapping[str, Dict[str, Any]],
    entries: List[OfficialEntry],
    default_model: Optional[str] = None,
    fingerprints: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> InstalledModel:
    """Um modelo instalado, a partir do sidecar ao lado do artefato.

    As métricas de TESTE vêm do run cujo ``model_artifact_fingerprint`` bate com
    o sha256 do arquivo; sem esse elo, do ``registry.json`` só se o sha256 de lá
    também bater. Um número de teste de outro artefato não aparece.
    """
    caminho = Path(path)
    sidecar = _read_json(caminho.with_name(f"{caminho.stem}_config.json"))
    contrato = sidecar.get("input_contract") or {}
    if not isinstance(contrato, dict):
        contrato = {}
    metricas = sidecar.get("metrics") or {}
    if not isinstance(metricas, dict):
        metricas = {}
    arquitetura = str(sidecar.get("architecture") or contrato.get("architecture") or "")
    slug = name[len("bench_") :] if name.startswith("bench_") else name
    entrada = next((e for e in entries if e.matches(arquitetura, slug)), None)
    origem = registry.get(compact(slug)) or {}
    finais = metricas.get("final_metrics") or {}
    if not isinstance(finais, dict):
        finais = {}
    validacao = {
        "eer": contrato.get("eer_value", finais.get("eer")),
        "accuracy": finais.get("accuracy"),
        "auc": finais.get("roc_auc"),
        "f1": finais.get("f1_score", finais.get("f1")),
        "n": finais.get("total_samples"),
    }
    digital = file_sha256(caminho) if fingerprints is not None or origem else None
    fonte_teste = None
    teste_bruto: Mapping[str, Any] = {}
    if digital and fingerprints and digital in fingerprints:
        teste_bruto = fingerprints[digital].get("clean") or {}
        fonte_teste = fingerprints[digital].get("run_id")
    elif digital and origem.get("model_sha256") == digital:
        teste_bruto = origem.get("metrics") or {}
        fonte_teste = "registry.json"
    teste = {
        k: teste_bruto.get(k)
        for k in (
            "eer",
            "eer_ci95_low",
            "eer_ci95_high",
            "auc_roc",
            "accuracy",
            "min_tdcf",
            "ece",
        )
        if isinstance(teste_bruto, Mapping)
    }
    if fonte_teste:
        teste["source"] = fonte_teste
    eficiencia = (
        dict(fingerprints[digital].get("efficiency") or {})
        if digital and fingerprints and digital in fingerprints
        else dict(origem.get("efficiency") or {}) if fonte_teste else {}
    )
    if not eficiencia.get("params"):
        resumo = metricas.get("model_summary") or {}
        if isinstance(resumo, dict) and resumo.get("total_params"):
            eficiencia["params"] = resumo["total_params"]
    estabilidade = origem.get("training_stability") or {}
    proveniencia = {
        "created_at": sidecar.get("created_at"),
        "trained_by": sidecar.get("trained_by"),
        "best_epoch": (
            estabilidade.get("best_epoch") if isinstance(estabilidade, dict) else None
        ),
        "recalibrated_at": (
            (sidecar.get("provenance") or {})
            .get("temperature_recalibration", {})
            .get("recalibrated_at")
            if isinstance(sidecar.get("provenance"), dict)
            else None
        ),
        "registry_sha256": origem.get("model_sha256"),
    }
    try:
        tamanho = round(caminho.stat().st_size / 1e6, 2)
    except OSError:
        tamanho = None
    treino = metricas.get("training_config") or {}
    return InstalledModel(
        name=name,
        path=str(caminho),
        architecture=arquitetura,
        model_type=str(sidecar.get("model_type") or caminho.suffix.lstrip(".")),
        entry=entrada,
        is_default=bool(default_model and name == default_model),
        size_mb=tamanho,
        calibration=_calibration(contrato),
        input=_input(contrato, sidecar),
        validation=validacao,
        test={k: v for k, v in teste.items() if v is not None},
        efficiency=eficiencia,
        training_config=dict(treino) if isinstance(treino, dict) else {},
        history=_history(metricas),
        provenance=proveniencia,
    )


def installed_models(
    model_paths: Mapping[str, Path],
    models_dir: Path,
    default_model: Optional[str] = None,
    fingerprints: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> List[InstalledModel]:
    """Todos os artefatos instalados, na ordem da tabela do artigo e depois por nome.

    ``fingerprints`` é ``results.fingerprint_index(...)``: liga cada artefato às
    métricas de teste do run que o produziu.
    """
    entradas = official_entries()
    registro = _registry_by_slug(models_dir)
    modelos = [
        describe_model(
            nome, Path(caminho), registro, entradas, default_model, fingerprints
        )
        for nome, caminho in model_paths.items()
    ]
    ordem = {e.benchmark_name: i for i, e in enumerate(entradas)}
    return sorted(
        modelos,
        key=lambda m: (
            ordem.get(m.entry.benchmark_name, 99) if m.entry else 99,
            m.name,
        ),
    )


def expected_hours(entry: OfficialEntry) -> Optional[float]:
    """Horas de GPU estimadas para a entrada no recorte do artigo."""
    from benchmarks.planning import expected_training_timeout_min

    try:
        return (
            expected_training_timeout_min(
                entry.benchmark_name,
                device_profile="gpu",
                epochs=100,
                fit_samples=ARTICLE_FIT_SAMPLES,
                safety_factor=1.0,
                minimum_min=0.0,
            )
            / 60.0
        )
    except Exception:  # noqa: BLE001 - custo é informativo
        return None


def article_hyperparameters(entry: OfficialEntry) -> Dict[str, Any]:
    """A receita que o benchmark do artigo usa para a entrada, para exibir."""
    if entry.kind == "classical":
        if compact(entry.benchmark_name) == "svm":
            from app.domain.models.architectures.svm import SVM_PARAM_GRID as grid
        else:
            from app.domain.models.architectures.random_forest import (
                RANDOM_FOREST_PARAM_GRID as grid,
            )
        return {
            "busca": "GridSearchCV por roc_auc",
            "validação cruzada": "StratifiedGroupKFold, 5 dobras",
            "vetor de entrada": "benchmark_tabular_v2 (183 descritores)",
            "candidatos": _grid_size(grid),
            "grid": _grid_text(grid),
            "calibração": "isotônica com dobras agrupadas; limiar de EER na validação",
        }
    if entry.kind == "ssl":
        receita = dict(SSL_ARTICLE_RECIPE)
        receita["checkpoint"] = SSL_BACKBONES.get(compact(entry.benchmark_name), "?")
        return receita
    from benchmarks.planning import article_training_config

    try:
        plano = article_training_config(entry.benchmark_name, device_profile="gpu")
    except ValueError as exc:
        return {"erro": str(exc)}
    saida = {k: plano[k] for k in PLAN_KEYS if plano.get(k) is not None}
    if plano.get("notes"):
        saida["nota do plano"] = str(plano["notes"])
    return saida


def _grid_size(grid: Any) -> int:
    blocos = grid if isinstance(grid, list) else [grid]
    total = 0
    for bloco in blocos:
        produto = 1
        for valores in bloco.values():
            produto *= len(valores)
        total += produto
    return total


def _grid_text(grid: Any) -> str:
    blocos = grid if isinstance(grid, list) else [grid]
    partes = []
    for bloco in blocos:
        partes.append(
            "; ".join(
                f"{k.split('__', 1)[-1]} ∈ {{{', '.join(str(v) for v in valores)}}}"
                for k, valores in bloco.items()
            )
        )
    return " | ".join(partes)
