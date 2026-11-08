"""Aba 📊 Benchmark: o estudo do TCC na interface.

Cinco sub-abas:

- **Artigo**: título, resumo e PDF do ``main.tex``, o protocolo e as
  limitações, e a tabela de resultados do artigo com a proveniência;
- **Modelos**: cada modelo instalado com a calibração que a detecção aplica,
  métricas de validação e de teste do run que o produziu, curva de convergência
  e atalhos para a detecção e o treino;
- **Hiperparâmetros**: a receita do artigo por entrada e o custo;
- **Resultados**: qualquer run em ``data/results``, com tabela, gráficos e as
  figuras do relatório;
- **Executar o TCC**: as pré-condições, o comando exato e o lançador com travas
  (``app/domain/services/benchmark/runner.py``).

Toda leitura passa pelo domínio (``app/domain/services/benchmark``), que só
devolve métricas agregadas: nada da partição de teste por amostra chega aqui.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from app.interfaces.gradio.presenters.benchmark_views import (
    article_html,
    lr,
    model_detail_html,
    num,
    pct,
    preflight_html,
    protocol_html,
    run_status_html,
)
from app.interfaces.gradio.utils.components import info_callout, page_header, ui_safe
from app.interfaces.gradio.utils.plotting import (
    PLOT_ACCENT,
    PLOT_ACCENT2,
    PLOT_DANGER,
    PLOT_FACE,
    PLOT_GRID,
    PLOT_TEXT,
    PLOT_WARNING,
    new_figure,
    safe_tight_layout,
    style_ax,
)

logger = logging.getLogger("gradio_benchmark_tab")

#: Nome do assistente de treino para cada entrada do artigo (``get_model_catalog``).
_WIZARD_NAMES = {"RandomForest": "Random Forest"}
_STATUS_REFRESH_S = 15.0


# ──────────────────────────── fontes ────────────────────────────


def _paths():
    from app.core.bootstrap import OperationalPaths

    return OperationalPaths.resolve()


def _paper_dir() -> Path:
    return _paths().results / "paper"


def _default_model() -> Optional[str]:
    try:
        from app.interfaces.gradio.tabs.detection import get_detection_service

        service = get_detection_service()
        return getattr(service, "default_model", None) if service else None
    except Exception:  # noqa: BLE001 - sem serviço, sem marca de padrão
        return None


def _installed():
    from app.domain.services.benchmark import catalog, results

    caminhos = _paths()
    return catalog.installed_models(
        catalog.discover_model_paths(caminhos.models),
        caminhos.models,
        _default_model(),
        results.fingerprint_index(caminhos.results),
    )


def _export(origem: str, nome: str) -> Optional[str]:
    """Copia um arquivo para a pasta de exports, a única que a interface serve."""
    from app.utils.file_utils import get_gradio_exports_directory

    fonte = Path(origem)
    if not fonte.is_file():
        return None
    destino = get_gradio_exports_directory() / "benchmark" / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (
            not destino.exists()
            or destino.stat().st_mtime < fonte.stat().st_mtime
            or destino.stat().st_size != fonte.stat().st_size
        ):
            shutil.copy2(fonte, destino)
    except OSError as exc:
        logger.warning("Não foi possível copiar %s: %s", fonte, exc)
        return None
    return str(destino)


def _slug(texto: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(texto)).strip("_")


# ──────────────────────────── artigo ────────────────────────────


def _article_frame() -> pd.DataFrame:
    from app.domain.services.benchmark.results import article_summary

    linhas = article_summary(_paper_dir())
    return pd.DataFrame(
        [
            {
                "Modelo": linha["model"],
                "EER": pct(linha["eer"]),
                "IC 95% do EER": f"{pct(linha['eer_ci95_low'])} – "
                f"{pct(linha['eer_ci95_high'])}",
                "AUC": num(linha["auc"], 4),
                "Acurácia": pct(linha["accuracy"]),
                "EER 10 dB": pct(linha["eer_10db"]),
                "EER 5 dB": pct(linha["eer_5db"]),
                "Pior locutor": pct(linha["worst_speaker_accuracy"], 1),
            }
            for linha in linhas
        ],
        columns=[
            "Modelo",
            "EER",
            "IC 95% do EER",
            "AUC",
            "Acurácia",
            "EER 10 dB",
            "EER 5 dB",
            "Pior locutor",
        ],
    )


def _article_source_note() -> str:
    from app.domain.services.benchmark.results import article_summary

    fontes = sorted(
        {
            linha["source"].split("/")[-2]
            for linha in article_summary(_paper_dir())
            if "/" in linha["source"]
        }
    )
    if not fontes:
        return ""
    return (
        "Números do artigo, consolidados de "
        + ", ".join(f"<code>{f}</code>" for f in fontes)
        + ". Esse run é do protocolo antigo (sem correção de banda, seleção pela "
        "perda): a sub-aba Resultados mostra o run vigente, do protocolo corrigido."
    )


def _significance_frame() -> pd.DataFrame:
    from app.domain.services.benchmark.results import article_significance

    pares = article_significance(_paper_dir())
    return pd.DataFrame(
        [
            {
                "Modelo A": p["model_a"],
                "Modelo B": p["model_b"],
                "Δ EER (A − B)": pct(p["eer_difference"]),
                "IC 95% da diferença": f"{pct(p['eer_difference_ci95'][0])} – "
                f"{pct(p['eer_difference_ci95'][1])}",
                "McNemar p (Holm)": num(p["mcnemar_p_holm"], 3),
            }
            for p in pares
        ],
        columns=[
            "Modelo A",
            "Modelo B",
            "Δ EER (A − B)",
            "IC 95% da diferença",
            "McNemar p (Holm)",
        ],
    )


@ui_safe("Falha ao preparar o PDF do artigo")
def _download_pdf():
    from app.domain.services.benchmark.article import article_info

    info = article_info(_paper_dir())
    if not info.pdf_path:
        raise gr.Error("Não há main.pdf em data/results/paper.")
    caminho = _export(info.pdf_path, "artigo_tcc.pdf")
    return gr.update(value=caminho, visible=bool(caminho))


# ──────────────────────────── modelos ────────────────────────────

_MODEL_COLUMNS = [
    "Modelo",
    "Família",
    "Padrão",
    "EER teste",
    "EER validação",
    "Limiar",
    "Temperatura",
    "OOD",
    "Banda",
    "Agregação",
    "Parâmetros",
    "MB",
    "Latência (ms)",
]


def _models_frame(modelos) -> pd.DataFrame:
    from app.domain.services.benchmark.catalog import official_entries

    linhas: List[Dict[str, Any]] = []
    instalados = set()
    for m in modelos:
        if m.entry:
            instalados.add(m.entry.benchmark_name)
        cal = m.calibration
        linhas.append(
            {
                "Modelo": m.display_name,
                "Família": m.entry.family if m.entry else "fora do artigo",
                "Padrão": "✓" if m.is_default else "",
                "EER teste": pct(m.test.get("eer")) if m.test else "—",
                "EER validação": pct(cal.get("eer_value")),
                "Limiar": num(cal.get("eer_threshold"), 4),
                "Temperatura": num(cal.get("temperature"), 2),
                "OOD": num(cal.get("ood_threshold"), 4),
                "Banda": (
                    f"{num(cal.get('band_cutoff_hz'), 0)} Hz"
                    if cal.get("band_cutoff_hz")
                    else "—"
                ),
                "Agregação": cal.get("aggregation_method") or "padrão",
                "Parâmetros": num(m.efficiency.get("params"), 0),
                "MB": num(m.size_mb, 1),
                "Latência (ms)": num(m.efficiency.get("latency_ms"), 1),
            }
        )
    for entrada in official_entries():
        if entrada.benchmark_name not in instalados:
            linhas.append(
                {
                    "Modelo": entrada.display_name,
                    "Família": entrada.family,
                    "Padrão": "",
                    "EER teste": "não instalado",
                    **dict.fromkeys(_MODEL_COLUMNS[4:], "—"),
                }
            )
    return pd.DataFrame(linhas, columns=_MODEL_COLUMNS)


def _model_choices(modelos) -> List[Tuple[str, str]]:
    return [(f"{m.display_name} ({m.name})", m.name) for m in modelos]


def _plan_vs_trained(modelo) -> pd.DataFrame:
    from app.domain.services.benchmark.catalog import article_hyperparameters

    plano = article_hyperparameters(modelo.entry) if modelo and modelo.entry else {}
    treino = modelo.training_config if modelo else {}
    chaves = list(plano)
    return pd.DataFrame(
        [
            {
                "Hiperparâmetro": chave,
                "Plano do artigo": str(plano.get(chave, "—")),
                "Gravado no treino": str(treino.get(chave, "—")),
            }
            for chave in chaves
        ],
        columns=["Hiperparâmetro", "Plano do artigo", "Gravado no treino"],
    )


def _convergence_figure(historico: Dict[str, List[float]], titulo: str):
    if not historico or not (historico.get("val_eer") or historico.get("val_loss")):
        return None
    fig, ax = new_figure(figsize=(10, 3.6))
    style_ax(ax, fig, titulo, "Época", "Valor")
    cores = {"val_eer": PLOT_DANGER, "val_loss": PLOT_ACCENT, "loss": PLOT_ACCENT2}
    rotulos = {
        "val_eer": "EER de validação",
        "val_loss": "perda de validação",
        "loss": "perda de treino",
    }
    for chave in ("val_eer", "val_loss", "loss"):
        valores = historico.get(chave)
        if valores:
            ax.plot(
                range(1, len(valores) + 1),
                valores,
                color=cores[chave],
                linewidth=1.5,
                label=rotulos[chave],
            )
    eer = historico.get("val_eer")
    if eer:
        melhor = min(range(len(eer)), key=lambda i: eer[i])
        ax.axvline(melhor + 1, color=PLOT_WARNING, linestyle="--", linewidth=1)
        ax.annotate(
            f"melhor época {melhor + 1}",
            (melhor + 1, eer[melhor]),
            color=PLOT_TEXT,
            fontsize=8,
            xytext=(6, 10),
            textcoords="offset points",
        )
    ax.legend(
        facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT, fontsize=8
    )
    safe_tight_layout(fig)
    return fig


def _find(modelos, nome):
    return next((m for m in modelos if m.name == nome), None)


@ui_safe("Falha ao carregar o modelo")
def _show_model(nome):
    modelos = _installed()
    modelo = _find(modelos, nome)
    if modelo is None:
        return model_detail_html(None), _plan_vs_trained(None), None
    return (
        model_detail_html(modelo),
        _plan_vs_trained(modelo),
        _convergence_figure(modelo.history, f"Convergência: {modelo.display_name}"),
    )


@ui_safe("Falha ao atualizar a lista de modelos")
def _refresh_models():
    """Tabela, lista e detalhe do modelo padrão (ou do primeiro instalado)."""
    modelos = _installed()
    escolhas = _model_choices(modelos)
    padrao = next((m.name for m in modelos if m.is_default), None)
    valor = padrao or (escolhas[0][1] if escolhas else None)
    modelo = _find(modelos, valor)
    return (
        _models_frame(modelos),
        gr.update(choices=escolhas, value=valor),
        model_detail_html(modelo),
        _plan_vs_trained(modelo),
        _convergence_figure(
            modelo.history if modelo else {},
            f"Convergência: {modelo.display_name}" if modelo else "",
        ),
    )


def _wizard_name(modelo) -> Optional[str]:
    if modelo is None or modelo.entry is None or modelo.entry.kind == "ssl":
        return None
    return _WIZARD_NAMES.get(modelo.entry.benchmark_name, modelo.entry.benchmark_name)


@ui_safe("Falha ao abrir a detecção")
def _go_detect(nome):
    if not nome:
        raise gr.Error("Escolha um modelo instalado.")
    return gr.Tabs(selected="tab_detect"), gr.update(value=nome)


@ui_safe("Falha ao abrir o treino")
def _go_train(nome):
    alvo = _wizard_name(_find(_installed(), nome))
    if alvo is None:
        raise gr.Error(
            "As entradas SSL do artigo (WavLM e HuBERT Original) treinam pelo runner "
            "PyTorch do benchmark; o assistente não as reproduz."
        )
    gr.Info(f"Arquitetura {alvo} escolhida. Indique o dataset no passo 1.")
    return gr.Tabs(selected="tab_train"), gr.update(value=alvo)


# ──────────────────────────── hiperparâmetros ────────────────────────────


def _measured_hours() -> Dict[str, float]:
    from app.domain.services.benchmark.catalog import compact
    from app.domain.services.benchmark.results import _read_json

    resumo = _read_json(_paths().results / "benchmark" / "run_summary.json") or {}
    saida: Dict[str, float] = {}
    for item in resumo.get("models") or []:
        if isinstance(item, dict) and item.get("elapsed_s"):
            saida[compact(item.get("model"))] = float(item["elapsed_s"]) / 3600
    return saida


def _hparams_frame() -> pd.DataFrame:
    from app.domain.services.benchmark.catalog import (
        article_hyperparameters,
        compact,
        expected_hours,
        official_entries,
    )

    medidas = _measured_hours()
    linhas = []
    for e in official_entries():
        receita = article_hyperparameters(e)
        if e.kind == "neural":
            resumo = (
                f"{receita.get('optimizer', '?')}, "
                f"LR {lr(receita.get('learning_rate'))}, "
                f"{receita.get('scheduler', '?')}, lote {receita.get('batch_size', '?')}, "
                f"dropout {num(receita.get('dropout_rate'), 2)}"
            )
        elif e.kind == "classical":
            resumo = f"{receita.get('candidatos')} candidatos, CV agrupada de 5 dobras"
        else:
            resumo = f"{receita.get('checkpoint')} congelado, AdamW 1e-3, lote 128"
        medido = medidas.get(compact(e.benchmark_name))
        linhas.append(
            {
                "Entrada": e.display_name,
                "Família": e.family,
                "Receita": resumo,
                "Custo estimado (h GPU)": num(expected_hours(e), 2),
                "Medido no run vigente (h)": num(medido, 2) if medido else "—",
            }
        )
    total = sum((expected_hours(e) or 0) for e in official_entries())
    linhas.append(
        {
            "Entrada": "Total",
            "Família": "",
            "Receita": "",
            "Custo estimado (h GPU)": num(total, 1),
            "Medido no run vigente (h)": (
                num(sum(medidas.values()), 1) if medidas else "—"
            ),
        }
    )
    return pd.DataFrame(linhas)


@ui_safe("Falha ao carregar a receita")
def _entry_recipe(nome):
    from app.domain.services.benchmark.catalog import (
        article_hyperparameters,
        official_entries,
    )

    entrada = next((e for e in official_entries() if e.display_name == nome), None)
    receita = article_hyperparameters(entrada) if entrada else {}
    return pd.DataFrame(
        [{"Hiperparâmetro": k, "Valor": str(v)} for k, v in receita.items()],
        columns=["Hiperparâmetro", "Valor"],
    )


# ──────────────────────────── resultados ────────────────────────────

_RESULT_COLUMNS = [
    "Modelo",
    "EER",
    "IC 95% do EER",
    "AUC",
    "Acurácia",
    "min t-DCF",
    "EER 10 dB",
    "EER 5 dB*",
    "Pior locutor",
    "Melhor época",
    "Latência (ms)",
]


def _run_choices() -> List[Tuple[str, str]]:
    from app.domain.services.benchmark.results import discover_runs

    return [
        (f"{r.label} — {r.n_models} modelos", r.run_id)
        for r in discover_runs(_paths().results)
    ]


def _load_view(run_id):
    from app.domain.services.benchmark.results import load_run, resolve_run

    if not run_id:
        return None
    pasta = resolve_run(_paths().results, run_id)
    return load_run(pasta, run_id)


def _results_frame(view) -> pd.DataFrame:
    if view is None:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    linhas = []
    for m in view.models:
        c = m.clean
        epoca = m.stability.get("best_epoch")
        linhas.append(
            {
                "Modelo": m.display_name + (" †" if m.superseded else ""),
                "EER": pct(c.get("eer")),
                "IC 95% do EER": f"{pct(c.get('eer_ci95_low'))} – {pct(c.get('eer_ci95_high'))}",
                "AUC": num(c.get("auc_roc"), 4),
                "Acurácia": pct(c.get("accuracy")),
                "min t-DCF": num(c.get("min_tdcf"), 3),
                "EER 10 dB": pct((m.robustness.get("10") or {}).get("eer")),
                "EER 5 dB*": pct((m.robustness.get("5") or {}).get("eer")),
                "Pior locutor": pct(m.worst_speaker_accuracy, 1),
                "Melhor época": f"{epoca}/100" if epoca else "—",
                "Latência (ms)": num(m.efficiency.get("latency_ms"), 1),
            }
        )
    return pd.DataFrame(linhas, columns=_RESULT_COLUMNS)


def _eer_figure(view):
    modelos = [
        m for m in (view.models if view else []) if m.clean.get("eer") is not None
    ]
    if not modelos:
        return None
    fig, ax = new_figure(figsize=(10, 3.8))
    style_ax(ax, fig, "EER limpo com IC 95%", "", "EER (%)")
    nomes = [m.display_name for m in modelos]
    eer = [100 * m.clean["eer"] for m in modelos]
    baixo = [
        100 * (m.clean["eer"] - m.clean.get("eer_ci95_low", m.clean["eer"]))
        for m in modelos
    ]
    alto = [
        100 * (m.clean.get("eer_ci95_high", m.clean["eer"]) - m.clean["eer"])
        for m in modelos
    ]
    ax.bar(
        nomes, eer, color=PLOT_ACCENT, yerr=[baixo, alto], ecolor=PLOT_TEXT, capsize=3
    )
    ax.tick_params(axis="x", rotation=35)
    safe_tight_layout(fig)
    return fig


def _robustness_figure(view):
    modelos = [m for m in (view.models if view else []) if m.robustness]
    if not modelos:
        return None
    fig, ax = new_figure(figsize=(10, 3.8))
    style_ax(
        ax, fig, "EER por condição de ruído (5 dB fica fora do treino)", "", "EER (%)"
    )
    condicoes = ["limpo", "30", "20", "10", "5"]
    for m in modelos:
        valores = [m.clean.get("eer")] + [
            (m.robustness.get(s) or {}).get("eer") for s in condicoes[1:]
        ]
        pontos = [100 * v if v is not None else math.nan for v in valores]
        ax.plot(condicoes, pontos, marker="o", linewidth=1.2, label=m.display_name)
    ax.set_xlabel("Condição (dB)")
    ax.legend(
        facecolor=PLOT_FACE,
        edgecolor=PLOT_GRID,
        labelcolor=PLOT_TEXT,
        fontsize=7,
        ncol=2,
    )
    safe_tight_layout(fig)
    return fig


@ui_safe("Falha ao carregar o run")
def _show_run(run_id):
    view = _load_view(run_id)
    escolhas = [(m.display_name, m.folder) for m in (view.models if view else [])]
    nota = ""
    if view is not None:
        partes = [f"Run <code>{view.info.run_id}</code>"]
        if view.info.status:
            partes.append(f"estado do resumo: {view.info.status}")
        if any(m.superseded for m in view.models):
            partes.append(
                "† resultado válido mantido em <code>_superseded/</code> "
                "(a pasta principal guarda uma execução substituída)"
            )
        partes.append("* 5 dB não aparece no treino: mede generalização a ruído")
        nota = " · ".join(partes)
    return (
        _results_frame(view),
        _eer_figure(view),
        _robustness_figure(view),
        gr.update(choices=escolhas, value=escolhas[0][1] if escolhas else None),
        nota,
    )


@ui_safe("Falha ao carregar o modelo do run")
def _show_run_model(run_id, folder):
    view = _load_view(run_id)
    modelo = next(
        (m for m in (view.models if view else []) if m.folder == folder), None
    )
    if modelo is None:
        return None, []
    figuras = []
    for nome, caminho in modelo.figures.items():
        exportado = _export(caminho, f"{_slug(run_id)}/{modelo.folder}_{nome}.png")
        if exportado:
            figuras.append((exportado, nome))
    return (
        _convergence_figure(modelo.history, f"Convergência: {modelo.display_name}"),
        figuras,
    )


# ──────────────────────────── execução ────────────────────────────


def _auth_enabled() -> bool:
    try:
        from app.core.auth.auth_handler import get_gradio_auth

        return get_gradio_auth() is not None
    except Exception:  # noqa: BLE001 - credencial incompleta não é login ativo
        return False


def _gpu() -> bool:
    try:
        from app.core.gpu import is_gpu_available

        return bool(is_gpu_available())
    except Exception:  # noqa: BLE001
        return False


def _new_spec():
    from app.domain.services.benchmark.runner import ArticleRunSpec

    return ArticleRunSpec.from_paths(_paths())


def _ui_runs() -> List[str]:
    from app.domain.services.benchmark.runner import RUN_PREFIX, UI_MARKER

    raiz = _paths().results
    if not raiz.is_dir():
        return []
    return sorted(
        (p.name for p in raiz.glob(f"{RUN_PREFIX}*") if (p / UI_MARKER).is_file()),
        reverse=True,
    )


def _commands(spec) -> Tuple[str, str, str]:
    from app.domain.services.benchmark.runner import build_command, docker_command

    nativo = " ".join(
        f'"{parte}"' if " " in parte else parte for parte in build_command(spec)
    )
    return nativo, docker_command(spec, "powershell"), docker_command(spec, "bash")


@ui_safe("Falha ao conferir as pré-condições")
def _check(ack):
    from app.domain.services.benchmark.runner import Check, can_start, preflight

    spec = _new_spec()
    checks = preflight(spec, gpu_available=_gpu, auth_enabled=_auth_enabled())
    checks.append(
        Check(
            "Confirmação",
            bool(ack),
            (
                "duração e custo aceitos"
                if ack
                else "marque a confirmação de que o run leva ~58 h de GPU"
            ),
        )
    )
    pronto = can_start(checks)
    nativo, ps, bash = _commands(spec)
    return (
        preflight_html(checks, pronto),
        gr.update(interactive=pronto),
        nativo,
        ps,
        bash,
    )


def _status_outputs(run_name: Optional[str] = None):
    from app.domain.services.benchmark.runner import run_status

    raiz = _paths().results
    status = run_status(raiz, raiz / run_name if run_name else None)
    return run_status_html(status), "\n".join(status.log_tail)


@ui_safe("Falha ao iniciar o benchmark")
def _start(ack):
    from app.domain.services.benchmark.runner import Check, preflight, start

    if not ack:
        raise gr.Error("Confirme que o run leva cerca de 58 h de GPU.")
    spec = _new_spec()
    checks = preflight(spec, gpu_available=_gpu, auth_enabled=_auth_enabled())
    checks.append(Check("Confirmação", True, "aceita"))
    trava = start(spec, checks)
    gr.Info(f"Run {trava['run_name']} iniciado (pid {trava['pid']}).")
    html, log = _status_outputs(spec.run_name)
    return (
        html,
        log,
        gr.update(choices=_ui_runs(), value=spec.run_name),
        gr.Timer(active=True),
    )


@ui_safe("Falha ao parar o benchmark")
def _stop():
    from app.domain.services.benchmark.runner import stop

    trava = stop(_paths().results)
    if trava is None:
        gr.Info("Nenhum run em andamento.")
        return (*_status_outputs(), gr.Timer(active=False))
    gr.Warning(f"Run {trava['run_name']} parado. Retome pela lista quando quiser.")
    return (*_status_outputs(trava["run_name"]), gr.Timer(active=False))


@ui_safe("Falha ao retomar o benchmark")
def _resume(run_name):
    from app.domain.services.benchmark.runner import ArticleRunSpec, preflight, start

    if not run_name:
        raise gr.Error("Escolha um run iniciado pela aba.")
    spec = ArticleRunSpec.from_paths(_paths(), run_name=run_name)
    checks = preflight(
        spec, gpu_available=_gpu, auth_enabled=_auth_enabled(), resume=True
    )
    trava = start(spec, checks, resume=True)
    gr.Info(f"Run {trava['run_name']} retomado (pid {trava['pid']}).")
    return (*_status_outputs(run_name), gr.Timer(active=True))


@ui_safe("Falha ao consolidar o run")
def _consolidate(run_name):
    from app.domain.services.benchmark.runner import (
        ArticleRunSpec,
        active_run,
        consolidate,
    )

    if not run_name:
        raise gr.Error("Escolha um run iniciado pela aba.")
    if active_run(_paths().results):
        raise gr.Error("Espere o run terminar (ou pare-o) antes de consolidar.")
    pid = consolidate(ArticleRunSpec.from_paths(_paths(), run_name=run_name))
    gr.Info(
        f"Consolidação iniciada (pid {pid}): tabela, figuras e testes pareados vão para "
        f"{run_name}/consolidated."
    )


@ui_safe("Falha ao atualizar o estado do run")
def _tick(run_name):
    return _status_outputs(run_name)


# ──────────────────────────── construção ────────────────────────────


def _safe(fn, *args, default=None):
    """Valor inicial da construção: a aba monta mesmo sem dados."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Aba Benchmark: %s indisponível: %s", getattr(fn, "__name__", fn), exc
        )
        return default


def create_benchmark_tab(main_tabs=None, model_select=None, arch_select=None):
    """Constrói a aba 📊 Benchmark.

    ``main_tabs``, ``model_select`` e ``arch_select`` ligam os atalhos à aba
    Detectar e ao assistente de treino; sem eles, os botões não aparecem.
    """
    from app.domain.services.benchmark.article import article_info
    from app.domain.services.benchmark.catalog import official_entries

    with gr.Tab("📊 Benchmark", id="tab_benchmark") as aba:
        page_header(
            "📊",
            "Benchmark do TCC",
            "Os modelos do estudo, com hiperparâmetros, calibração e resultados, e a "
            "reprodução do artigo.",
        )
        with gr.Tabs():
            # ── Artigo ──
            with gr.Tab("📄 Artigo", id="tab_bench_article"):
                gr.HTML(
                    _safe(lambda: article_html(article_info(_paper_dir())), default="")
                )
                with gr.Row(elem_classes="action-row"):
                    pdf_btn = gr.Button("Baixar o PDF do artigo", variant="secondary")
                pdf_file = gr.File(
                    label="Artigo (PDF)", visible=False, interactive=False
                )
                pdf_btn.click(_download_pdf, outputs=[pdf_file])
                gr.HTML(protocol_html())
                gr.Markdown("#### Resultados publicados no artigo")
                nota = _safe(_article_source_note, default="")
                if nota:
                    info_callout(nota, variant="warning")
                gr.Dataframe(
                    value=_safe(_article_frame, default=pd.DataFrame()),
                    interactive=False,
                    wrap=True,
                )
                with gr.Accordion("Comparações pareadas entre modelos", open=False):
                    gr.Markdown(
                        "McNemar e bootstrap pareado por locutor, com correção de "
                        "Holm. Diferença não significativa quer dizer que os dois "
                        "modelos são indistinguíveis neste teste."
                    )
                    gr.Dataframe(
                        value=_safe(_significance_frame, default=pd.DataFrame()),
                        interactive=False,
                        wrap=True,
                    )

            # ── Modelos ──
            with gr.Tab("🧠 Modelos", id="tab_bench_models"):
                gr.Markdown(
                    "Os modelos que a detecção carrega, com a calibração que ela "
                    "aplica. As métricas de teste vêm do run que produziu cada "
                    "artefato, identificado pela impressão digital do arquivo."
                )
                # Carrega ao abrir a aba: o elo com o run lê o sha256 de cada
                # artefato (centenas de MB), que não cabe no boot do app.
                models_df = gr.Dataframe(
                    value=pd.DataFrame(columns=_MODEL_COLUMNS),
                    interactive=False,
                    wrap=True,
                )
                with gr.Row(elem_classes="action-row"):
                    model_pick = gr.Dropdown(choices=[], label="Modelo", scale=3)
                    refresh_models_btn = gr.Button("Atualizar", scale=1)
                with gr.Row(elem_classes="action-row"):
                    detect_btn = gr.Button(
                        "🎯 Testar na detecção",
                        variant="primary",
                        visible=main_tabs is not None and model_select is not None,
                    )
                    train_btn = gr.Button(
                        "🎓 Treinar esta arquitetura",
                        variant="secondary",
                        visible=main_tabs is not None and arch_select is not None,
                    )
                detail_html = gr.HTML()
                with gr.Accordion(
                    "Hiperparâmetros: plano do artigo × treino", open=False
                ):
                    hp_df = gr.Dataframe(interactive=False, wrap=True)
                convergence_plot = gr.Plot(label="Convergência na validação")

                model_pick.change(
                    _show_model,
                    inputs=[model_pick],
                    outputs=[detail_html, hp_df, convergence_plot],
                )
                refresh_models_btn.click(
                    _refresh_models,
                    outputs=[
                        models_df,
                        model_pick,
                        detail_html,
                        hp_df,
                        convergence_plot,
                    ],
                )
                if main_tabs is not None and model_select is not None:
                    detect_btn.click(
                        _go_detect,
                        inputs=[model_pick],
                        outputs=[main_tabs, model_select],
                    )
                if main_tabs is not None and arch_select is not None:
                    train_btn.click(
                        _go_train,
                        inputs=[model_pick],
                        outputs=[main_tabs, arch_select],
                    )

            # ── Hiperparâmetros ──
            with gr.Tab("⚙️ Hiperparâmetros", id="tab_bench_hparams"):
                gr.Markdown(
                    "A receita que o benchmark do artigo usa em cada entrada, no "
                    "perfil de GPU, e o custo em horas de GPU (RTX 3060) no recorte "
                    "de 15 mil."
                )
                gr.Dataframe(
                    value=_safe(_hparams_frame, default=pd.DataFrame()),
                    interactive=False,
                    wrap=True,
                )
                nomes = [
                    e.display_name for e in _safe(official_entries, default=[]) or []
                ]
                entry_pick = gr.Dropdown(
                    choices=nomes, value=nomes[0] if nomes else None, label="Entrada"
                )
                recipe_df = gr.Dataframe(
                    value=_safe(
                        _entry_recipe, nomes[0] if nomes else None, default=None
                    ),
                    interactive=False,
                    wrap=True,
                )
                entry_pick.change(
                    _entry_recipe, inputs=[entry_pick], outputs=[recipe_df]
                )

            # ── Resultados ──
            with gr.Tab("📈 Resultados", id="tab_bench_results"):
                runs = _safe(_run_choices, default=[]) or []
                with gr.Row(elem_classes="action-row"):
                    run_pick = gr.Dropdown(
                        choices=runs,
                        value=runs[0][1] if runs else None,
                        label="Run",
                        scale=3,
                    )
                    refresh_runs_btn = gr.Button("Atualizar", scale=1)
                run_note = gr.HTML()
                results_df = gr.Dataframe(interactive=False, wrap=True)
                with gr.Row(elem_classes="responsive-grid plot-grid"):
                    with gr.Column(min_width=320):
                        eer_plot = gr.Plot(label="EER com IC 95%")
                    with gr.Column(min_width=320):
                        robust_plot = gr.Plot(label="EER por condição")
                run_model_pick = gr.Dropdown(label="Modelo do run")
                run_convergence = gr.Plot(label="Convergência na validação")
                gallery = gr.Gallery(
                    label="Figuras do relatório", columns=3, height="auto"
                )
                run_pick.change(
                    _show_run,
                    inputs=[run_pick],
                    outputs=[
                        results_df,
                        eer_plot,
                        robust_plot,
                        run_model_pick,
                        run_note,
                    ],
                )
                run_model_pick.change(
                    _show_run_model,
                    inputs=[run_pick, run_model_pick],
                    outputs=[run_convergence, gallery],
                )
                refresh_runs_btn.click(
                    lambda: gr.update(choices=_safe(_run_choices, default=[]) or []),
                    outputs=[run_pick],
                )

            # ── Executar ──
            with gr.Tab("▶️ Executar o TCC", id="tab_bench_run") as run_tab:
                gr.Markdown(
                    "Reproduz o estudo com o dataset, os modelos e as configurações "
                    "do artigo. Cada run grava numa pasta nova "
                    "(`data/results/tcc_reproducao_<data>`) e numa pasta de modelos "
                    "própria: os modelos da detecção e o run vigente não são tocados."
                )
                info_callout(
                    "São cerca de 58 h de GPU (RTX 3060), sendo 29 h só do RawGAT-ST. "
                    "O teste já foi observado: um run novo é reprodução, não "
                    "confirmação.",
                    variant="warning",
                )
                ack = gr.Checkbox(
                    label="Entendo que o run leva cerca de 58 h de GPU e ocupa a "
                    "máquina até terminar",
                    value=False,
                )
                preflight_out = gr.HTML()
                with gr.Row(elem_classes="action-row"):
                    check_btn = gr.Button("Conferir pré-condições", variant="secondary")
                    start_btn = gr.Button(
                        "▶ Iniciar o benchmark do artigo",
                        variant="primary",
                        interactive=False,
                    )
                    stop_btn = gr.Button("⏹ Parar", variant="stop")
                with gr.Accordion("Comando exato", open=False):
                    native_cmd = gr.Code(label="Nesta máquina", language="shell")
                    gr.Markdown(
                        "Sem GPU aqui, rode pelo compose de benchmark numa máquina "
                        "com GPU NVIDIA. O `XFAKE_BENCHMARK_OUT` do `.env` aponta para "
                        "uma pasta renomeada: as linhas abaixo o substituem."
                    )
                    ps_cmd = gr.Code(label="Docker (PowerShell)", language="shell")
                    bash_cmd = gr.Code(label="Docker (bash)", language="shell")
                gr.Markdown("#### Andamento")
                with gr.Row(elem_classes="action-row"):
                    runs_pick = gr.Dropdown(
                        choices=_safe(_ui_runs, default=[]) or [],
                        label="Runs iniciados pela aba",
                        scale=3,
                    )
                    resume_btn = gr.Button("Retomar", scale=1)
                    consolidate_btn = gr.Button("Consolidar", scale=1)
                status_out = gr.HTML()
                log_out = gr.Code(label="Saída do orquestrador", language="shell")
                timer = gr.Timer(_STATUS_REFRESH_S, active=False)

                check_btn.click(
                    _check,
                    inputs=[ack],
                    outputs=[preflight_out, start_btn, native_cmd, ps_cmd, bash_cmd],
                )
                ack.change(
                    _check,
                    inputs=[ack],
                    outputs=[preflight_out, start_btn, native_cmd, ps_cmd, bash_cmd],
                )
                start_btn.click(
                    _start,
                    inputs=[ack],
                    outputs=[status_out, log_out, runs_pick, timer],
                )
                stop_btn.click(_stop, outputs=[status_out, log_out, timer])
                resume_btn.click(
                    _resume, inputs=[runs_pick], outputs=[status_out, log_out, timer]
                )
                consolidate_btn.click(_consolidate, inputs=[runs_pick])
                runs_pick.change(
                    _tick, inputs=[runs_pick], outputs=[status_out, log_out]
                )
                timer.tick(_tick, inputs=[runs_pick], outputs=[status_out, log_out])
                run_tab.select(
                    _check,
                    inputs=[ack],
                    outputs=[preflight_out, start_btn, native_cmd, ps_cmd, bash_cmd],
                )
                run_tab.select(
                    lambda: gr.Timer(active=True),
                    outputs=[timer],
                    show_progress="hidden",
                )

        # O primeiro conteúdo das sub-abas Modelos e Resultados vem ao abrir a aba,
        # para não pagar a leitura dos runs na construção da interface.
        aba.select(
            _refresh_models,
            outputs=[models_df, model_pick, detail_html, hp_df, convergence_plot],
            show_progress="hidden",
        )
        aba.select(
            _show_run,
            inputs=[run_pick],
            outputs=[results_df, eer_plot, robust_plot, run_model_pick, run_note],
            show_progress="hidden",
        )
    return aba
