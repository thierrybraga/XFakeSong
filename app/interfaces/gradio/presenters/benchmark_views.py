"""HTML da aba Benchmark: artigo, protocolo, detalhe de modelo e execução.

Funções puras: recebem o que o domínio (``app/domain/services/benchmark``)
devolve e produzem HTML com escape, sem Gradio nem serviço.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, List, Mapping, Optional

PROTOCOL_ROWS = (
    ("Corpus", "CETUC (voz real) × clones XTTS-v2, pareados enunciado a enunciado"),
    ("Recorte", "15.000 amostras balanceadas (benchmark_dataset_15k.npz)"),
    (
        "Partições",
        "12.162 / 1.456 / 1.382, disjuntas por locutor e por frase; teste selado",
    ),
    ("Entrada", "janela mono de 3 s a 16 kHz"),
    (
        "Correção de banda",
        "passa-baixas de 7,5 kHz, remoção de DC e RMS −26 dBFS nas duas classes",
    ),
    (
        "Ruído",
        "AWGN a 30/20/10 dB no treino (uma cópia); avaliação também a 5 dB, fora do treino",
    ),
    (
        "Seleção de época",
        "menor EER de validação nas neurais Keras; menor perda nas SSL",
    ),
    ("Orçamento", "100 épocas, semente 42, lote do plano por arquitetura"),
    ("Entradas", "11: 2 clássicas, 7 neurais Keras e 2 SSL congeladas"),
)

LIMITATIONS = (
    "Um corpus e um gerador: o estudo é dentro do domínio, sem prova de "
    "generalização a outros geradores, idiomas ou gravações de campo.",
    "Uma semente por arquitetura e uma realização de ruído por condição: onde os "
    "IC 95% se cruzam, os modelos são indistinguíveis.",
    "SVM e Random Forest são análises exploratórias: o front-end tabular foi "
    "revisto depois de uma observação do teste.",
)


def pct(valor: Any, casas: int = 2) -> str:
    """0,0116 -> `1,16%`; ausente -> `—`."""
    try:
        return f"{100 * float(valor):.{casas}f}".replace(".", ",") + "%"
    except (TypeError, ValueError):
        return "—"


def num(valor: Any, casas: int = 3) -> str:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return "—"
    if abs(numero) >= 1e5 and numero == int(numero):
        return f"{int(numero):,}".replace(",", ".")
    texto = f"{numero:.{casas}g}" if abs(numero) < 1e-3 else f"{numero:.{casas}f}"
    return texto.replace(".", ",")


def lr(valor: Any) -> str:
    """Taxa de aprendizado sem arredondar: 0,001 -> `0,001`; 1e-05 -> `1e-05`."""
    try:
        return f"{float(valor):g}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def _rows(pares: Iterable[tuple]) -> str:
    return "".join(
        f"<div><dt>{escape(str(k))}</dt><dd>{escape(str(v))}</dd></div>"
        for k, v in pares
    )


def article_html(info: Any) -> str:
    """Cabeçalho do artigo: título, autoria, resumo e o aviso do PDF."""
    if not getattr(info, "title", None) and not getattr(info, "abstract", None):
        return (
            '<div class="empty-state">O artigo não foi encontrado em '
            "<code>data/results/paper/main.tex</code>.</div>"
        )
    aviso = ""
    if info.pdf_is_stale:
        aviso = (
            '<p class="bench-warn">⚠ O PDF é de '
            f"{escape(str(info.pdf_modified or '?'))}, anterior à fonte "
            f"({escape(str(info.tex_modified or '?'))}). Recompile o LaTeX para "
            "ver a versão atual.</p>"
        )
    elif not info.pdf_path:
        aviso = '<p class="bench-warn">⚠ Não há PDF compilado do artigo.</p>'
    palavras = (
        f'<p class="bench-keywords"><strong>Palavras-chave:</strong> '
        f"{escape(info.keywords)}</p>"
        if info.keywords
        else ""
    )
    return (
        '<section class="bench-article">'
        f"<h3>{escape(info.title or 'Artigo do TCC')}</h3>"
        f'<p class="bench-author">{escape(info.author or "")}</p>'
        f'<p class="bench-abstract">{escape(info.abstract or "")}</p>'
        f"{palavras}{aviso}"
        "</section>"
    )


def protocol_html() -> str:
    limites = "".join(f"<li>{escape(t)}</li>" for t in LIMITATIONS)
    return (
        '<div class="bench-grid">'
        '<section class="bench-card"><h4>Protocolo do artigo</h4>'
        f'<dl class="bench-dl">{_rows(PROTOCOL_ROWS)}</dl></section>'
        '<section class="bench-card"><h4>Limitações declaradas</h4>'
        f"<ul>{limites}</ul>"
        '<p class="dim">Detalhes em <code>docs/evaluation/'
        "canonical-study-and-legacy-inventory.md</code>.</p></section>"
        "</div>"
    )


def model_detail_html(modelo: Any, plano: Optional[Mapping[str, Any]] = None) -> str:
    """Calibração, entrada, métricas e proveniência de um modelo instalado."""
    if modelo is None:
        return '<div class="empty-state">Escolha um modelo instalado.</div>'
    cal = modelo.calibration
    agregacao = cal.get("aggregation_method")
    calibracao = [
        ("Limiar de decisão (EER)", num(cal.get("eer_threshold"), 4)),
        ("EER de validação", pct(cal.get("eer_value"))),
        ("Origem do limiar", cal.get("threshold_source") or "validação do run"),
        ("Temperatura", num(cal.get("temperature"), 3)),
        (
            "Temperatura sobre",
            "logits" if cal.get("temperature_on_logits") else "probabilidades",
        ),
        ("Limiar de OOD", num(cal.get("ood_threshold"), 4)),
        (
            "Correção de banda",
            (
                f"{num(cal.get('band_cutoff_hz'), 0)} Hz"
                if cal.get("band_cutoff_hz")
                else "não declarada"
            ),
        ),
        (
            "Agregação de janelas",
            (
                f"{agregacao} (deslocamento {num(cal.get('aggregation_offset'), 3)})"
                if agregacao
                else "padrão (média, sem calibração)"
            ),
        ),
    ]
    entrada = modelo.input
    formato = [
        ("Front-end", entrada.get("feature_frontend") or "—"),
        ("Forma de entrada", str(entrada.get("input_shape") or "—")),
        ("Taxa", f"{entrada.get('sample_rate') or '—'} Hz"),
        ("Recorte", entrada.get("crop_strategy") or "—"),
    ]
    teste = modelo.test
    metricas = [
        (
            "EER de teste",
            (
                f"{pct(teste.get('eer'))} [{pct(teste.get('eer_ci95_low'))}–"
                f"{pct(teste.get('eer_ci95_high'))}]"
                if teste
                else "sem run ligado a este artefato"
            ),
        ),
        ("AUC de teste", num(teste.get("auc_roc"), 4) if teste else "—"),
        ("Origem das métricas de teste", teste.get("source") or "—"),
        ("Acurácia de validação", pct(modelo.validation.get("accuracy"))),
        ("Parâmetros", num(modelo.efficiency.get("params"), 0)),
        ("Tamanho", f"{num(modelo.size_mb, 1)} MB"),
        ("Latência", f"{num(modelo.efficiency.get('latency_ms'), 1)} ms"),
    ]
    prov = modelo.provenance
    origem = [
        ("Criado em", prov.get("created_at") or "—"),
        ("Recalibrado em", prov.get("recalibrated_at") or "—"),
        ("Treinado por", prov.get("trained_by") or "benchmark"),
        ("Arquivo", modelo.path),
    ]
    padrao = (
        ' <span class="bench-tag">padrão da detecção</span>'
        if modelo.is_default
        else ""
    )
    return (
        f'<div class="bench-detail"><h4>{escape(modelo.display_name)}{padrao}</h4>'
        '<div class="bench-grid">'
        f'<section class="bench-card"><h5>Calibração</h5><dl class="bench-dl">{_rows(calibracao)}</dl></section>'
        f'<section class="bench-card"><h5>Métricas</h5><dl class="bench-dl">{_rows(metricas)}</dl></section>'
        f'<section class="bench-card"><h5>Entrada</h5><dl class="bench-dl">{_rows(formato)}</dl></section>'
        f'<section class="bench-card"><h5>Proveniência</h5><dl class="bench-dl">{_rows(origem)}</dl></section>'
        "</div></div>"
    )


def preflight_html(checks: List[Any], ready: bool) -> str:
    itens = "".join(
        f'<li class="{"ok" if c.ok else "fail"}"><span aria-hidden="true">'
        f'{"✓" if c.ok else "✗"}</span> <strong>{escape(c.name)}</strong>: '
        f"{escape(c.detail)}</li>"
        for c in checks
    )
    resumo = (
        '<p class="bench-ready">Tudo pronto para iniciar o run do artigo.</p>'
        if ready
        else '<p class="bench-warn">O run não pode ser iniciado daqui enquanto '
        "houver item marcado com ✗.</p>"
    )
    return f'<div class="bench-preflight"><ul>{itens}</ul>{resumo}</div>'


_STATUS_LABEL = {
    "ok": "concluído",
    "error": "erro",
    "timeout": "tempo esgotado",
    "running": "em andamento",
    "partial": "parcial",
}


def run_status_html(status: Any) -> str:
    """Estado de um run: o modelo em curso, o progresso e a lista de concluídos."""
    if status is None or not getattr(status, "out_dir", None):
        return '<div class="empty-state">Nenhum run iniciado pela aba.</div>'
    estado = (
        "em andamento"
        if status.active
        else _STATUS_LABEL.get(str(status.summary_status), "parado")
    )
    atual = ""
    if status.current is not None:
        c = status.current
        progresso = (
            f"época {c.epoch}/{c.epochs}" if c.epoch and c.epochs else "preparando"
        )
        eta = f", ~{num(c.eta_min, 0)} min para o fim do modelo" if c.eta_min else ""
        barra = (
            f'<progress max="{c.epochs}" value="{c.epoch}"></progress>'
            if c.epoch and c.epochs
            else ""
        )
        atual = (
            f'<p class="bench-current">Treinando <strong>{escape(c.model)}</strong>: '
            f"{escape(progresso)}{escape(eta)}</p>{barra}"
        )
    linhas = "".join(
        f"<li>{escape(m.model)}: {escape(_STATUS_LABEL.get(m.status, m.status))}"
        + (f" ({num((m.elapsed_s or 0) / 3600, 2)} h)" if m.elapsed_s else "")
        + "</li>"
        for m in status.models
    )
    concluidos = (
        f"<h5>Modelos concluídos ({len(status.models)}/11)</h5><ul>{linhas}</ul>"
        if linhas
        else "<p class='dim'>Nenhum modelo concluído ainda.</p>"
    )
    return (
        '<div class="bench-status" role="status" aria-live="polite">'
        f"<p><strong>{escape(str(status.run_name))}</strong>: {escape(estado)}</p>"
        f"{atual}{concluidos}"
        f'<p class="dim">Pasta: <code>{escape(str(status.out_dir))}</code></p>'
        "</div>"
    )
