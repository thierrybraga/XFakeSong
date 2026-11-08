"""O artigo do TCC na aplicação: título, autoria, resumo e o PDF.

A fonte é ``data/results/paper/main.tex``. O título e o autor saem do
``\\hypersetup`` (``pdftitle``/``pdfauthor``) e o resumo do bloco ``RESUMO``
(NBR 6028). A conversão para texto cobre só o que o resumo usa (``\\num``,
``\\SI``, ``\\textit``...), não LaTeX em geral.

O ``main.pdf`` é compilado à parte e pode estar defasado: ``pdf_is_stale``
compara as datas para a tela avisar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_UNIDADES = {
    r"\second": "s",
    r"\kilo\hertz": "kHz",
    r"\hertz": "Hz",
    r"\decibel": "dB",
    r"\milli\second": "ms",
}


@dataclass
class ArticleInfo:
    title: Optional[str]
    author: Optional[str]
    abstract: Optional[str]
    keywords: Optional[str]
    tex_path: Optional[str]
    pdf_path: Optional[str]
    tex_modified: Optional[str]
    pdf_modified: Optional[str]
    pdf_is_stale: bool


def latex_to_text(fonte: str) -> str:
    """Texto legível de um trecho de LaTeX simples (o do resumo)."""
    texto = re.sub(r"(?<!\\)%.*", "", fonte)
    texto = re.sub(r"\\sisetup\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", texto)
    texto = re.sub(
        r"\\SI\{([^{}]*)\}\{([^{}]*)\}",
        lambda m: f"{m.group(1)} {_UNIDADES.get(m.group(2).strip(), m.group(2))}",
        texto,
    )
    texto = re.sub(r"\\num\{([^{}]*)\}", r"\1", texto)
    texto = re.sub(r"\\(?:textit|emph|textbf|texttt|mbox)\{([^{}]*)\}", r"\1", texto)
    texto = re.sub(r"\$([^$]*)\$", r"\1", texto)
    texto = texto.replace("^\\ast", "*").replace("\\ast", "*")
    texto = texto.replace("\\%", "%").replace("~", " ").replace("--", "–")
    texto = re.sub(r"\\[vh]space\*?\{[^}]*\}", " ", texto)
    texto = re.sub(r"\\(?:noindent|par)\b", " ", texto)
    texto = re.sub(r"\\[a-zA-Z]+\*?", "", texto)
    texto = texto.replace("{", "").replace("}", "").replace("^", "")
    return re.sub(r"\s+", " ", texto).strip()


def _campo_hypersetup(fonte: str, chave: str) -> Optional[str]:
    achado = re.search(rf"{chave}\s*=\s*\{{([^}}]*)\}}", fonte)
    return achado.group(1).strip() if achado else None


def _resumo(fonte: str) -> tuple[Optional[str], Optional[str]]:
    achado = re.search(
        r"\\textbf\{RESUMO\}.*?\\begin\{singlespace\}(.*?)\\end\{singlespace\}",
        fonte,
        flags=re.S,
    )
    if not achado:
        return None, None
    bloco = achado.group(1)
    palavras = None
    partes = re.split(r"\\textbf\{Palavras-chave:\}", bloco)
    if len(partes) == 2:
        bloco, palavras = partes
    return latex_to_text(bloco), (latex_to_text(palavras) if palavras else None)


def _data(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def article_info(paper_dir: Path) -> ArticleInfo:
    """O que a aba mostra do artigo; campos ausentes ficam ``None``."""
    pasta = Path(paper_dir)
    tex, pdf = pasta / "main.tex", pasta / "main.pdf"
    fonte = ""
    if tex.is_file():
        try:
            fonte = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            fonte = ""
    resumo, palavras = _resumo(fonte) if fonte else (None, None)
    data_tex, data_pdf = _data(tex), _data(pdf)
    return ArticleInfo(
        title=_campo_hypersetup(fonte, "pdftitle") if fonte else None,
        author=_campo_hypersetup(fonte, "pdfauthor") if fonte else None,
        abstract=resumo,
        keywords=palavras,
        tex_path=str(tex) if tex.is_file() else None,
        pdf_path=str(pdf) if pdf.is_file() else None,
        tex_modified=data_tex.isoformat(timespec="minutes") if data_tex else None,
        pdf_modified=data_pdf.isoformat(timespec="minutes") if data_pdf else None,
        pdf_is_stale=bool(data_tex and data_pdf and data_pdf < data_tex),
    )
