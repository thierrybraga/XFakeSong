"""Check generated MkDocs HTML for missing local files and HTML fragments."""

from __future__ import annotations

import argparse
import fnmatch
import posixpath
import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


class HtmlIndex(HTMLParser):
    """Collect local links and fragment targets from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.fragments: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        anchor_name = attributes.get("name") if tag == "a" else None
        if element_id:
            self.fragments.add(element_id)
        if anchor_name:
            self.fragments.add(anchor_name)
        if tag in {"a", "area", "link"}:
            href = attributes.get("href")
            if href:
                self.links.append(href)


def site_base_path(config: Path) -> tuple[str, str | None]:
    """Read the Pages path prefix and host from the MkDocs site_url."""
    if not config.exists():
        return "/", None
    match = re.search(
        r"(?m)^site_url:\s*[\"']?([^\s\"']+)",
        config.read_text(encoding="utf-8"),
    )
    if not match:
        return "/", None
    parsed = urlsplit(match.group(1))
    base = unquote(parsed.path or "/")
    return (base if base.endswith("/") else f"{base}/"), parsed.netloc or None


def excluded_markdown(config: Path) -> list[str]:
    """Read the simple block-list patterns from MkDocs exclude_docs."""
    if not config.exists():
        return []
    content = config.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^exclude_docs:\s*\|\s*\n((?:^[ \t]+.*\n?)*)", content)
    if not match:
        return []
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def expected_doc_page(source: Path, docs_root: Path, site_root: Path) -> Path:
    """Map a Markdown source to its MkDocs directory-URL output page."""
    relative = source.relative_to(docs_root)
    parent = site_root.joinpath(*relative.parent.parts)
    if relative.stem == "index":
        return parent / "index.html"
    return parent / relative.stem / "index.html"


def is_excluded(relative: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(relative, pattern)
        or (
            "/" not in pattern
            and fnmatch.fnmatchcase(PurePosixPath(relative).name, pattern)
        )
        for pattern in patterns
    )


def target_for_href(
    href: str, source: Path, site_root: Path, base_path: str, host: str | None
) -> tuple[Path | None, str]:
    """Resolve a local link; external links return no target."""
    parsed = urlsplit(href.strip())
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return None, unquote(parsed.fragment)
    if parsed.netloc and parsed.netloc != host:
        return None, unquote(parsed.fragment)

    path = unquote(parsed.path)
    if path.startswith("/"):
        if base_path != "/" and (path + "/").startswith(base_path):
            path = path[len(base_path) :]
        else:
            path = path.lstrip("/")
        relative = PurePosixPath(path)
    elif path:
        relative = PurePosixPath(posixpath.normpath((source.parent / path).as_posix()))
        try:
            relative = relative.relative_to(PurePosixPath(site_root.as_posix()))
        except ValueError:
            return site_root.parent, unquote(parsed.fragment)
    else:
        relative = PurePosixPath(source.relative_to(site_root).as_posix())

    target = site_root.joinpath(*relative.parts)
    if path.endswith("/") or not target.suffix:
        target = target / "index.html"
    return target, unquote(parsed.fragment)


def check_site(site_root: Path) -> list[str]:
    site_root = site_root.resolve()
    project_root = next(
        (
            parent
            for parent in site_root.parents
            if (parent / "mkdocs.yml").is_file() and (parent / "docs").is_dir()
        ),
        None,
    )
    if project_root is None:
        return [
            f"raiz do projeto não encontrada acima de {site_root}; "
            "esperava mkdocs.yml e docs/"
        ]
    config = project_root / "mkdocs.yml"
    docs_root = project_root / "docs"
    base_path, host = site_base_path(config)
    html_files = sorted(site_root.rglob("*.html"))
    indexes: dict[Path, HtmlIndex] = {}
    errors: list[str] = []

    for source in sorted(docs_root.rglob("*.md")):
        relative = source.relative_to(docs_root).as_posix()
        if is_excluded(relative, excluded_markdown(config)):
            continue
        output = expected_doc_page(source, docs_root, site_root)
        if not output.is_file():
            errors.append(
                f"página Markdown não gerada: {relative} -> "
                f"{output.relative_to(site_root)}"
            )

    for source in html_files:
        parser = HtmlIndex()
        parser.feed(source.read_text(encoding="utf-8"))
        indexes[source.resolve()] = parser

    for source, parser in indexes.items():
        for href in parser.links:
            target, fragment = target_for_href(href, source, site_root, base_path, host)
            if target is None:
                continue
            target = target.resolve()
            try:
                target.relative_to(site_root)
            except ValueError:
                errors.append(
                    f"{source.relative_to(site_root)}: link fora do site: {href}"
                )
                continue
            if not target.is_file():
                errors.append(
                    f"{source.relative_to(site_root)}: destino ausente: {href}"
                )
                continue
            if fragment and target.suffix.lower() == ".html":
                target_index = indexes.get(target)
                if target_index is None:
                    target_index = HtmlIndex()
                    target_index.feed(target.read_text(encoding="utf-8"))
                if fragment not in target_index.fragments:
                    errors.append(
                        f"{source.relative_to(site_root)}: fragmento ausente em "
                        f"{target.relative_to(site_root)}: #{fragment}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site", nargs="?", type=Path, default=Path("infrastructure/site")
    )
    args = parser.parse_args()
    if not args.site.is_dir():
        print(f"Diretório gerado não encontrado: {args.site}", file=sys.stderr)
        return 2
    errors = check_site(args.site)
    if errors:
        print(f"Links locais inválidos: {len(errors)}")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Links locais e fragmentos válidos em {args.site}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
