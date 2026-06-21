"""Upload trained XFakeSong artifacts to Hugging Face Hub.

The script is intentionally conservative: it uploads the consolidated model
folder by default, supports a dry-run without credentials, and never prints the
token. Use HF_TOKEN or HUGGINGFACE_HUB_TOKEN for authenticated uploads.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "app" / "models"

DEFAULT_MODEL_PATTERNS = [
    "*.keras",
    "*.pkl",
    "*.json",
    "*.md",
    "*.pt",
    "*.bin",
    "*.safetensors",
    "*.onnx",
    "benchmark_final/**",
]

DEFAULT_IGNORE_PATTERNS = [
    "__pycache__/**",
    ".ipynb_checkpoints/**",
    "*.aux",
    "*.log",
    "*.out",
    "*.toc",
    "*.pdf",
    "*.tmp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload trained XFakeSong models to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository id, for example: usuario/xfakesong-models",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Hub repository type. Use 'model' for trained model artifacts.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Local folder to upload. Defaults to app/models.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="models",
        help="Destination folder inside the Hub repository.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or revision to upload to.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repository as private if it does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be uploaded. No token required.",
    )
    parser.add_argument(
        "--include-overleaf",
        action="store_true",
        help="Also upload tcc_overleaf/ to tcc_overleaf/.",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Also upload curated benchmark reports and figures from results/.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help=(
            "Additional allow pattern passed to huggingface_hub.upload_folder. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help=(
            "Additional ignore pattern passed to huggingface_hub.upload_folder. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="Upload XFakeSong trained models",
        help="Commit message used by Hugging Face Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Prefer HF_TOKEN env var instead.",
    )
    return parser.parse_args()


def resolve_inside_project(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(
            f"Refusing to upload path outside project root: {resolved}"
        ) from exc
    return resolved


def should_ignore(relative_path: Path) -> bool:
    text = relative_path.as_posix()
    return (
        "__pycache__/" in text
        or ".ipynb_checkpoints/" in text
        or relative_path.suffix.lower()
        in {".aux", ".log", ".out", ".toc", ".pdf", ".tmp"}
    )


def collect_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and not should_ignore(path.relative_to(folder))
    )


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def print_plan(label: str, source: Path, path_in_repo: str) -> None:
    files = collect_files(source)
    total = sum(path.stat().st_size for path in files)
    print(f"\n[{label}] {source} -> {path_in_repo}")
    print(f"Arquivos: {len(files)} | Tamanho: {human_size(total)}")
    for path in files[:30]:
        print(f"  - {path.relative_to(source).as_posix()}")
    if len(files) > 30:
        print(f"  ... +{len(files) - 30} arquivos")


def token_from_args(args: argparse.Namespace) -> str | None:
    return (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def upload_folder(
    api: object,
    *,
    folder_path: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    commit_message: str,
    revision: str | None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
) -> None:
    kwargs = {
        "folder_path": str(folder_path),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "commit_message": commit_message,
        "ignore_patterns": list(ignore_patterns or DEFAULT_IGNORE_PATTERNS),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    if revision:
        kwargs["revision"] = revision
    api.upload_folder(**kwargs)


def upload_curated_results(
    api: object,
    args: argparse.Namespace,
    ignore_patterns: list[str],
) -> None:
    candidates = [
        PROJECT_ROOT / "results" / "tcc_consolidated",
        PROJECT_ROOT / "results" / "model_manifest.json",
        PROJECT_ROOT / "results" / "MODEL_AUDIT.md",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            upload_folder(
                api,
                folder_path=candidate,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo=f"results/{candidate.name}",
                commit_message=f"{args.commit_message}: results/{candidate.name}",
                revision=args.revision,
                allow_patterns=None,
                ignore_patterns=ignore_patterns,
            )
        elif candidate.is_file():
            path_in_repo = f"results/{candidate.name}"
            kwargs = {
                "path_or_fileobj": str(candidate),
                "path_in_repo": path_in_repo,
                "repo_id": args.repo_id,
                "repo_type": args.repo_type,
                "commit_message": f"{args.commit_message}: {path_in_repo}",
            }
            if args.revision:
                kwargs["revision"] = args.revision
            api.upload_file(**kwargs)


def main() -> None:
    args = parse_args()
    source = resolve_inside_project(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    allow_patterns = DEFAULT_MODEL_PATTERNS + list(args.allow_patterns or [])
    ignore_patterns = DEFAULT_IGNORE_PATTERNS + list(args.ignore_patterns or [])

    print_plan("modelos", source, args.path_in_repo)
    if args.include_overleaf:
        overleaf = PROJECT_ROOT / "tcc_overleaf"
        if overleaf.is_dir():
            print_plan("overleaf", overleaf, "tcc_overleaf")
    if args.include_results:
        results = PROJECT_ROOT / "results" / "tcc_consolidated"
        if results.is_dir():
            print_plan("resultados", results, "results/tcc_consolidated")

    if args.dry_run:
        print("\nDry-run concluido. Nenhum arquivo foi enviado.")
        return

    token = token_from_args(args)
    if not token:
        raise SystemExit(
            "Token ausente. Defina HF_TOKEN ou HUGGINGFACE_HUB_TOKEN, "
            "ou passe --token."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Instale a dependencia: pip install 'huggingface_hub>=0.25,<1.0'"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    upload_folder(
        api,
        folder_path=source,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        revision=args.revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    if args.include_overleaf:
        overleaf = PROJECT_ROOT / "tcc_overleaf"
        if overleaf.is_dir():
            upload_folder(
                api,
                folder_path=overleaf,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo="tcc_overleaf",
                commit_message=f"{args.commit_message}: tcc_overleaf",
                revision=args.revision,
                allow_patterns=None,
                ignore_patterns=ignore_patterns,
            )

    if args.include_results:
        upload_curated_results(api, args, ignore_patterns)

    print(f"\nUpload concluido: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
