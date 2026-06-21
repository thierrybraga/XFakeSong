#!/usr/bin/env python3
"""Synchronize trained XFakeSong models from a Hugging Face model repository.

The script is intentionally conservative for Spaces:
- no-op when MODEL_REPO_ID/XFAKE_MODEL_REPO_ID is not configured;
- skips download when the expected benchmark artifacts already exist;
- copies only model/runtime artifacts into app/models by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


EXPECTED_ARTIFACTS = (
    "bench_aasist.keras",
    "bench_conformer.keras",
    "bench_efficientnet_lstm.keras",
    "bench_ensemble.keras",
    "bench_hubert_original.pt",
    "bench_hybrid_cnn_transformer.keras",
    "bench_multiscalecnn.keras",
    "bench_randomforest.pkl",
    "bench_rawgat_st.keras",
    "bench_rawnet2.keras",
    "bench_sonic_sleuth.keras",
    "bench_spectrogramtransformer.keras",
    "bench_svm.pkl",
    "bench_wavlm_original.pt",
)


ALLOW_PATTERNS = (
    "bench_*",
    "benchmark_final_manifest.json",
    "wavlm_backbone/*",
    "hubert_backbone/*",
    "benchmark_final/**",
)


def _configured_repo(cli_repo: str | None) -> str | None:
    return (
        cli_repo
        or os.getenv("MODEL_REPO_ID")
        or os.getenv("XFAKE_MODEL_REPO_ID")
        or os.getenv("DEEPFAKE_MODEL_REPO_ID")
    )


def _missing_artifacts(models_dir: Path) -> list[str]:
    return [name for name in EXPECTED_ARTIFACTS if not (models_dir / name).exists()]


def _copy_tree_contents(src: Path, dst: Path) -> int:
    copied = 0
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    return copied


def sync_models(repo_id: str | None, models_dir: Path, force: bool = False) -> int:
    repo_id = _configured_repo(repo_id)
    models_dir.mkdir(parents=True, exist_ok=True)

    missing = _missing_artifacts(models_dir)
    if not repo_id:
        print("[sync_hf_models] MODEL_REPO_ID não definido; usando modelos locais.")
        if missing:
            print(f"[sync_hf_models] Artefatos ausentes localmente: {missing}")
        return 0

    if not force and not missing:
        print("[sync_hf_models] Modelos já presentes; download ignorado.")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "huggingface_hub é necessário para sincronizar modelos do Hub."
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    print(f"[sync_hf_models] Baixando modelos de {repo_id}...")
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            allow_patterns=list(ALLOW_PATTERNS),
            local_dir=os.getenv("XFAKE_MODEL_CACHE_DIR") or None,
        )
    )
    copied = _copy_tree_contents(snapshot_path, models_dir)
    missing_after = _missing_artifacts(models_dir)
    print(f"[sync_hf_models] Arquivos copiados: {copied}")
    if missing_after:
        raise RuntimeError(
            "Sincronização incompleta; artefatos ausentes: "
            + ", ".join(missing_after)
        )
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--models-dir",
        default=os.getenv("DEEPFAKE_MODELS_DIR")
        or os.getenv("MODELS_DIR")
        or "app/models",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sync_models(args.repo_id, Path(args.models_dir), force=args.force)


if __name__ == "__main__":
    main()
