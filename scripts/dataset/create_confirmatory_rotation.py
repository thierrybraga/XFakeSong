#!/usr/bin/env python3
"""Cria split confirmatório sem reutilizar o teste legado como novo teste.

O novo teste e a nova validação são selecionados exclusivamente do antigo
TREINO, estratificados por classe+fonte. O antigo teste entra apenas no novo
treino. O script não treina nem avalia modelos e recusa sobrescrita.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values.astype(str), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    if not source.exists():
        parser.error(f"fonte não encontrada: {source}")
    if out.exists():
        parser.error(f"saída já existe e não será sobrescrita: {out}")
    lock = out.with_suffix(out.suffix + ".test-lock.json")
    if lock.exists():
        parser.error(f"selo já existe para a saída: {lock}")

    with np.load(source, allow_pickle=False) as data:
        required = {
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
            "groups", "speaker_ids", "metadata_json",
        }
        if not required.issubset(data.files):
            parser.error(f"NPZ fonte sem chaves: {sorted(required - set(data.files))}")

        y_old_train = np.asarray(data["y_train"], dtype="int64")
        y_old_val = np.asarray(data["y_val"], dtype="int64")
        y_old_test = np.asarray(data["y_test"], dtype="int64")
        n_old_train, n_old_val, n_old_test = map(
            len, (y_old_train, y_old_val, y_old_test)
        )
        groups_all = np.asarray(data["groups"]).astype(str)
        speakers_all = np.asarray(data["speaker_ids"]).astype(str)
        if len(groups_all) != n_old_train + n_old_val + n_old_test:
            parser.error("groups desalinhado")
        groups_old_train = groups_all[:n_old_train]
        composite = np.char.add(
            np.char.add(y_old_train.astype(str), "::"), groups_old_train
        )
        old_train_idx = np.arange(n_old_train, dtype="int64")
        remaining_idx, new_test_idx = train_test_split(
            old_train_idx,
            test_size=n_old_test,
            random_state=args.seed,
            stratify=composite,
        )
        remaining_composite = composite[remaining_idx]
        new_train_old_part_idx, new_val_idx = train_test_split(
            remaining_idx,
            test_size=n_old_val,
            random_state=args.seed + 1,
            stratify=remaining_composite,
        )
        if set(new_test_idx) & set(new_val_idx):
            raise RuntimeError("sobreposição interna test/val")

        meta = json.loads(str(data["metadata_json"].item()))
        old_paths = []
        for split in ("train", "val", "test"):
            old_paths.extend(meta["splits"][split]["paths"])
        old_paths = np.asarray(old_paths, dtype=str)

        old_val_global = np.arange(
            n_old_train, n_old_train + n_old_val, dtype="int64"
        )
        old_test_global = np.arange(
            n_old_train + n_old_val,
            n_old_train + n_old_val + n_old_test,
            dtype="int64",
        )
        new_train_global = np.concatenate(
            [new_train_old_part_idx, old_val_global, old_test_global]
        )
        new_val_global = np.asarray(new_val_idx, dtype="int64")
        new_test_global = np.asarray(new_test_idx, dtype="int64")

        # Carrega os tensores somente após fixar todos os índices.
        X_old_train = np.asarray(data["X_train"], dtype="float32")
        X_old_val = np.asarray(data["X_val"], dtype="float32")
        X_old_test = np.asarray(data["X_test"], dtype="float32")
        X_new_train = np.concatenate(
            [X_old_train[new_train_old_part_idx], X_old_val, X_old_test], axis=0
        )
        X_new_val = X_old_train[new_val_idx]
        X_new_test = X_old_train[new_test_idx]
        y_new_train = np.concatenate(
            [y_old_train[new_train_old_part_idx], y_old_val, y_old_test]
        )
        y_new_val = y_old_train[new_val_idx]
        y_new_test = y_old_train[new_test_idx]

        split_indices = {
            "train": new_train_global,
            "val": new_val_global,
            "test": new_test_global,
        }
        new_meta = dict(meta)
        new_meta.update(
            {
                "source": str(source),
                "dataset_version": "confirmatory-v2",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "split_strategy": "confirmatory_rotation_from_legacy_train",
                "rotation_seed": int(args.seed),
                "test_prior_role": "legacy_train_only",
                "old_test_new_role": "train_only",
                "limitation": (
                    "Teste confirmatório interno: novo teste não foi avaliação "
                    "legada, mas deriva do mesmo corpus consolidado."
                ),
            }
        )
        new_meta["splits"] = {}
        for split, idx, y_split in (
            ("train", new_train_global, y_new_train),
            ("val", new_val_global, y_new_val),
            ("test", new_test_global, y_new_test),
        ):
            paths = old_paths[idx].tolist()
            g = groups_all[idx]
            new_meta["splits"][split] = {
                "samples": int(len(idx)),
                "real": int(np.sum(y_split == 0)),
                "fake": int(np.sum(y_split == 1)),
                "paths": paths,
                "source_counts": _counts(g),
            }

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            X_train=X_new_train,
            y_train=y_new_train,
            X_val=X_new_val,
            y_val=y_new_val,
            X_test=X_new_test,
            y_test=y_new_test,
            groups=np.concatenate(
                [groups_all[new_train_global], groups_all[new_val_global], groups_all[new_test_global]]
            ),
            speaker_ids=np.concatenate(
                [speakers_all[new_train_global], speakers_all[new_val_global], speakers_all[new_test_global]]
            ),
            metadata_json=np.asarray(json.dumps(new_meta, ensure_ascii=False)),
        )

    manifest = {
        "protocol_version": "xfakesong-confirmatory-rotation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(out),
        "output_sha256": _sha256(out),
        "seed": int(args.seed),
        "selection_strata": "class_label::source_group",
        "new_test_prior_role": "legacy_train_only",
        "old_test_new_role": "train_only",
        "counts": {
            "train": int(len(new_train_global)),
            "val": int(len(new_val_global)),
            "test": int(len(new_test_global)),
        },
        "indices_in_legacy_concatenation": {
            key: value.astype(int).tolist() for key, value in split_indices.items()
        },
    }
    manifest_path = out.with_suffix(out.suffix + ".rotation.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"NPZ confirmatório: {out}")
    print(f"Manifesto: {manifest_path}")
    print(f"SHA-256: {manifest['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())