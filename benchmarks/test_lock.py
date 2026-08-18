"""Selo do conjunto de teste: prova de que o teste não foi tocado no treino.

Extraído de `scripts/benchmark/run_models_sequential.py` em 2026-07-27. Estava
implementado dentro de UM orquestrador, então o entrypoint direto
(`scripts/benchmark/run_benchmark.py` — o documentado para `--full` e para
modelo isolado) gravava o SHA-256 da partição de teste nos resultados mas nunca
o conferia contra o selo. Um selo que só um dos caminhos verifica não é um selo.

O selo é gerado por `scripts/dataset/freeze_benchmark_test.py` ANTES de qualquer
treino e amarra três coisas: o SHA-256 do arquivo inteiro, uma identidade leve
da partição de teste (nomes + CRC32 + tamanho dos membros do ZIP, sem
descompactar tensores) e as declarações explícitas de que o teste estava
intocado e foi criado antes do treino.
"""

from __future__ import annotations

import ast
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "inspect_npz",
    "sha256_file",
    "validate_test_lock",
    "TestLockError",
]


class TestLockError(ValueError):
    """Selo ausente, malformado ou incompatível com o dataset apresentado."""


def _npy_shape(member) -> tuple[int, ...]:
    """Lê apenas o cabeçalho NPY dentro do NPZ, sem descompactar os tensores."""

    if member.read(6) != b"\x93NUMPY":
        raise ValueError("membro NPZ sem cabeçalho NPY válido")
    major, _minor = member.read(2)
    size_fmt = "<H" if major == 1 else "<I"
    size = struct.calcsize(size_fmt)
    header_len = struct.unpack(size_fmt, member.read(size))[0]
    header = ast.literal_eval(member.read(header_len).decode("latin1").strip())
    return tuple(int(v) for v in header["shape"])


def inspect_npz(path: Path) -> Dict[str, Any]:
    """Valida estrutura e cria identidade leve do teste congelado."""

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {
            "X_train.npy",
            "y_train.npy",
            "X_val.npy",
            "y_val.npy",
            "X_test.npy",
            "y_test.npy",
        }
        predefined = required.issubset(names)
        y_members = (
            ["y_train.npy", "y_val.npy", "y_test.npy"] if predefined else ["y.npy"]
        )
        if not all(name in names for name in y_members):
            raise ValueError("NPZ sem rótulos completos X/y ou train/val/test")
        counts = {}
        for name in y_members:
            with archive.open(name) as member:
                shape = _npy_shape(member)
            counts[name.removesuffix(".npy")] = int(shape[0])
        test_identity = None
        if predefined:
            parts = []
            identity_members = ["X_test.npy", "y_test.npy"]
            identity_members.extend(
                name
                for name in ("cluster_ids.npy", "source_ids.npy", "sample_paths.npy")
                if name in names
            )
            for name in identity_members:
                info = archive.getinfo(name)
                parts.append(f"{name}:{info.CRC:08x}:{info.file_size}")
            test_identity = hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()
    has_cluster_ids = "cluster_ids.npy" in names
    has_source_ids = "source_ids.npy" in names or "groups.npy" in names
    has_sample_paths = "sample_paths.npy" in names
    return {
        "predefined_splits": predefined,
        "split_counts": counts,
        "sample_count": int(sum(counts.values())),
        "test_archive_identity_sha256": test_identity,
        "test_archive_identity_method": (
            "sha256(zip_member_name_crc32_uncompressed_size)"
        ),
        "has_cluster_ids": has_cluster_ids,
        "has_source_ids": has_source_ids,
        "has_sample_paths": has_sample_paths,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_test_lock(
    dataset_path: Path,
    inspection: Dict[str, Any],
    lock_path: Path,
) -> Dict[str, Any]:
    """Confere o selo contra o dataset apresentado. Levanta se divergir."""

    dataset_path = Path(dataset_path)
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise TestLockError(
            f"selo do teste não encontrado: {lock_path}. Gere um novo teste intocado "
            "e execute scripts/dataset/freeze_benchmark_test.py antes do treino."
        )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    required_true = (
        payload.get("declared_untouched") is True
        and payload.get("created_before_training") is True
    )
    if not required_true:
        raise TestLockError("selo não declara teste intocado e criado antes do treino")
    if int(payload.get("dataset_size_bytes", -1)) != dataset_path.stat().st_size:
        raise TestLockError("dataset mudou de tamanho após o selo do teste")
    expected_test = inspection.get("test_archive_identity_sha256")
    if payload.get("test_archive_identity_sha256") != expected_test:
        raise TestLockError("partição de teste difere daquela registrada no selo")
    actual_dataset_sha256 = sha256_file(dataset_path)
    if payload.get("dataset_sha256") != actual_dataset_sha256:
        raise TestLockError("SHA-256 do dataset difere daquele registrado no selo")
    return {
        **payload,
        "lock_path": str(lock_path.resolve()),
        "validated": True,
    }


def validate_dataset_against_lock(
    dataset_path: str | Path, lock_path: str | Path
) -> Dict[str, Any]:
    """Atalho: inspeciona o NPZ e confere o selo em uma chamada."""

    dataset_path = Path(dataset_path)
    return validate_test_lock(dataset_path, inspect_npz(dataset_path), Path(lock_path))
