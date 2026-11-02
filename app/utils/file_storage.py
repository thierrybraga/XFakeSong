"""Operações de arquivo sem dependências de runtime da aplicação."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional, Union


def break_hardlink(file_path: Union[str, Path]) -> bool:
    """Desassocia um arquivo antes de uma gravação in-place."""
    path = Path(file_path)
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if path.stat().st_nlink <= 1:
            return False
    except OSError:
        return False

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(path, temporary)
        os.replace(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def break_hardlinks_in_tree(
    directory: Union[str, Path],
    *,
    exclude: Optional[Callable[[Path], bool]] = None,
) -> int:
    """Desassocia hardlinks de uma árvore antes de regravar suas saídas."""
    root = Path(directory)
    if not root.is_dir():
        return 0
    detached = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if exclude is not None and exclude(path):
            continue
        detached += int(break_hardlink(path))
    return detached


def link_or_copy_file(
    source: Union[str, Path], destination: Union[str, Path]
) -> str:
    """Publica um hardlink quando possível, com cópia segura como fallback."""
    src = Path(source)
    dst = Path(destination)
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "hardlink"
        except OSError:
            pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
    method = "hardlink"
    try:
        try:
            os.link(src, temporary)
        except OSError:
            method = "copy"
            shutil.copy2(src, temporary)
        os.replace(temporary, dst)
        return method
    finally:
        temporary.unlink(missing_ok=True)


def copy_file_atomic(
    source: Union[str, Path], destination: Union[str, Path]
) -> None:
    """Copia para um arquivo INDEPENDENTE do original (nunca hardlink).

    Para backups: um hardlink compartilharia o inode, e qualquer escrita
    in-place no original alteraria também o "backup".
    """
    src = Path(source)
    dst = Path(destination)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    file_path: Union[str, Path],
    text: str,
    *,
    encoding: str = "utf-8",
    newline: Optional[str] = None,
) -> None:
    """Grava texto em temporário e substitui o destino sem alterar hardlinks."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding=encoding, newline=newline)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
