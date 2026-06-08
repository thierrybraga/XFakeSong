"""Cobertura de `app/core/utils/file_utils.py`.

Operações de arquivo puras, exercitadas com `tmp_path` (sem tocar o ambiente
real): hashing, info, busca por extensão, cópia/movimentação segura, extração
de arquivos comprimidos, diretórios temporários, validação, manifesto,
verificação de integridade e renomeação em lote.
"""

import tarfile
import zipfile

import pytest

from app.core.utils import file_utils as fu


# ── básicos ───────────────────────────────────────────────────────────────

def test_ensure_directory(tmp_path):
    d = fu.ensure_directory(tmp_path / "x" / "y")
    assert d.is_dir()


def test_safe_filename_variants():
    assert fu.safe_filename("a/b:c.txt") == "abc.txt"
    assert fu.safe_filename("  ...  ") == "unnamed_file"
    assert fu.safe_filename("***") == "unnamed_file"
    long = "a" * 300 + ".txt"
    assert len(fu.safe_filename(long)) <= 255


def test_get_file_hash_and_verify_integrity(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"conteudo")
    h = fu.get_file_hash(f)
    assert fu.verify_file_integrity(f, h) is True
    assert fu.verify_file_integrity(f, "deadbeef") is False
    # arquivo inexistente → False (sem levantar)
    assert fu.verify_file_integrity(tmp_path / "missing", h) is False


# ── get_file_info ─────────────────────────────────────────────────────────

def test_get_file_info_ok(tmp_path):
    f = tmp_path / "song.wav"
    f.write_bytes(b"RIFF1234")
    info = fu.get_file_info(f)
    assert info["name"] == "song.wav"
    assert info["suffix"] == ".wav"
    assert info["size_bytes"] == 8
    assert info["is_file"] is True
    assert "md5_hash" in info and info["md5_hash"]


def test_get_file_info_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fu.get_file_info(tmp_path / "nope.wav")


# ── busca por extensão ────────────────────────────────────────────────────

def test_find_files_by_extension_recursive_and_flat(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.MP3").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.wav").write_bytes(b"x")

    # normaliza extensões com/sem ponto e caixa
    flat = fu.find_files_by_extension(tmp_path, ["wav", ".mp3"], recursive=False)
    assert {p.name for p in flat} == {"a.wav", "b.MP3"}

    rec = fu.find_files_by_extension(tmp_path, [".wav"], recursive=True)
    assert {p.name for p in rec} == {"a.wav", "c.wav"}


def test_find_audio_and_archive_helpers(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.zip").write_bytes(b"x")
    assert {p.name for p in fu.find_audio_files(tmp_path)} == {"a.wav"}
    assert {p.name for p in fu.find_archive_files(tmp_path)} == {"b.zip"}


# ── cópia / movimentação ──────────────────────────────────────────────────

def test_copy_file_safe(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"12345")
    dst = tmp_path / "out" / "dst.bin"
    assert fu.copy_file_safe(src, dst) is True
    assert dst.read_bytes() == b"12345"

    # destino existe sem overwrite → erro
    with pytest.raises(FileExistsError):
        fu.copy_file_safe(src, dst)
    # com overwrite → ok
    assert fu.copy_file_safe(src, dst, overwrite=True) is True


def test_copy_file_safe_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        fu.copy_file_safe(tmp_path / "ghost", tmp_path / "x")


def test_move_file_safe(tmp_path):
    src = tmp_path / "m.bin"
    src.write_bytes(b"abc")
    dst = tmp_path / "moved" / "m.bin"
    assert fu.move_file_safe(src, dst) is True
    assert not src.exists() and dst.exists()

    with pytest.raises(FileNotFoundError):
        fu.move_file_safe(tmp_path / "ghost", tmp_path / "z")


# ── extração de arquivos ──────────────────────────────────────────────────

def test_extract_zip(tmp_path):
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inside/file.txt", "hello")
    out = tmp_path / "unzipped"
    extracted = fu.extract_archive(archive, out)
    assert any(p.name == "file.txt" for p in extracted)
    assert (out / "inside" / "file.txt").read_text() == "hello"


def test_extract_tgz(tmp_path):
    src = tmp_path / "data.txt"
    src.write_text("payload")
    archive = tmp_path / "pack.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname="data.txt")
    out = tmp_path / "untar"
    extracted = fu.extract_archive(archive, out)
    assert any(p.name == "data.txt" for p in extracted)


def test_extract_unsupported_format(tmp_path):
    bad = tmp_path / "x.rar"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        fu.extract_archive(bad, tmp_path / "o")


def test_extract_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        fu.extract_archive(tmp_path / "ghost.zip", tmp_path / "o")


def test_extract_corrupted_zip(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"not a real zip")
    with pytest.raises(ValueError):
        fu.extract_archive(bad, tmp_path / "o")


def test_extract_zip_rejects_path_traversal(tmp_path):
    """Zip Slip: entrada com ../ deve ser rejeitada (não escrever fora do dest)."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(ValueError):
        fu.extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_tar_rejects_path_traversal(tmp_path):
    """CVE-2007-4559: membro de tar com ../ deve ser rejeitado."""
    import io

    archive = tmp_path / "evil.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"pwned"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        fu.extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


# ── diretórios temporários ────────────────────────────────────────────────

def test_temp_directory_lifecycle():
    d = fu.create_temp_directory(prefix="temp_xfake_")
    assert d.is_dir()
    assert "temp_xfake_" in d.name
    # remove porque o nome contém "temp"
    assert fu.cleanup_temp_directory(d) is True
    assert not d.exists()


def test_cleanup_temp_directory_requires_force_for_non_temp(tmp_path):
    plain = tmp_path / "mydata"
    plain.mkdir()
    # nome não contém "temp" e force=False → não remove
    assert fu.cleanup_temp_directory(plain) is False
    assert plain.exists()
    # force=True remove
    assert fu.cleanup_temp_directory(plain, force=True) is True
    assert not plain.exists()


# ── validação de tipo ─────────────────────────────────────────────────────

def test_validate_file_type_and_is_audio_archive(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    assert fu.validate_file_type(wav, ["wav", "mp3"]) is True
    assert fu.validate_file_type(wav, ["mp3"]) is False
    # check_content não rejeita um wav coerente
    assert fu.validate_file_type(wav, [".wav"], check_content=True) is True
    assert fu.is_audio_file(wav) is True
    assert fu.is_archive_file(wav) is False


# ── tamanho / manifesto ───────────────────────────────────────────────────

def test_get_directory_size(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"12345")
    (tmp_path / "b.bin").write_bytes(b"123")
    sub = tmp_path / "s"
    sub.mkdir()
    (sub / "c.bin").write_bytes(b"12")
    total, count = fu.get_directory_size(tmp_path)
    assert total == 10
    assert count == 3


def test_create_file_manifest_with_output(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world!!")
    out = tmp_path / "manifest.json"
    manifest = fu.create_file_manifest(tmp_path, output_file=out)
    assert manifest["file_count"] >= 2
    assert manifest["total_size"] > 0
    assert out.exists()


# ── renomeação em lote ────────────────────────────────────────────────────

def test_batch_rename_files(tmp_path):
    (tmp_path / "b.wav").write_bytes(b"x")
    (tmp_path / "a.wav").write_bytes(b"x")
    renamed = fu.batch_rename_files(tmp_path, extensions=[".wav"])
    assert len(renamed) == 2
    names = {p.name for p in tmp_path.iterdir()}
    # ordenado por nome: a→001, b→002
    assert "001_a.wav" in names
    assert "002_b.wav" in names
