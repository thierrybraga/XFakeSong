from app.core.utils.helpers import (
    safe_filename, get_file_hash, format_file_size, format_duration
)


def test_safe_filename():
    assert safe_filename("test.txt") == "test.txt"
    assert safe_filename("test/file.txt") == "test_file.txt"
    assert safe_filename("test:file.txt") == "test_file.txt"
    # Test length truncation (simplified check)
    long_name = "a" * 300 + ".txt"
    assert len(safe_filename(long_name)) <= 255


def test_get_file_hash(tmp_path):
    f = tmp_path / "hash_test.txt"
    f.write_text("content")

    hash_val = get_file_hash(f)
    assert len(hash_val) == 32  # MD5 length in hex


def test_format_file_size():
    assert format_file_size(0) == "0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024) == "1.0 MB"


def test_format_duration():
    assert format_duration(30) == "30.0s"
    assert format_duration(60) == "1.0m"
    assert format_duration(3600) == "1.0h"
