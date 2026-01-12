import pytest
from pathlib import Path
from app.domain.services.upload_service import AudioUploadService
from app.core.interfaces.base import ProcessingStatus, DatasetType


@pytest.fixture
def upload_service(tmp_path):
    return AudioUploadService(upload_directory=str(tmp_path / "uploads"))


@pytest.fixture
def sample_audio_file(tmp_path):
    # Create a dummy WAV file
    file_path = tmp_path / "test_audio.wav"
    with open(file_path, "wb") as f:
        # Minimal header to look like a file, but service only checks ext
        f.write(
            b"RIFF" + b"\x00" * 32 + b"WAVEfmt " +
            b"\x00" * 16 + b"data" + b"\x00" * 10
        )
    return str(file_path)


def test_initialization(upload_service):
    assert upload_service.upload_directory.exists()


def test_validate_file_invalid_extension(upload_service, tmp_path):
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("text")

    result = upload_service.upload_file(str(file_path), DatasetType.TRAINING)
    assert result.status != ProcessingStatus.SUCCESS
    # Check for error message
    assert result.errors
    assert "formato" in result.errors[0].lower() or \
           "not supported" in result.errors[0].lower()


def test_upload_success(upload_service, sample_audio_file):
    result = upload_service.upload_file(
        sample_audio_file, DatasetType.TRAINING
    )

    assert result.status == ProcessingStatus.SUCCESS
    assert result.data is not None
    assert Path(result.data.file_path).exists()
    assert DatasetType.TRAINING.value in result.data.file_path


def test_create_dataset(upload_service):
    result = upload_service.create_dataset("my_dataset", DatasetType.TRAINING)

    assert result.status == ProcessingStatus.SUCCESS
    metadata = result.data
    assert metadata.name == "my_dataset"
    assert metadata.dataset_type == DatasetType.TRAINING

    # Check if directory was created
    expected_path = upload_service.upload_directory / \
        DatasetType.TRAINING.value / "my_dataset"
    assert expected_path.exists()
