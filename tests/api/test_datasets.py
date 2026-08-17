"""API de datasets: criação, upload e listagem sob `/api/v1/datasets`.

SUJEITO: as rotas HTTP, não o `UploadService` por trás delas — a lógica do
serviço é coberta em `tests/unit/test_upload_service.py`. Aqui o contrato é o
código de status, o payload aceito e a forma da resposta.
"""

def test_list_datasets_empty(client, mock_upload_service, api_key_headers):

    response = client.get(
        "/api/v1/datasets/", headers=api_key_headers
    )
    assert response.status_code == 200
    assert response.json() == []


def test_create_dataset_via_api(client, mock_upload_service, api_key_headers):
    payload = {
        "name": "new_dataset",
        "type": "training",
        "description": "My dataset"
    }
    response = client.post(
        "/api/v1/datasets/", data=payload, headers=api_key_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new_dataset"
    assert data["dataset_type"] == "training"

    # Verify service call
    mock_upload_service.create_dataset.assert_called_once()


def test_create_dataset_invalid_type(client, api_key_headers):
    payload = {
        "name": "new_dataset",
        "type": "invalid_type"
    }
    response = client.post(
        "/api/v1/datasets/", data=payload, headers=api_key_headers
    )
    assert response.status_code == 400


def test_upload_to_dataset(client, mock_upload_service, api_key_headers):
    dataset_dir = mock_upload_service.upload_directory / "training" / "my_dataset"
    dataset_dir.mkdir(parents=True)

    files = {'file': ('test.wav', b'audio data', 'audio/wav')}
    data = {'type': 'training'}

    response = client.post(
        "/api/v1/datasets/my_dataset/upload",
        files=files,
        data=data,
        headers=api_key_headers
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert (dataset_dir / "test.wav").read_bytes() == b"audio data"


def test_upload_rejects_unsupported_format(
    client, mock_upload_service, api_key_headers
):
    dataset_dir = mock_upload_service.upload_directory / "training" / "my_dataset"
    dataset_dir.mkdir(parents=True)

    response = client.post(
        "/api/v1/datasets/my_dataset/upload",
        files={"file": ("payload.exe", b"not audio", "application/octet-stream")},
        data={"type": "training"},
        headers=api_key_headers,
    )

    assert response.status_code == 415
    assert not (dataset_dir / "payload.exe").exists()




def test_upload_rejects_windows_reserved_filename(
    client, mock_upload_service, api_key_headers
):
    dataset_dir = mock_upload_service.upload_directory / "training" / "my_dataset"
    dataset_dir.mkdir(parents=True)

    response = client.post(
        "/api/v1/datasets/my_dataset/upload",
        files={"file": ("NUL.wav", b"audio", "audio/wav")},
        data={"type": "training"},
        headers=api_key_headers,
    )

    assert response.status_code == 400
    assert not (dataset_dir / "NUL.wav").exists()



def test_delete_dataset(client, mock_upload_service, api_key_headers):
    response = client.delete(
        "/api/v1/datasets/old_dataset?type=training", headers=api_key_headers
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Dataset deleted"

    mock_upload_service.delete_dataset.assert_called_once()
