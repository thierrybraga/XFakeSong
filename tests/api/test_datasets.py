def test_list_datasets_empty(client, mock_upload_service):
    # Mock directory behavior
    mock_upload_service.upload_directory.exists.return_value = False

    response = client.get("/api/v1/datasets/")
    assert response.status_code == 200
    assert response.json() == []


def test_create_dataset(client, mock_upload_service, api_key_headers):
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
    # Setup mock to simulate existing dataset
    mock_upload_service.upload_directory.joinpath.return_value \
        .exists.return_value = True

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


def test_delete_dataset(client, mock_upload_service, api_key_headers):
    response = client.delete(
        "/api/v1/datasets/old_dataset?type=training", headers=api_key_headers
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Dataset deleted"

    mock_upload_service.delete_dataset.assert_called_once()
