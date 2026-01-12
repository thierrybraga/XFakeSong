def test_list_architectures(client):
    response = client.get("/api/v1/training/architectures")
    assert response.status_code == 200
    data = response.json()
    assert "architectures" in data
    assert "AASIST" in data["architectures"]


def test_start_training(client, api_key_headers):
    payload = {
        "model_name": "new_model",
        "architecture": "aasist",
        "dataset_path": "/tmp/dataset_v1",
        "epochs": 10,
        "batch_size": 32
    }
    response = client.post(
        "/api/v1/training/start", json=payload, headers=api_key_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert "job_id" in data


def test_check_training_status(client, api_key_headers):
    # First start a job
    payload = {
        "model_name": "new_model",
        "architecture": "AASIST",
        "dataset_path": "/tmp/dataset_v1"
    }
    start_response = client.post(
        "/api/v1/training/start", json=payload, headers=api_key_headers
    )
    job_id = start_response.json()["job_id"]

    # Check status
    response = client.get(f"/api/v1/training/status/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] in ["pending", "running", "completed"]
