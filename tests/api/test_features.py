def test_list_feature_types(client):
    response = client.get("/api/v1/features/types")
    assert response.status_code == 200
    data = response.json()
    assert "available_types" in data
    assert "mfcc" in data["available_types"]


# Teste de extração exigiria mockar AudioLoadingService e
# AudioFeatureExtractionService
# Como o router instancia AudioLoadingService internamente, o teste unitário
# puro é difícil sem patching profundo
# Vamos focar no teste de integração ou validação de entrada por enquanto
def test_extract_features_invalid_file(client):
    files = {'file': ('test.txt', b'text data', 'text/plain')}
    response = client.post("/api/v1/features/extract", files=files)
    assert response.status_code == 400
