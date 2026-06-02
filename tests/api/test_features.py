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


def test_extract_features_real_success(client, tmp_path):
    """Regressão: extração REAL (sem mock) deve serializar sem erro.

    O fixture `client` não sobrescreve o feature service, então este teste
    exercita o caminho real: AudioLoadingService + extract_single, que retorna
    um objeto AudioFeatures. Pega o bug em que o router fazia `.items()` direto
    no AudioFeatures (AttributeError → 500).
    """
    import numpy as np
    import soundfile as sf

    sr = 16000
    y = (0.3 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr)).astype("float32")
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), y, sr)

    with open(wav, "rb") as fh:
        resp = client.post(
            "/api/v1/features/extract",
            files={"file": ("tone.wav", fh.read(), "audio/wav")},
            data={"feature_types": '["spectral"]', "normalize": "true"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "features" in body and isinstance(body["features"], dict)
    assert len(body["features"]) > 0  # extraiu features reais
    assert "metadata" in body
