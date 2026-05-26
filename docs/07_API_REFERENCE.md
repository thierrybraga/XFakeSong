# Referência da API REST

A API REST do XFakeSong é construída com **FastAPI** e servida via **Uvicorn** dentro do app unificado (`gradio_app.py`/`main_fastapi.py`). Em ambiente local: `http://localhost:7860`.

**Prefixo de todos os endpoints**: `/api/v1`

## Documentação Interativa

- **Swagger UI**: `http://localhost:7860/api/docs`
- **ReDoc**: `http://localhost:7860/api/redoc`
- **OpenAPI Schema**: `http://localhost:7860/api/openapi.json`

---

## Rate Limiting

A maioria dos endpoints tem limite por minuto e por IP (ex.: `10/minute`, `30/minute`). Retorna HTTP 429 ao exceder.

## Autenticação

Endpoints de mutação (criar dataset, iniciar treinamento, etc.) requerem API Key:
```
X-API-Key: <sua_chave>
```
Configure via env var `XFAKESONG_API_KEY`. Se ausente, o modo dev permite acesso sem autenticação (com aviso no log).

## Tracing

Toda resposta inclui `X-Request-ID` para rastreamento. Pode também ser enviado pelo cliente.

## Erros (RFC 7807)

Erros seguem [RFC 7807 Problem Details](https://datatracker.ietf.org/doc/html/rfc7807):
```json
{
  "type": "about:blank",
  "title": "Validation Error",
  "status": 400,
  "detail": "Mensagem específica",
  "error_code": "VALIDATION_ERROR",
  "request_id": "abc123",
  "errors": [{"field": "x", "message": "..."}]
}
```

---

## System (`/api/v1/system`)

| Método | Path | Descrição |
|--------|------|-----------|
| GET | `/status` | Status operacional + active_services |
| GET | `/health` | Health detalhado (DB, models, storage, uptime) |
| GET | `/bootstrap` | Endpoint trivial de bootstrap |
| GET | `/version` | Versões: app, Python, TF, Keras, gradio, git SHA, platform |
| GET | `/info` | Snapshot consolidado (status + versões + modelos) |

---

## Detection (`/api/v1/detection`)

### `POST /analyze`
Detecta deepfake em um único arquivo.

**Form-data:**
- `file` (UploadFile): áudio (.wav/.mp3/.flac/.m4a/.ogg)
- `model_name` (opcional): nome de modelo treinado específico
- `architecture` (opcional): nome de arquitetura para auto-find
- `variant` (opcional): variante da arquitetura
- `normalize` (default True): normaliza áudio antes da detecção
- `segmented` (default False): inferência em janelas para áudios longos

**Response 200** (`PredictionResult`):
```json
{
  "is_fake": true,
  "confidence": 0.92,
  "probabilities": {"real": 0.08, "fake": 0.92},
  "model_name": "AASIST_v1",
  "features_used": ["mel_spectrogram"],
  "metadata": {...},
  "temperature_applied": 1.42,
  "ood_score": 0.55,
  "is_ood": false,
  "ood_threshold": 0.2,
  "classification_threshold": 0.48
}
```

Os campos `temperature_applied`, `ood_*` e `classification_threshold` foram adicionados nos Sprints 1.4 / 2.5 / 4.5 (todos opcionais — compatibilidade retroativa preservada).

### `POST /multi-model` (Sprint 4.4)
Inferência via fusão de múltiplos modelos.

**Form-data:**
- `file`: áudio
- `model_names`: JSON-encoded `["AASIST_v1", "Conformer_v1"]` (≥2)
- `fusion`: `weighted_avg` | `soft_voting` | `majority_vote` | `max_conf`
- `weights` (opcional): JSON `[0.4, 0.6]`
- `use_tta` (default False): TTA por modelo

**Response 200** (`MultiModelPredictionResult`):
```json
{
  "is_fake": true,
  "confidence": 0.87,
  "probabilities": {"real": 0.13, "fake": 0.87},
  "fusion": "weighted_avg",
  "n_models": 2,
  "fake_votes": 2,
  "model_agreement": 1.0,
  "per_model": [
    {"model": "AASIST_v1", "is_fake": true, "fake_prob": 0.92, ...},
    {"model": "Conformer_v1", "is_fake": true, "fake_prob": 0.81, ...}
  ],
  "metadata": {}
}
```

### `POST /uncertainty` (Sprint 5.4)
Predição com MC Dropout — quantificação de incerteza.

**Form-data:**
- `file`: áudio
- `model_name` (opcional): default = padrão do service
- `n_samples` (5-200, default 20): forward passes MC

**Response 200** (`UncertaintyResult`):
```json
{
  "is_fake": false,
  "confidence": 0.62,
  "epistemic_uncertainty": 0.041,
  "predictive_entropy": 0.553,
  "is_uncertain": true,
  "n_mc_samples": 20,
  "temperature_applied": 1.42,
  "classification_threshold": 0.48,
  "mc_fallback": false
}
```

Use `is_uncertain=true` para implementar decisão "abstenha-se" — útil quando o modelo é hesitante.

### `GET /models`
Lista modelos carregados/disponíveis + default.

### `GET /architectures`
Lista as 14 arquiteturas suportadas.

---

## Features (`/api/v1/features`)

### `POST /extract`
Extrai features de um áudio.

**Form-data:**
- `file`: áudio
- `feature_types`: JSON `["spectral", "cepstral"]`
- `normalize` (default True)

**Response 200** (`FeatureExtractionResult`).

### `GET /types`
Lista os 11 tipos de features (`spectral`, `cepstral`, `temporal`, `prosodic`, `formant`, `voice_quality`, `complexity`, `perceptual`, `predictive`, `timefreq`, `speech`, `mel_spectrogram`).

---

## Training (`/api/v1/training`)

### `POST /start` (requer API Key)
Inicia job de treinamento em background.

**Body** (`TrainingRequest`):
```json
{
  "architecture": "aasist",
  "dataset_path": "data/train.npz",
  "model_name": "aasist_v1",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.0008,
  "parameters": {"dropout_rate": 0.2}
}
```

**Response 200** (`TrainingResponse`) com `job_id`.
Em caso de falha ao criar job: HTTP **503** (corrigido na API.8 — antes era 200 com status="error").

### `GET /status/{job_id}`
Status atual do job (com cache TTL).

### `POST /cross-validate` (Sprint 4.1, requer API Key)
Inicia K-fold CV em background.

**Body** (`CrossValidationRequest`):
```json
{
  "architecture": "conformer",
  "dataset_path": "data/train.npz",
  "config": {"epochs": 30, "batch_size": 32},
  "n_folds": 5,
  "save_fold_models": false
}
```

### `GET /cross-validate/{job_id}`
Resultado agregado quando CV concluída — retorna `aggregated` (mean/std/min/max por métrica) + `best_fold`.

### `GET /architectures`
Lista arquiteturas para treinamento.

---

## History (`/api/v1/history`)

### `GET /`
Histórico paginado de análises (`limit`, `offset`).

### `GET /{analysis_id}` / `DELETE /{analysis_id}`
Detalhes ou exclusão de uma análise.

---

## Datasets (`/api/v1/datasets`)

| Método | Path | API Key | Descrição |
|--------|------|---------|-----------|
| GET | `/` | — | Lista datasets (filtro `type=training/validation/test`) |
| POST | `/` | ✓ | Cria dataset vazio |
| POST | `/{name}/upload` | ✓ | Upload de arquivo |
| DELETE | `/{name}` | ✓ | Remove dataset |

---

## Voice Profiles (`/api/v1/profiles`)

| Método | Path | Descrição |
|--------|------|-----------|
| GET | `/` | Lista todos os perfis |
| POST | `/` | Cria perfil (JSON `ProfileCreate`) |
| GET | `/{id}` | Detalhes |
| PUT | `/{id}` | Atualiza |
| DELETE | `/{id}` | Remove |
| POST | `/{id}/samples` | Upload de amostras |
| DELETE | `/{id}/samples/{filename}` | Remove amostra |
| POST | `/{id}/train` | Treina modelo do perfil |
| POST | `/{id}/detect` | Verifica se áudio é do perfil |

---

## Testes

Smoke tests em `tests/api/test_smoke.py` validam:
- 33 rotas registradas
- OpenAPI schema válido
- Endpoints triviais retornam 200
- Schemas Pydantic expõem campos novos (Sprint 1.4/2.5/4.4/4.5/5.4)
- Backward compatibility com clientes antigos

Execute:
```bash
pytest tests/api/test_smoke.py -v
```
