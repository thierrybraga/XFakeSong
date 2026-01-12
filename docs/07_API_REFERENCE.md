# Referência da API REST

A API REST do XfakeSong permite integração com outros sistemas, automação de tarefas e acesso programático às funcionalidades de detecção de deepfakes, extração de features e treinamento.

## Base URL
A API é servida no mesmo host da aplicação. Em ambiente local: `http://localhost:7860`.
O prefixo para todos os endpoints é `/api/v1`.

## Documentação Interativa (Swagger UI)
A documentação interativa completa (OpenAPI) está disponível em:
- **Swagger UI**: `/docs` (ex: `http://localhost:7860/docs`)
- **ReDoc**: `/redoc` (ex: `http://localhost:7860/redoc`)

---

## Endpoints Principais

### 🔍 Detecção (`/api/v1/detection`)

#### Detectar Deepfake em Áudio
`POST /detection/analyze`

Envia um arquivo de áudio para análise.

**Parâmetros (Form Data):**
- `file`: Arquivo de áudio (obrigatório).
- `model_name`: Nome do modelo específico a usar (opcional).
- `architecture`: Nome da arquitetura para busca automática (opcional).
- `normalize`: Normalizar áudio antes de processar (padrão: `true`).
- `segmented`: Usar análise segmentada (janelamento) (padrão: `false`).

**Exemplo de Resposta:**
```json
{
  "is_fake": true,
  "confidence": 0.98,
  "probabilities": {
    "fake": 0.98,
    "real": 0.02
  },
  "model_name": "aasist_base",
  "features_used": ["raw"],
  "metadata": { ... }
}
```

#### Listar Modelos Disponíveis
`GET /detection/models`

Retorna modelos carregados e disponíveis no disco.

---

### 🎼 Features (`/api/v1/features`)

#### Extrair Características
`POST /features/extract`

Extrai vetores de características (MFCC, Espectrograma, etc.) de um áudio.

**Parâmetros (Form Data):**
- `file`: Arquivo de áudio.
- `feature_types`: Lista JSON de tipos (ex: `["mfcc", "chroma"]`).

**Exemplo de Resposta:**
```json
{
  "features": {
    "mfcc": [[...], [...]],
    "chroma": [[...]]
  },
  "metadata": {
    "duration": 4.5,
    "sample_rate": 22050
  }
}
```

---

### 🧠 Treinamento (`/api/v1/training`)

#### Iniciar Treinamento
`POST /training/start`

Inicia um job de treinamento em background (Simulação/Mock na versão atual).

**Corpo (JSON):**
```json
{
  "architecture": "aasist",
  "dataset_path": "/data/dataset_v1",
  "model_name": "meu_modelo_custom",
  "epochs": 50
}
```

#### Verificar Status
`GET /training/status/{job_id}`

Retorna o status de um job de treinamento.

---

### 📜 Histórico (`/api/v1/history`)

#### Listar Análises Recentes
`GET /history/`

Retorna lista paginada de análises realizadas.

#### Obter Detalhes
`GET /history/{id}`

Retorna detalhes completos de uma análise específica.

---

### ⚙️ Sistema (`/api/v1/system`)

#### Status do Sistema
`GET /system/status`

Verifica saúde e serviços ativos.
