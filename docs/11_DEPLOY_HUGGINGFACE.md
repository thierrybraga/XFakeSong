# Deploy no Hugging Face Spaces

O XfakeSong foi projetado para ser facilmente implantado no [Hugging Face Spaces](https://huggingface.co/spaces), uma plataforma popular para hospedar demonstrações de Machine Learning.

Este guia cobre o processo de deploy usando tanto o **Gradio SDK** (padrão) quanto **Docker** customizado, além de explicar como consumir a API da aplicação hospedada.

## 1. Pré-requisitos

*   Uma conta no [Hugging Face](https://huggingface.co/join).
*   Um novo Space criado (visibilidade Pública ou Privada).
*   (Opcional) Git instalado localmente.

---

## 2. Método 1: Gradio SDK (Recomendado)

O projeto já possui a configuração necessária (`README.md` metadata) para ser detectado automaticamente como uma aplicação Gradio.

### Passo 1: Configuração do Metadata
O arquivo `README.md` na raiz do projeto já contém o bloco YAML necessário:

```yaml
---
title: XfakeSong
emoji: 🛡️
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---
```

### Passo 2: Dependências
Certifique-se de que o arquivo `requirements.txt` na raiz contém todas as bibliotecas necessárias. O Hugging Face instalará automaticamente tudo listado ali.

> **Nota:** Bibliotecas de sistema (como `ffmpeg`) já estão pré-instaladas no ambiente padrão do Gradio SDK. Se precisar de libs exóticas, use o Método 2 (Docker).

### Passo 3: Deploy via Git
Você pode fazer push do código diretamente para o repositório do seu Space.

1.  Clone o repositório do seu Space:
    ```bash
    git clone https://huggingface.co/spaces/SEU_USUARIO/NOME_DO_SPACE
    ```
2.  Copie os arquivos do projeto para dentro da pasta clonada (excluindo `.git` original).
3.  Faça o push:
    ```bash
    git add .
    git commit -m "Deploy inicial XfakeSong"
    git push
    ```

### Passo 4: Sincronização Automática (GitHub Actions)
Para manter seu Space sincronizado com o GitHub automaticamente, crie um workflow em `.github/workflows/sync_to_hub.yml`:

```yaml
name: Sync to Hugging Face hub
on:
  push:
    branches: [main]

  # Permite execução manual
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: git push https://SEU_USUARIO:$HF_TOKEN@huggingface.co/spaces/SEU_USUARIO/NOME_DO_SPACE main
```
*Configure o segredo `HF_TOKEN` nas configurações do seu repositório GitHub.*

---

## 3. Método 2: Docker (Avançado)

Se precisar de controle total sobre o sistema operacional (ex: versões específicas do FFmpeg, drivers CUDA customizados), use o Docker.

### Passo 1: Ajustar Metadata
Altere o `sdk` no `README.md`:

```yaml
sdk: docker
app_port: 7860
```

### Passo 2: Dockerfile
O projeto já inclui um `Dockerfile` otimizado. O Hugging Face construirá a imagem automaticamente baseada nele.

**Pontos de Atenção:**
*   **Permissões:** O HF Spaces roda containers com um usuário não-root (ID 1000). O Dockerfile atual já configura o usuário `appuser` corretamente.
*   **Cache:** O diretório `/app/models` e `/app/data` deve ter permissões de escrita se a aplicação precisar salvar arquivos em tempo de execução (embora o armazenamento seja efêmero, a menos que use Persistent Storage).

---

## 4. Acessando a Interface e API

Uma vez implantado, seu Space estará acessível em `https://huggingface.co/spaces/SEU_USUARIO/NOME_DO_SPACE`.

### Interface Web (GUI)
A interface Gradio completa estará disponível no navegador, permitindo:
*   Upload de áudio para detecção.
*   Visualização de espectrogramas.
*   Treinamento de modelos (se hardware permitir).

### Acesso via API (Gradio Client)
O Gradio gera automaticamente uma API RESTful para sua aplicação. Você pode interagir com ela programaticamente usando a biblioteca `gradio_client`.

#### Instalação
```bash
pip install gradio_client
```

#### Exemplo de Uso (Python)
```python
from gradio_client import Client

# Conectar ao Space (substitua pelo seu)
client = Client("thier/XfakeSong")

# Fazer predição (exemplo para aba de detecção)
# Os endpoints podem variar. Verifique clicando em "Use via API" no rodapé do Space.
result = client.predict(
		"caminho/para/audio.wav", # Arquivo de áudio
		api_name="/predict"       # Nome do endpoint
)

print(result)
```

### Endpoints Disponíveis
Para listar todos os endpoints disponíveis na sua aplicação:
```python
client.view_api()
```
Isso retornará uma lista como:
```text
Client.predict() Usage Info
---------------------------
Named API endpoints: 1

 - predict(audio, api_name="/predict") -> value_0
    Parameters:
     - [Audio] audio: filepath or URL to file
    Returns:
     - [JSON] value_0: Result JSON
```

---

## 5. Variáveis de Ambiente e Segredos

Se sua aplicação precisar de chaves de API ou configurações sensíveis:

1.  Vá em **Settings** no seu Space.
2.  Desça até **Variables and secrets**.
3.  Adicione as variáveis (ex: `DB_CONNECTION_STRING`, `JWT_SECRET`).
4.  No código Python, acesse via `os.environ`:

```python
import os
secret_key = os.environ.get("JWT_SECRET")
```

## 6. Hardware e Performance

*   **CPU Basic (Grátis):** 2 vCPU, 16GB RAM. Bom para inferência de modelos leves (SVM, RawNet2).
*   **GPU (Pago):** T4, A10G, A100. Necessário para treinamento rápido ou inferência de Transformers pesados (WavLM, HuBERT).

Para alterar o hardware, vá em **Settings > Hardware** no painel do Space.
