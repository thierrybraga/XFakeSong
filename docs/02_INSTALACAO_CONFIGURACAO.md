# Instalação e Configuração

Este guia detalha o processo de configuração do ambiente para executar o sistema XfakeSong em diferentes cenários: Local, Docker e Hugging Face Spaces.

## 📋 Pré-requisitos
- **Python 3.11** ou superior.
- **Pip** atualizado.
- **Git** e **Git LFS** (para modelos grandes).
- **Docker** (Opcional, para modo produção).

---

## 🚀 Instalação Local

### 1. Clonar o Repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd TCC
```

### 2. Configuração Automática (Recomendado)
O sistema possui scripts de inicialização que configuram o ambiente, instalam dependências e iniciam a aplicação.

**Windows:**
```batch
start.bat
```

**Linux/macOS:**
```bash
chmod +x start.sh
./start.sh
```
*Selecione a opção **[1] Modo TESTE** no menu.*

### 3. Configuração Manual (Alternativa)
Se preferir configurar manualmente:

1. **Criar Ambiente Virtual**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instalar Dependências**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Bootstrap e Execução**:
   ```bash
   # Criar pastas necessárias
   python main.py --bootstrap-dirs

   # Iniciar Interface Gráfica
   python main.py --gradio
   ```

---

## 🐳 Execução em Produção (Docker)

O modo produção utiliza Docker para garantir um ambiente isolado e reproduzível.

### Via Script (Recomendado)
Execute `start.bat` (Windows) ou `./start.sh` (Linux) e selecione a opção **[2] Modo PRODUÇÃO**.

### Via Docker Compose (Manual)
1. **Construir e Iniciar**:
   ```bash
   docker-compose up --build -d
   ```
2. **Acompanhar Logs**:
   ```bash
   docker-compose logs -f
   ```
3. **Parar**:
   ```bash
   docker-compose down
   ```
A aplicação estará disponível em `http://localhost:7860`.

---

## 🤗 Deploy no Hugging Face Spaces

O projeto já está configurado para deploy direto no Hugging Face Spaces (SDK Gradio).

### 1. Preparação
Certifique-se de ter o **Git LFS** instalado para suportar arquivos de modelo grandes:
```bash
git lfs install
```

### 2. Configuração do Space
1. Crie um novo Space no Hugging Face: [huggingface.co/new-space](https://huggingface.co/new-space)
2. Selecione:
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (ou superior para treinamento)
   - **Public/Private**: A sua escolha

### 3. Deploy
Você pode fazer deploy de duas formas:

**Opção A: Conectando ao GitHub (Recomendado)**
- No menu de configurações do Space, conecte este repositório GitHub. O deploy será automático a cada push.

**Opção B: Push Direto**
```bash
git remote add space https://huggingface.co/spaces/SEU_USUARIO/NOME_DO_SPACE
git push space main
```

### Arquivos de Configuração do Space
- `app.py`: Ponto de entrada específico para o HF Spaces.
- `packages.txt`: Dependências do sistema (ffmpeg, libsndfile1).
- `requirements.txt`: Dependências Python (com versão do Gradio fixada).
- `.gitattributes`: Configuração do Git LFS para modelos.
- `README.md`: Contém o cabeçalho YAML de metadados do Space.

---

## 🧪 Executando Testes

Para garantir a integridade do sistema, execute os testes unitários e de integração:

```bash
# Instalar dependências de teste (se ainda não instaladas)
pip install pytest pytest-cov

# Executar todos os testes
pytest

# Executar com relatório de cobertura
pytest --cov=app tests/
```

---

## ⚙️ Variáveis de Ambiente (.env)

Copie `.env.example` para `.env` e ajuste conforme necessário:

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `DEEPFAKE_ENV` | `development` ou `production` | `development` |
| `GRADIO_SERVER_PORT` | Porta da interface web | `7860` |
| `DEEPFAKE_MODELS_DIR`| Diretório de modelos | `./app/models` |
