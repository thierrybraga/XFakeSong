# XfakeSong — UI com Gradio

Sistema XfakeSong para upload, extração de features, treinamento e inferência de detecção de deepfake de áudio com interface Gradio.

## Requisitos
- Python 3.11+
- Pip atualizado (`python -m pip install --upgrade pip`)
- Dependências do projeto: `requirements.txt`

## Instalação
```bash
# Dentro do diretório do projeto
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Inicializar o Gradio
```bash
# Porta padrão 7860
python main.py --gradio --gradio-port 7860
```
- Abra `http://127.0.0.1:7860/` no navegador.
- Para compartilhar publicamente um link (Gradio share):
```bash
python main.py --gradio --gradio-port 7860 --gradio-share
```

## Dicas de uso
- Aba "Análise Única": faça upload do áudio e selecione o modelo; a classificação e confiança são exibidas.
- Aba "Treino/Modelos": crie modelos (DL) e treine com features segmentadas; resultados e gráficos são salvos em `app/results/`.
- Aba "Resultados & Gráficos": lista JSONs e imagens gerados para inspeção.

## Diretórios úteis
- `app/models/` — artefatos de modelos.
- `app/results/` — métricas, históricos e figuras.
- Bootstrap rápido de diretórios:
```bash
python main.py --bootstrap-dirs
```

## Solução de problemas
- `net::ERR_ABORTED ... /gradio_api/queue/data` no console:
  - Em uso local, a UI roda sem fila; evite múltiplos cliques simultâneos.
  - Se usar `--gradio-share`, a fila é habilitada; aguarde processamento dos callbacks.
- "attempted relative import beyond top-level package":
  - Já corrigido com imports absolutos das arquiteturas (`app.domain.models.architectures.*`). Caso persista, atualize o ambiente e recompile:
```bash
python -m compileall -q app main.py
```
- Portas ocupadas:
  - Mude a porta: `--gradio-port 7861`.

## Desenvolvimento
- Validação rápida de sintaxe:
```bash
python -m compileall -q app main.py
```
- Logs:
  - Console ativo por padrão; arquivo de log em `./logs/deepfake_app.log` se `enable_file_logging=True`.

---
Para dúvidas, abra a UI e confira os rótulos das abas e botões; eles refletem as operações disponíveis no sistema.
