# Changelog

Mudanças relevantes para quem instala, opera ou reproduz o SpeechChecker. O formato
segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

O projeto não tinha changelog nem tags antes desta entrada. O histórico anterior está
no Git, e `app/__init__.py` declara a versão `1.1.0`.

## [Não lançado]

### Alterado

- **Pilha web:** Gradio 4.44 para 6.17, FastAPI 0.116 para 0.162, Starlette 0.47 para
  1.10, Pillow 12.3 e datasets 5. `torch`, `transformers`, TensorFlow, Keras, NumPy e
  scikit-learn não mudaram, e as predições dos modelos existentes foram conferidas
  contra a versão anterior.
- **Aparência do Gradio:** tema, CSS e cabeçalho saíram de `gr.Blocks`, porque o
  Gradio 6 os recebe em `launch()` e `mount_gradio_app()`. Quem montar a interface por
  conta própria deve passar `APPEARANCE`, de `app/interfaces/gradio/app.py`.
- **Download de datasets:** repositórios do Hugging Face baseados em script deixam de
  carregar, porque o datasets 5 removeu `trust_remote_code`.
- **Estrutura:** scripts foram para `tools/` e Docker foi para `infrastructure/`.
  Módulos grandes foram decompostos mantendo as APIs públicas: assistente de treino,
  perfis de voz, análise forense, `trainer.py`, runner do benchmark e `layers.py`. Os
  modelos `.keras` já salvos continuam carregando.

### Corrigido

- **Investigar:** a linha do tempo, o mapa de anomalias e o histograma do lote mostravam
  a confiança do veredito como se fosse a probabilidade de fake, com limiar fixo de 0,5.
  Agora usam P(fake) e o limiar do modelo.
- **Modelos clássicos:** um run novo agora grava a correção de banda no contrato do
  sidecar, e a inferência a reproduz. Os artefatos já promovidos tinham esse carimbo e
  não mudam. Como `fit_classical.py` entra no fingerprint, o `--resume` retreina os
  clássicos.
- **Benchmark:** o artefato preservado do run agora é uma cópia real. Com hardlink,
  qualquer gravação em `data/models/bench_*` alterava em silêncio o artefato registrado.
- **CI:** a validação dos perfis de inferência do Compose não exige mais um `.env`.

### Segurança

- O gate de release agora usa `tools/ops/check_pip_audit.py`: bloqueia todo aviso fora
  de `configs/security/pip-audit-exceptions.json`, versão diferente da registrada e
  exceção vencida. A migração reduziu os avisos de 8 para 3 pacotes.
- **Exceções abertas, com revisão até 2027-02-01:** `pillow`, `torch` e
  `transformers`, 84 IDs ao todo. A justificativa de cada uma e o motivo de não
  atualizar estão no arquivo de exceções.

### Limitações conhecidas

- **Publicação acadêmica bloqueada:** há pendências de licença do corpus e de
  proveniência das métricas do artigo. Ver
  [o registro da revisão manual](docs/development/release-review.md).
- **API de treino:** quem tem a chave da API pode escolher, em `parameters`, o
  repositório do Hugging Face carregado pelos modelos SSL. Com os pinos atuais de
  `torch` e `transformers`, isso equivale a execução de código no servidor. Trate a
  chave como acesso administrativo.
- **Gradio limitado a 6.17:** a linha 6.18 exige `huggingface_hub>=1.2`, e o
  `transformers` fixado exige `<1.0`.
- **GPU e Linux:** a CI não tem GPU. A aprovação dos perfis NVIDIA só significa que a
  configuração é válida.
