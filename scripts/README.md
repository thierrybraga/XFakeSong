# scripts/ — CLIs operacionais do XFakeSong

CLIs organizados por domínio. Todos são executados **a partir da raiz do
repositório**:

```bash
python scripts/<categoria>/<nome>.py [opções]
```

## Convenções

- **Categorias**: `dataset/` (dados), `training/` (treino/retreino),
  `benchmark/` (execução do benchmark), `reporting/` (consolidação, figuras,
  LaTeX, XAI) e `ops/` (diagnóstico, Docker, testes, HF Hub, notebooks).
- **Nomes**: `verbo_objeto.py` (ex.: `build_dataset.py`, `export_model_card.py`).
- **Cabeçalho padrão**: cada script resolve a raiz do repo com
  `Path(__file__).resolve().parents[2]` e a insere em `sys.path` antes de
  importar `app/` ou `benchmarks/`. Helpers compartilhados (logging, raiz)
  vivem em [`scripts/_bootstrap.py`](_bootstrap.py) — use-os em scripts novos.
- **Docstring obrigatório** com propósito, entradas/saídas e bloco `Uso:`;
  logging via `logging` (nunca `print` para diagnóstico); type hints em
  funções públicas; `argparse` com `--help` funcional.
- A lógica de negócio permanece em `app/` e `benchmarks/` — scripts são
  camadas finas de orquestração (Clean Architecture; ver `AGENTS.md`).

## dataset/ — aquisição, construção e auditoria de dados

| Script | Função |
| --- | --- |
| `download_datasets.py` | Baixa os datasets PT-BR (BRSpeech-DF, MLS Portuguese, TTS-Portuguese, Fake Voices/XTTS, CommonVoice, FLEURS) com cache local e verificação. |
| `build_dataset.py` | Orquestra a Fase 1: composição balanceada real/fake por fonte (tiers `small/medium/large`) e splits estratificados em `app/datasets/splits`. |
| `preprocess_dataset.py` | Valida e normaliza WAVs (16 kHz mono, amplitude, duração 1–30 s, remoção de corrompidos/duplicatas) com relatório detalhado. |
| `export_npz_from_splits.py` | Exporta os splits para um `.npz` canônico de áudio bruto (`benchmark_audio_raw_balanced_15k.npz`). |
| `rebuild_speaker_manifest.py` | Reconstrói `speaker_manifest.json` a partir de metadados locais rastreáveis (sem inventar falantes). |
| `audit_speaker_manifest.py` | Audita a cobertura de IDs reais de falante no dataset ativo; falha abaixo do mínimo configurado. |
| `export_speaker_table.py` | Exporta CSV/JSONL com uma linha por WAV (classe, split, fonte, speaker, duração). |
| `audit_dataset_leakage.py` | Auditoria forense de vazamento de domínio (atalho intra-fonte via features globais + regressão logística). |

## training/ — treinamento e retreino

| Script | Função |
| --- | --- |
| `train_advanced.py` | Pipeline de treinamento por arquitetura nas splits PT-BR, com métricas completas por modelo. |
| `train_by_family.py` | Entrypoint único por família (`--family {classical-tabular,spectral-convolutional,spectral-attention,waveform-end-to-end,ssl-pretrained,extended}`), lendo presets de `configs/training/*.yaml` e delegando a `benchmark/run_models_sequential.py`. Substitui os antigos `train_classical/tensorflow/pytorch/ssl.py`. |
| `retrain_ajustado.sh` / `.bat` | Retreino dos modelos ajustados pós-diagnóstico (config `configs/training/retune_ajustado.yaml`; ver `docs/evaluation/retraining-adjustments.md`). |
| `retrain_wsl2.sh` | Fluxo de retreino completo sob WSL2/GPU com consolidação e atualização do LaTeX ao final. |
| `ablate_wavlm_finetune.py` | Ablação exploratória de fine-tuning do WavLM (fora do recorte do TCC). |

## benchmark/ — execução do benchmark

| Script | Função |
| --- | --- |
| `run_benchmark.py` | CLI principal do benchmark: treina/avalia arquiteturas pelo pipeline real (`--quick`, `--full`, `--model`, `--plan-only`) e gera tabelas/figuras/JSON. |
| `run_models_sequential.py` | Orquestrador um-modelo-por-vez com timeout, `--resume` e log/pasta próprios por modelo; roteia SSL para o runner PyTorch. |
| `run_tcc_pipeline.py` | Automação ponta a ponta do TCC: download → validação/splits → NPZ canônico → benchmark completo. |
| `run_clean_benchmark_pipeline.py` | Run limpo sem misturar artefatos antigos (limpeza, imagem Docker nomeada, manifesto, execução sequencial). |
| `run_wavlm_original_benchmark.py` | Runner PyTorch/`transformers` dos SSL reais (WavLM/HuBERT congelados + cabeça supervisionada, augmentation e calibração sob ruído). |
| `benchmark_latency.py` | Mede latência de inferência por arquitetura (média±desvio, P50/P95, throughput, memória). |
| `robustness_test.py` | Teste isolado de robustez AWGN (SNR 10/20/30 dB) para uma arquitetura. |

## reporting/ — consolidação, artefatos, figuras e XAI

| Script | Função |
| --- | --- |
| `build_paper_from_benchmark.py` | **Entrypoint canônico do artigo**: encadeia consolidate → validate → tabelas → `pdflatex` num comando só (orquestrador fino; não reimplementa lógica). |
| `consolidate_results.py` | Lê `results.json` de um ou mais runs, monta `benchmark_summary.json` e (re)gera todas as figuras nomeadas do TCC. |
| `update_tcc_latex.py` | Gera o fragmento `results/paper/tabelas_benchmark.tex` a partir do sumário consolidado (fonte única das tabelas do TCC). |
| `validate_artifacts.py` | Valida artefatos de modelos/resultados (presença, esquema, coerência) sem carregar pesos. |
| `sync_completed_benchmark_artifacts.py` | Promove modelos concluídos para `app/models/benchmark_final/<arch>/`. |
| `generate_completed_benchmark_artifacts.py` | Regera relatórios/figuras apenas-avaliação a partir de modelos já treinados (`app/models/bench_*`). |
| `materialize_benchmark_artifacts.py` | Materializa manifestos locais de artefatos treinados (fluxo Docker/WSL com bind mount). |
| `export_model_card.py` | Exporta o model card Markdown consolidado dos artefatos treinados (`app/models/MODEL_CARD.md`). |
| `export_rf_feature_importance.py` | Extrai `feature_importances_` do Random Forest promovido e gera figura+tabela LaTeX (63 descritores). |
| `export_tcc_extra_figures.py` | Gera curvas DET e distribuições de score (AASIST×RawGAT-ST) a partir de `predictions_clean.csv`. |
| `rebuild_inference_contracts.py` | Regenera os sidecars `bench_*_config.json` dos modelos promovidos a partir do run real (`metrics.json` + `predictions_clean.csv`): `eer_threshold` verdadeiro e `input_contract` completo com `feature_frontend` do benchmark. |
| `run_shap_analysis.py` | **XAI**: análise SHAP dos clássicos (RF/SVM sobre o vetor tabular) e mapas de ativação Grad-CAM das redes espectrais Keras; ver `app/domain/xai/`. |

## ops/ — operação, infraestrutura e qualidade

| Script | Função |
| --- | --- |
| `doctor.py` | Diagnóstico de instalação/inicialização (`--fix` tenta corrigir): Python, venv, dependências, porta, GPU. |
| `docker_build.py` | Helper de build/execução dos perfis Docker (`inference/train/benchmark × cpu/nvidia`) sobre os composes segmentados. |
| `verify_environments.py` | Sobe containers efêmeros por família e verifica importações/versões das bibliotecas-chave (e GPU nos perfis NVIDIA). |
| `run_tests.sh` | Entrypoint de testes usado pela CI (`fast`, `cov`, suites por marcador). |
| `build_smoke_test.sh` | Smoke test do build Docker completo. |
| `sync_hf_models.py` | Baixa modelos treinados do HF Hub para `app/models` (no-op sem `MODEL_REPO_ID`; usado no boot de Spaces). |
| `upload_models_to_hf.py` | Publica artefatos consolidados no HF Hub (dry-run sem credenciais; nunca imprime token). |
| `build_notebooks.py` | Regenera os notebooks de estudo/reprodução em `docs/notebooks/` com API real do projeto. |
| `setup_gpu_windows.bat` | Configuração de GPU/CUDA em Windows nativo. |
| `run_large_dataset_and_benchmark.ps1` | Fluxo Windows: build da imagem, dataset tier `large`, auditoria e benchmark sequencial. |

## Migração (jul/2026)

Scripts realocados de `scripts/*.py` para as categorias acima. Removidos por
consolidação (funcionalidade preservada):

| Removido | Use no lugar |
| --- | --- |
| `benchmark_all.py` | `scripts/benchmark/run_models_sequential.py` (mesmos flags) |
| `train_classical.py` | `scripts/training/train_by_family.py --family classical-tabular` |
| `train_tensorflow.py` | `scripts/training/train_by_family.py --family spectral-attention` |
| `train_pytorch.py` | `scripts/training/train_by_family.py --family waveform-end-to-end` |
| `train_ssl.py` | `scripts/training/train_by_family.py --family ssl-pretrained` |
