# 15 — Sistema de Benchmark, Modelos Treinados e Resultados

O pacote `benchmarks/` gera, de forma **reprodutível** e usando
o **pipeline real** (`TrainingService → ModelLoader → Predictor →
MetricsCalculator`) e a **API** (FastAPI `TestClient`), os dados empíricos do
projeto: desempenho por arquitetura, eficiência computacional, robustez a ruído
e teste de sistema da API.

> Importante: este harness usa o pipeline **já corrigido** (treino→salvar→
> carregar→prever funcional). Os números aqui substituem com fidelidade os
> medidos manualmente, incluindo os modelos `raw-audio` e os baselines
> clássicos (SVM/RF) que faltavam.

## Estado consolidado atual

O material acadêmico foi consolidado em uma única fonte LaTeX pronta para
Overleaf:

| Artefato | Caminho |
|---|---|
| Fonte principal do artigo | `tcc_overleaf/main.tex` |
| Pacote Overleaf limpo | `tcc_overleaf.zip` |
| Figuras usadas no artigo | `tcc_overleaf/figures/*.png` |
| Matrizes de confusão por arquitetura | `tcc_overleaf/figures/confusion_matrices/*.png` |
| Dataset do benchmark atual | `app/datasets/benchmark_audio_raw_balanced_15k.npz` |
| Modelos default da Gradio/API | `app/models/bench_*` |
| Modelos completos por arquitetura | `app/models/benchmark_final/<arquitetura>/` |
| Manifesto dos modelos consolidados | `app/models/benchmark_final_manifest.json` |
| Resultados e relatórios de benchmark | `results/<run>/` |

Não há PDFs versionados como fonte de verdade. O PDF deve ser gerado a partir de
`tcc_overleaf/main.tex` no Overleaf ou localmente com `pdflatex`.

A versão navegável da fundamentação e análise experimental está em
[Estudo Experimental](20_ESTUDO_EXPERIMENTAL.md), incluindo equações,
fluxograma, modelos, resultados, discussão, limitações e comandos de reprodução.

### Modelos treinados consolidados

Os 14 diretórios finais em `app/models/benchmark_final/` preservam o artefato
completo de cada arquitetura, incluindo backbones SSL quando aplicável. No topo
de `app/models/`, ficam os arquivos carregáveis diretamente pela interface e
pela API (`.keras`/`.pkl`) e seus respectivos `bench_*_config.json`.

| Arquitetura | Artefato principal | Diretório completo |
|---|---|---|
| AASIST | `app/models/bench_aasist.keras` | `app/models/benchmark_final/aasist/` |
| Conformer | `app/models/bench_conformer.keras` | `app/models/benchmark_final/conformer/` |
| EfficientNet-LSTM | `app/models/bench_efficientnet_lstm.keras` | `app/models/benchmark_final/efficientnet_lstm/` |
| Ensemble | `app/models/bench_ensemble.keras` | `app/models/benchmark_final/ensemble/` |
| HuBERT Original | `app/models/benchmark_final/hubert_original/bench_hubert_original.pt` | `app/models/benchmark_final/hubert_original/` |
| Hybrid CNN-Transformer | `app/models/bench_hybrid_cnn_transformer.keras` | `app/models/benchmark_final/hybrid_cnn_transformer/` |
| MultiscaleCNN | `app/models/bench_multiscalecnn.keras` | `app/models/benchmark_final/multiscalecnn/` |
| RandomForest | `app/models/bench_randomforest.pkl` | `app/models/benchmark_final/randomforest/` |
| RawGAT-ST | `app/models/bench_rawgat_st.keras` | `app/models/benchmark_final/rawgat_st/` |
| RawNet2 | `app/models/bench_rawnet2.keras` | `app/models/benchmark_final/rawnet2/` |
| Sonic Sleuth | `app/models/bench_sonic_sleuth.keras` | `app/models/benchmark_final/sonic_sleuth/` |
| SpectrogramTransformer | `app/models/bench_spectrogramtransformer.keras` | `app/models/benchmark_final/spectrogramtransformer/` |
| SVM | `app/models/bench_svm.pkl` | `app/models/benchmark_final/svm/` |
| WavLM Original | `app/models/benchmark_final/wavlm_original/bench_wavlm_original.pt` | `app/models/benchmark_final/wavlm_original/` |

WavLM Original e HuBERT Original são artefatos PyTorch/SSL completos; por isso
ficam preservados no diretório completo com o backbone (`wavlm_backbone/` ou
`hubert_backbone/`). Os demais modelos Keras/sklearn têm cópia direta no topo de
`app/models/`.

### Publicação dos modelos no Hugging Face Hub

A fonte oficial para publicação é `app/models/`, pois é a mesma pasta usada
pela Gradio/API como default. Antes de enviar, verifique o plano de upload:

```bash
python scripts/upload_models_to_hf.py \
    --repo-id SEU_USUARIO/xfakesong-models \
    --dry-run
```

Depois envie para um repositório do tipo **Model**:

```bash
python scripts/upload_models_to_hf.py \
    --repo-id SEU_USUARIO/xfakesong-models \
    --private
```

O script usa `HF_TOKEN` ou `HUGGINGFACE_HUB_TOKEN`, cria o repositório quando
necessário e sobe os arquivos para `models/` no Hub. Use
`--include-overleaf` e `--include-results` apenas quando quiser anexar o pacote
do artigo e relatórios consolidados junto ao repositório de modelos.

### Resultados numéricos usados no artigo

O benchmark atual usa o dataset nominal de 15k
`app/datasets/benchmark_audio_raw_balanced_15k.npz`, gerado pelo tier
`medium`, com 15.000 amostras alvo, split estratificado 70/15/15 e 2.250
amostras de teste. O arquivo `.npz` consolidado tem 2.769,01 MiB e foi
exportado a partir de 15.000 WAVs ativos em PCM linear, 16 bits, mono e
16 kHz. Os modelos neurais finais foram treinados
por 100 épocas quando aplicável; SVM e RandomForest usam GridSearchCV + ajuste
final.

| Modelo | Accuracy | AUC ROC | EER | Decisão prática |
|---|---:|---:|---:|---|
| Conformer | 100,00% | 1,0000 | 0,00% | Demonstração principal de maior qualidade |
| Sonic Sleuth | 100,00% | 1,0000 | 0,00% | Demo leve/estável |
| Hybrid CNN-Transformer | 99,96% | 1,0000 | 0,00% | Melhor compromisso neural para Gradio/API |
| MultiscaleCNN | 99,73% | 1,0000 | 0,18% | Comparação neural convolucional |
| SVM | 99,02% | 0,9995 | 0,98% | Baseline rápido em CPU |
| RandomForest | 98,18% | 0,9986 | 1,91% | Baseline clássico complementar |
| RawNet2 | 96,36% | 0,9961 | 3,56% | Modelo raw-audio funcional |
| Ensemble | 95,82% | 0,9970 | 4,18% | Fusão multi-feature, robustez com ressalvas |
| RawGAT-ST | 95,29% | 0,9940 | 4,71% | Comparação com atenção em grafos |
| AASIST | 93,64% | 0,9918 | 6,36% | Comparação com arquitetura GAT |
| HuBERT Original | 92,71% | 0,9708 | 7,29% | Referência SSL |
| EfficientNet-LSTM | 91,16% | 0,9721 | 9,07% | Transfer learning funcional, não prioritário para demo |
| WavLM Original | 86,36% | 0,9240 | 13,64% | Referência SSL experimental |
| SpectrogramTransformer | 71,51% | 0,7779 | 28,49% | Candidato a novo tuning |

## Como rodar

```bash
# 1) Verificação do harness (sintético, 1 época) — segundos:
python scripts/run_benchmark.py --quick

# 2) Pipeline completo do TCC: download, processamento, split, treino,
#    inferência, gráficos PNG e relatórios Markdown:
python scripts/run_tcc_pipeline.py \
    --download \
    --target-per-class 7500 \
    --full-benchmark \
    --epochs 100 \
    --device-profile gpu \
    --out results/tcc_full_15k \
    --npz app/datasets/benchmark_audio_raw_balanced_15k.npz

# 3) Execução do TCC direto no benchmark, usando dataset real .npz já exportado:
python scripts/run_benchmark.py \
    --full \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --epochs 100 \
    --device-profile gpu

# 4) Benchmark neural completo, sem SVM/RF:
python scripts/run_benchmark.py \
    --neural \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --epochs 100 \
    --device-profile gpu \
    --out results/bench_neural_tcc

# 5) Sob medida:
python scripts/run_benchmark.py \
    --archs WavLM HuBERT RawNet2 "Sonic Sleuth" AASIST RawGAT-ST Conformer \
    "Hybrid CNN-Transformer" SpectrogramTransformer EfficientNet-LSTM \
    MultiscaleCNN Ensemble SVM RandomForest \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --epochs 100 --snr 30 20 10 --api --out results/bench_tcc

# 6) Modelo individual:
python scripts/run_benchmark.py \
    --model AASIST \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --epochs 100 \
    --out results/bench_aasist
```

Por padrão, os relatórios, métricas, CSVs e figuras ficam em `results/benchmark/`
ou no diretório informado por `--out`. Os modelos treinados pelo benchmark ficam
em `app/models/`, o mesmo diretório usado pela Gradio/API para inferência. Use
`--models-dir outro/diretorio` ou `DEEPFAKE_MODELS_DIR` apenas quando quiser
isolar uma execução. Caminhos relativos são resolvidos a partir da raiz do
projeto, mesmo quando o comando é chamado de outro diretório.

Para usar os modelos treinados na demonstração visual, consulte
[Interface Gradio](23_INTERFACE_GRADIO.md). Para publicar modelos e demo,
consulte [GitHub Pages e Hugging Face](24_PUBLICACAO_GITHUB_HF.md).

### Execução por família de ambiente

Além do preset completo, o benchmark pode ser executado por família
computacional. Essa é a rota recomendada quando o objetivo é isolar dependências
e aproveitar GPU/CPU de forma controlada.

| Família | Preset | Comando |
|---|---|---|
| Clássicos | `configs/training/classical.yaml` | `python scripts/train_classical.py` |
| TensorFlow/Keras | `configs/training/tensorflow.yaml` | `python scripts/train_tensorflow.py` |
| PyTorch áudio | `configs/training/pytorch.yaml` | `python scripts/train_pytorch.py` |
| SSL/Transformers | `configs/training/ssl.yaml` | `python scripts/train_ssl.py` |

```bash
# Revisão sem iniciar treino
python scripts/train_tensorflow.py --plan-only

# Modelo individual dentro da família TensorFlow/Keras
python scripts/train_tensorflow.py \
    --models SpectrogramTransformer \
    --epochs 100 \
    --device-profile gpu \
    --out results/spectrogram_transformer_retrain

# Execução Docker com GPU para SSL
docker compose -f docker/compose/train.nvidia.yml run --rm ssl-transformers

# Benchmark completo em container NVIDIA
docker compose -f docker/compose/benchmark.nvidia.yml run --rm benchmark
```

Cada wrapper delega para `scripts/run_models_sequential.py`; por isso o padrão
de saída continua o mesmo: `run_summary.json`, `run_summary.md`,
`<modelo>/run.log`, `<modelo>/results.json` e
`<modelo>/architectures/<modelo>/*.png`.

## Roteiro oficial do dataset robusto

O benchmark do TCC atual usa o tier `medium`, balanceado 1:1 com `7.500`
áudios reais e `7.500` áudios fake, totalizando 15.000 amostras alvo. Em
27/06/2026, Common Voice/FLEURS no Hugging Face não entregaram a cota real
PT-BR: a tentativa PT-BR estrita validada ficou em **9.008 amostras**
(4.504/4.504). A consolidação canônica de 15k, portanto, usa reforço real fora
do HF e registra a composição efetiva no manifesto.

### Estatísticas do dataset consolidado

Revisão local: **28/06/2026**.

| Item | Valor |
|---|---:|
| WAVs ativos | 15.000 |
| Real / fake | 7.500 / 7.500 |
| Tamanho dos WAVs ativos | 3.746,26 MiB |
| Duração dos WAVs ativos | 2.045,61 min / 34,09 h |
| Tamanho do NPZ | 2.769,01 MiB |
| Duração efetiva no NPZ | 1.250,00 min / 20,83 h |
| Formato dos WAVs | WAV PCM linear, 16 bits, mono, 16 kHz |
| Entrada raw no NPZ | `(80000, 1)` por amostra |
| Chaves de falante no ativo | 73 |

| Classe | Fonte | Arquivos | MiB | Minutos | Falantes/chaves |
|---|---|---:|---:|---:|---:|
| real | BRSpeech-DF bonafide | 3.750 | 847,84 | 462,95 | 1 fallback |
| real | MLS Portuguese | 1.875 | 894,37 | 488,40 | 21 |
| real | TTS-Portuguese Corpus | 1.875 | 645,12 | 352,28 | 1 |
| fake | BRSpeech-DF spoof | 3.750 | 836,65 | 456,84 | 1 fallback |
| fake | Fake Voices XTTS | 3.750 | 522,27 | 285,15 | 50 |

1. baixar BRSpeech-DF e separar `bonafide` em real e `spoof` em fake;
2. completar a classe real com MLS Portuguese e TTS-Portuguese Corpus;
3. manter Common Voice/FLEURS/CETUC apenas como legado local quando já existirem
   ou voltarem a ficar disponíveis;
4. completar a classe fake com Fake Voices XTTS;
5. normalizar tudo para WAV mono 16 kHz, remover arquivos inválidos,
   silenciosos, fora de duração e duplicados;
6. criar split estratificado 70/15/15;
7. exportar `app/datasets/benchmark_audio_raw_balanced_15k.npz`;
8. executar o preflight (`benchmark_plan.json`/`.md`) com preset, ambiente,
   dataset e hiperparâmetros efetivos;
9. treinar, inferir e gerar relatórios/gráficos para as 14 arquiteturas.

### Catálogo de fontes usado no benchmark

O catálogo único de datasets fica em `app/core/dataset_catalog.py` e é usado
pela aba Gradio **Datasets/Download**, pela documentação e pelo exportador
`scripts/run_tcc_pipeline.py`. Ele registra, para cada fonte: tipo (`real`,
`fake` ou `both`), flag de download, prefixos de arquivo, licença, idioma,
quantidade/duração conhecida, falantes e uso recomendado no benchmark.

O preset mais completo para novas rodadas é **Benchmark Robusto Recomendado**:

| Classe | Fontes recomendadas |
|---|---|
| Real | BRSpeech-DF bonafide, MLS Portuguese, TTS-Portuguese Corpus, ASVspoof 2019 bonafide, In-the-Wild bonafide |
| Fake | BRSpeech-DF spoof, Fake Voices, WaveFake, ASVspoof 2019 spoof, In-the-Wild spoof |

Esse preset aumenta diversidade de idioma, falantes, geradores e vocoders. Para
o TCC, registre sempre a composição efetiva: MLS é português/LibriVox e não
PT-BR estrito; TTS-Portuguese é PT-BR, mas single-speaker; Common Voice/FLEURS
ficam como fontes legadas enquanto estiverem indisponíveis no HF.

O `.npz` exportado pelo pipeline inclui os metadados usados pelo benchmark:

| Campo no NPZ/manifesto | Uso |
|---|---|
| `metadata_json.source_summary` | Contagem por fonte e horas estimadas pelas janelas exportadas |
| `metadata_json.dataset_catalog` | Snapshot do catálogo usado naquela execução |
| `metadata_json.splits.<split>.source_summary` | Composição por fonte em treino, validação e teste |
| `groups` | Fonte por amostra, derivada do prefixo; usada em `--group-split` e `--cross-generator` |
| `speaker_ids` | Falante por amostra; usa ID real do manifesto quando disponível e fallback por fonte quando não disponível |

### Tiers e protocolos de split

O dataset é montado por **tier** (ver [docs/12_DATASETS.md](12_DATASETS.md)). O
tier determina tamanho, fontes e estratégia de split; o benchmark expõe os
protocolos correspondentes:

| Tier | `build_dataset` | Protocolo de avaliação no benchmark |
|------|-----------------|-------------------------------------|
| test / small | `--tier test\|small` | split estratificado para smoke/iteração |
| medium | `--tier medium` | benchmark canônico 15k, split estratificado |
| large | `--tier large` | auditoria 20k com `--speaker-split` ou `--unseen-speaker <id>` |

Protocolos anti-vazamento disponíveis no `run_benchmark.py` / `run_tcc_pipeline.py`:

- `--group-split` / `--cross-generator <gerador>` — disjunto por **fonte/gerador**
  (preset `group_tcc` / `cross_generator_tcc`).
- `--speaker-split` / `--unseen-speaker <falante>` — disjunto por **falante**
  (preset `unseen_speaker_tcc`), exige um `.npz` com `speaker_ids` úteis; é
  recomendado no tier `large`.

```bash
# tier medium ponta a ponta: benchmark canônico 15k
python scripts/run_tcc_pipeline.py --download --tier medium \
    --full-benchmark --epochs 100 --device-profile gpu \
    --out results/tcc_medium_15k \
    --npz app/datasets/benchmark_audio_raw_balanced_15k.npz
```

O script `scripts/build_dataset.py` arquiva excedentes em
`app/datasets/overflow/` por padrão, em vez de apagar os WAVs brutos. Use
`--delete-excess` apenas quando o descarte destrutivo for intencional.

Comando completo recomendado:

```bash
python scripts/run_tcc_pipeline.py \
    --download \
    --target-per-class 7500 \
    --full-benchmark \
    --epochs 100 \
    --device-profile gpu \
    --out results/tcc_full_15k \
    --npz app/datasets/benchmark_audio_raw_balanced_15k.npz
```

Para um ensaio rápido do roteiro sem downloads:

```bash
python scripts/run_tcc_pipeline.py \
    --smoke \
    --epochs 1 \
    --batch-size 4 \
    --latency-runs 1 \
    --out results/smoke_route
```

Para revisar tudo antes de iniciar o treinamento longo:

```bash
python scripts/run_benchmark.py \
    --full \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --epochs 100 \
    --out results/tcc_full_15k \
    --plan-only
```

Para revisar um modelo individual:

```bash
python scripts/run_benchmark.py \
    --model RawNet2 \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/bench_rawnet2 \
    --plan-only
```

Esse comando valida o `.npz` e grava:

| Arquivo | Conteúdo |
|---|---|
| `benchmark_plan.json` | preset, dataset, ambiente, arquiteturas e hiperparâmetros efetivos |
| `benchmark_plan.md` | tabela legível com epochs, batch, learning rate e ajuste CPU/GPU por arquitetura |

O alvo final usado no artigo é `7.500` amostras reais + `7.500` amostras fake.
O roteiro aceita alvos maiores para novas rodadas, mas os resultados,
intervalos de confiança e gráficos do TCC foram consolidados sobre
`benchmark_audio_raw_balanced_15k.npz`. Use `--skip-download` quando os WAVs e
splits já estiverem prontos localmente.

## Preset e hiperparâmetros pré-treino

O preset oficial é `full_tcc`:

- arquiteturas: WavLM, HuBERT, RawNet2, Sonic Sleuth, AASIST, RawGAT-ST,
  Conformer, Hybrid CNN-Transformer, SpectrogramTransformer,
  EfficientNet-LSTM, MultiscaleCNN, Ensemble, SVM e RandomForest;
- dataset: `.npz` balanceado exportado do split 70/15/15;
- épocas globais do benchmark TCC: `100` por padrão nos presets completos,
  ajustável com `--epochs`;
- robustez: AWGN em `30`, `20` e `10` dB;
- latência: mediana de `30` execuções por arquitetura;
- API: probe habilitado no preset completo.

Para treinar apenas os modelos neurais, use o preset `neural_tcc`
(`--neural` ou `--preset neural_tcc`). Ele roda as 12 arquiteturas neurais:
WavLM, HuBERT, RawNet2, Sonic Sleuth, AASIST, RawGAT-ST, Conformer,
Hybrid CNN-Transformer, SpectrogramTransformer, EfficientNet-LSTM,
MultiscaleCNN e Ensemble. SVM/RF ficam reservados para o baseline clássico.

Antes do treino, o preflight aplica hiperparâmetros recomendados por
arquitetura e adapta o `batch_size` ao perfil de dispositivo:

- `--device-profile auto`: usa GPU se o TensorFlow detectar CUDA, senão CPU;
- `--device-profile cpu`: limita batches de modelos pesados para evitar OOM/RAM;
- `--device-profile gpu`: habilita caps conservadores de VRAM e mixed precision
  quando a arquitetura permite;
- `--no-optimize-hparams`: desliga recomendações por arquitetura e usa os
  valores globais `--epochs`/`--batch-size`.

Hiperparâmetros neurais efetivos do benchmark:

| Arquitetura | Entrada | Batch base | LR | Dropout | L2 | Observação |
|---|---|---:|---:|---:|---:|---|
| WavLM | raw audio | 1 em GPU / 4 em CPU | 1e-5 | 0.2 | 1e-4 | sem augmentation/mixed precision; batch conservador por VRAM; backbone SSL ou fallback reportado |
| HuBERT | raw audio | 1 em GPU / 4 em CPU | 1e-5 | 0.2 | 1e-4 | `from_pt=True` quando backbone real existe; fallback reportado |
| RawNet2 | raw audio | 24 | 1e-5 | 0.3 | 1e-4 | LR conservador para Sinc/GRU |
| Sonic Sleuth | espectrograma/features | 32 | 1e-3 | 0.3 | 1e-4 | ReduceLROnPlateau |
| AASIST | espectrograma | 16 | 1e-4 | 0.2 | 1e-4 | mantém loss/margem da arquitetura |
| RawGAT-ST | raw audio | 16 | 1e-4 | 0.2 | 1e-4 | SincNet + GAT espectral/temporal com LR estável |
| Conformer | espectrograma | 32 | 1e-3 | 0.3 | 1e-4 | WarmupCosineDecay |
| Hybrid CNN-Transformer | espectrograma | 32 | 1e-3 | 0.2 | 1e-4 | 3 blocos residuais + 2 camadas Transformer |
| SpectrogramTransformer | espectrograma | 16 | 1e-4 | 0.1 | 1e-5 | AST com batch conservador por memória |
| EfficientNet-LSTM | espectrograma | 32 | 5e-4 | 0.4 | 2e-4 | transfer learning + Bi-LSTM |
| MultiscaleCNN | espectrograma | 64 | 2e-3 | 0.5 | 5e-4 | baseline neural convolucional |
| Ensemble | espectrograma/features | 32 | 1e-3 | 0.3 | 1e-4 | 100 épocas no benchmark neural completo |

O campo `epochs` no plano é sempre o valor solicitado no CLI (`--epochs`),
enquanto `recommended_epochs` documenta a receita completa por arquitetura.
Assim, o mesmo pipeline serve para smoke test (`--epochs 1`), piloto
(`--epochs 20`) e treinamento final (`--epochs 100`).

O pipeline completo chama esse preflight automaticamente antes de iniciar o
benchmark. Use `--skip-benchmark-preflight` apenas para depuração local.

## Benchmark de modelo individual

Use `--model` quando quiser treinar e avaliar apenas uma arquitetura. O script
continua gerando os mesmos artefatos (`results.json`, `tcc_report.md`,
`figures/*.png`, `architectures/<modelo>/*.png`), mas restritos ao modelo
selecionado.

```bash
python scripts/run_tcc_pipeline.py \
    --skip-download \
    --skip-preprocess \
    --model SpectrogramTransformer \
    --npz app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/bench_spectrogram_transformer
```

Para múltiplos modelos, mantenha `--archs`. `--model` e `--archs` são
mutuamente exclusivos.

## Execução sequencial com timeout

Para rodar todas as arquiteturas de forma resiliente, use o orquestrador
sequencial. Ele executa um modelo por vez, cria uma subpasta por modelo, grava
`run.log`, aplica timeout e permite retomar com `--resume`.

```bash
python scripts/run_models_sequential.py \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/sequential_15k \
    --device-profile gpu \
    --timeout-min 90
```

Rodar somente os modelos neurais:

```bash
python scripts/run_models_sequential.py \
    --neural-only \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/sequential_neural_15k \
    --device-profile gpu \
    --timeout-min 90
```

Revisar planos neurais antes do treino:

```bash
python scripts/run_models_sequential.py \
    --neural-only \
    --plan-only \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/sequential_neural_plan \
    --device-profile cpu
```

No `plan-only` com `.npz` real, cada subprocesso carrega o dataset para
registrar forma, balanço e metadados. Em datasets grandes, espere alguns
segundos por modelo mesmo sem iniciar treino.

Retomar somente modelos pendentes:

```bash
python scripts/run_models_sequential.py \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/sequential_15k \
    --device-profile gpu \
    --timeout-min 90 \
    --resume
```

Executar um subconjunto:

```bash
python scripts/run_models_sequential.py \
    --models AASIST RawNet2 Conformer \
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
    --out results/sequential_neural_subset
```

No Windows nativo com TensorFlow 2.11+, CUDA não é exposto ao TensorFlow. Para
`--device-profile gpu`, rode esse script dentro do WSL2/Linux com
`tensorflow[and-cuda]` instalado.

Por padrão, o CLI do benchmark imprime apenas o resumo final e avisos
importantes. Use `--verbose` para depuração detalhada de treino, registry,
factories e salvamento de modelos.

O `.npz` deve conter `X_train`/`y_train` (e opcionalmente `X_val`/`X_test`); o
harness **reconcatena e re-divide 70/15/15 estratificado** com semente fixa,
garantindo um conjunto de teste *held-out* controlado. Sem `--dataset`, usa um
dataset sintético separável (apenas para validar o harness).

!!! warning "WavLM/HuBERT e backbone SSL"
    O benchmark registra o ambiente em `results.json`. Se `transformers` ou o
    backbone compatível não estiverem disponíveis, **WavLM** e **HuBERT** rodam
    com fallback simplificado em TensorFlow. Use essa condição apenas para
    validar o pipeline; para comparar qualidade de modelo no TCC, registre
    explicitamente se o backbone SSL real ou o fallback foi usado.

## O que é medido

| Dimensão | Métricas |
|---|---|
| Desempenho (teste limpo) | acurácia, precisão, recall, F1, **EER**, **AUC-ROC**, **min-tDCF** |
| Eficiência | nº de parâmetros, tamanho em disco (MB), **latência** (ms/amostra, mediana) |
| Robustez | acurácia/EER/AUC sob **AWGN** em cada SNR (`--snr`) |
| Convergência | flag por arquitetura (AUC ≥ limiar) + curva de validação |
| API (`--api`) | status + latência por endpoint (lê a superfície OpenAPI real) |

O ruído AWGN é aplicado no **espaço de entrada** do modelo (forma de onda para
raw-audio; espectrograma para os demais) — escolha deliberada para um teste
uniforme e reprodutível em todas as arquiteturas, documentada no relatório.

## Saídas → mapeamento para as tabelas/figuras do TCC

Os artefatos de análise são gravados em `--out` (default
`results/benchmark/`). Os pesos/configs treinados ficam em `app/models/` por
default para serem reutilizados diretamente pela interface e pela API:

| Arquivo | Uso no TCC |
|---|---|
| `tables/tab_resultados.tex` | **Tabela "Desempenho das arquiteturas"** (acur/EER/AUC/min-tDCF/lat/conv) |
| `tables/tab_eficiencia.tex` | **Tabela "Eficiência computacional"** (params/MB/latência) |
| `tables/tab_robustez.tex` | **Tabela "Robustez sob ruído AWGN"** (acur/EER por SNR) |
| `dataset.md` / `dataset_manifest.json` | Composição, origem, split, processamento e hiperparâmetros globais do dataset |
| `benchmark_plan.md` / `benchmark_plan.json` | Preset e hiperparâmetros efetivos antes do treino |
| `tcc_report.md` | Relatório Markdown com dataset, hiperparâmetros, métricas, inferências e imagens PNG |
| `figures/roc.png` | Curvas ROC (visualiza a AUC) |
| `figures/confusion_matrices.png` | Matrizes de confusão agregadas |
| `figures/score_distributions.png` | Distribuição dos scores por classe |
| `figures/robustez.png` | Acurácia × SNR (degradação sob ruído) |
| `figures/eficiencia.png` | Latência × acurácia (verde=convergiu) |
| `figures/convergencia.png` | Curvas de acurácia de validação por época |
| `app/models/bench_*` | Modelos e configs default carregados pela Gradio/API |
| `app/models/benchmark_final/<modelo>/` | Cópia completa do modelo final por arquitetura |
| `app/models/benchmark_final_manifest.json` | Manifesto dos modelos finais consolidados |
| `architectures/<modelo>/models/*` | Cópia do modelo dentro da execução original do benchmark |
| `architectures/<modelo>/hyperparameter_tuning.json` | Configuração do GridSearchCV, melhor score e melhores hiperparâmetros |
| `architectures/<modelo>/hyperparameter_tuning.csv` | Todos os candidatos avaliados no tuning, score médio e ranking |
| `architectures/<modelo>/*.json/csv/md/png` | Métricas, predições, robustez, resumo e figuras individuais |
| `results.csv` / `results.json` | Dados brutos (reprodutibilidade / anexos) |
| `summary.md` | Resumo legível (ambiente, dataset, tabela-resumo, API) |
| `tcc_overleaf/main.tex` | Artigo consolidado para Overleaf |
| `tcc_overleaf/figures/*.png` | Figuras finais referenciadas pelo artigo |
| `tcc_overleaf.zip` | Pacote limpo para upload no Overleaf |

Cada arquitetura possui uma pasta própria em `architectures/<modelo>/`.
Exemplo para SVM:

```text
architectures/svm/
├── metrics.json
├── predictions_clean.csv
├── robustness.csv
├── hyperparameter_tuning.json
├── hyperparameter_tuning.csv
├── summary.md
├── confusion_matrix.png
├── roc.png
├── score_distribution.png
├── convergence.png
└── models/
    └── bench_svm.pkl
```

As tabelas `.tex` usam `\singlespacing`, decimais com vírgula e as cores
`successgreen`/`dangerred` — **basta `\input{}`** no documento (o preâmbulo do
TCC já define esses pacotes/cores).

## Estrutura final para apresentação e builds

Para uma demonstração com Gradio/API, o diretório de modelos padrão é:

```text
app/models/
├── bench_aasist.keras
├── bench_aasist_config.json
├── bench_conformer.keras
├── bench_conformer_config.json
├── ...
├── bench_svm.pkl
├── bench_svm_config.json
├── benchmark_final_manifest.json
└── benchmark_final/
    ├── aasist/
    ├── conformer/
    ├── hubert_original/
    ├── wavlm_original/
    └── ...
```

`DetectionService`, `TrainingService`, Gradio e API usam `app/models/` como
default. Em builds Docker/Hugging Face, preserve esse diretório ou configure
`MODELS_DIR`, `DEEPFAKE_MODELS_DIR` ou `XFAKE_MODELS_DIR` apontando para uma
pasta persistente equivalente.

O pacote do artigo fica separado dos modelos:

```text
tcc_overleaf/
├── main.tex
├── README_OVERLEAF.md
└── figures/
    ├── benchmark_accuracy_auc.png
    ├── benchmark_eer.png
    ├── benchmark_latency.png
    ├── benchmark_robustness.png
    ├── benchmark_size.png
    ├── training_stability.png
    └── confusion_matrices/
```

O arquivo `tcc_overleaf.zip` deve conter somente `main.tex`, README e figuras,
sem PDF, `.aux`, `.log`, `.out` ou `.toc`.

## Reprodutibilidade

`results.json` registra o **ambiente** (SO, Python, TensorFlow, GPU/CPU,
dispositivo), a **configuração** completa (sementes, épocas, SNRs) e o
**balanceamento** do conjunto de teste — anexe-o para garantir reprodutibilidade.
