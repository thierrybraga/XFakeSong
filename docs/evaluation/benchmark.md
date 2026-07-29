# 15 — Sistema de Benchmark, Modelos Treinados e Resultados

> ## ⚠️ O dataset canônico mudou em 26/07/2026
>
> O artefato vigente é **`data/datasets/benchmark_dataset.npz`** — CETUC
> pareado com clones XTTS-v2, disjunção dupla locutor × frase, janela de 3 s.
> Ver [Protocolo de Dataset](../data/dataset-protocol.md) e
> [Dataset do Benchmark](../data/benchmark-dataset.md).
>
> Todos os NPZ `benchmark_audio_raw_balanced_15k*` foram **apagados do disco**.
> Os resultados consolidados reportados nesta página foram obtidos sobre o
> artefato anterior, que tinha atalho de fonte de 87,6% e disjunção de falante vácua:
> eles medem desempenho *in-domain com atalho disponível* e **não** devem ser
> comparados com a literatura nem com execuções sobre o dataset atual. **Requerem
> retreino.** Os comandos abaixo já apontam para o dataset atual.
>
> O modo `--academic-protocol` recusa artefatos sem `cluster_ids`/`source_ids` ou
> com atalho fonte-rótulo acima do limite.


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

!!! danger "Artefatos apagados em 2026-07-28 — esta seção é histórica"
    `data/models/` e `data/results/benchmark_academic_v2/` foram **removidos do
    disco e do índice do git** antes da nova bateria. Os números, caminhos e
    contagens desta seção e da de [Resultados numéricos](#resultados-numericos-usados-no-artigo)
    descrevem execuções sobre `benchmark_audio_raw_balanced_15k_academic_v2` —
    o dataset anterior, que tinha atalho de fonte de 87,6% e disjunção de
    falante vácua, e que também já não existe.

    Ficam como registro do que foi feito, **não** como estado atual nem como
    referência comparável. O material do artigo em `data/results/paper/`
    continua versionado e será regenerado pela nova bateria.

O material acadêmico foi consolidado em uma única fonte LaTeX em
`data/results/paper/`:

| Artefato | Caminho |
|---|---|
| Fonte principal do artigo | `data/results/paper/main.tex` |
| Manifesto de geração e hashes | `data/results/paper/paper_build_manifest.json` |
| Figuras usadas no artigo | `data/results/paper/figures/*.png` |
| Matrizes de confusão por arquitetura | `data/results/paper/figures/confusion_matrices/*.png` |
| Dataset do benchmark atual | `data/datasets/benchmark_dataset.npz` (o artigo consolidado ainda reflete o artefato anterior, apagado) |
| Modelos default da Gradio/API | `data/models/bench_*` |
| Modelos completos por arquitetura | `data/models/benchmark_final/<arquitetura>/` |
| Manifesto dos modelos consolidados | `data/models/registry.json` |
| Resultados e relatórios de benchmark | `data/results/<run>/` |

Não há PDFs versionados como fonte de verdade. O PDF deve ser gerado a partir de
`data/results/paper/main.tex` com `pdflatex` (qualquer instalação TeX Live completa).

A versão navegável da fundamentação e análise experimental está em
[Estudo Experimental](experimental-study.md), incluindo equações,
fluxograma, modelos, resultados, discussão, limitações e comandos de reprodução.

### Modelos treinados consolidados

Os diretórios finais em `data/models/benchmark_final/` preservam o artefato
completo de cada modelo promovido, incluindo backbones SSL quando aplicável.
No checkout atual `data/models/` está **vazio**: os artefatos anteriores foram
removidos em 2026-07-28. O `registry.json` e os diretórios por arquitetura são
recriados pela promoção, ao fim da nova bateria
(`scripts/reporting/sync_completed_benchmark_artifacts.py`).

### Incerteza e repetições (protocolo desde 2026-07-27)

Duas lacunas metodológicas foram fechadas:

- **A incerteza vai para as tabelas.** Os IC 95% de bootstrap (1000
  reamostragens, por *cluster* quando há IDs de proveniência) eram calculados e
  descartados na geração do LaTeX — o artigo publicava pontos secos. As tabelas
  de desempenho e de robustez agora trazem a incerteza, e a legenda declara
  **qual**: `±` desvio entre sementes quando há repetições, `[lo; hi]` do
  bootstrap quando a execução é única.
- **`n_seeds` repete o treino.** O bootstrap mede a variância de amostragem do
  **teste**; ele não diz nada sobre a variância de **treino** (inicialização,
  dropout, ordem de batch, realização do ruído). Com `n_seeds=N`, cada
  arquitetura é treinada N vezes e as métricas viram média ± desvio amostral.

Regras do protocolo de repetição:

| Item | Comportamento |
|---|---|
| Semente que varia | apenas a de **treino** (`seed`, `seed+1`, …) |
| Split | preso a `seed` — o teste selado é o mesmo em todas as repetições |
| Ruído de avaliação | preso a `seed` — mesmas realizações de AWGN |
| Artefato promovido | o da **primeira** semente, nunca o da melhor (escolher a melhor pelo teste seria seleção no conjunto de teste) |
| Auditoria | cada execução fica em `seed_runs` no JSON de resultados |

```bash
python scripts/benchmark/run_benchmark.py --full --dataset <npz>   # n_seeds=1
# 3 repetições (3x o tempo de GPU):
python -c "from benchmarks.config import BenchmarkConfig; ..."     # n_seeds=3
```

Com `n_seeds=1` o comportamento e o schema de saída são idênticos aos
anteriores — a agregação só entra quando há mais de uma execução.

### Rastreabilidade do run (desde 2026-07-27)

Cada `results.json` passa a carregar o que é necessário para reconstruir o
experimento:

| Bloco | Conteúdo |
|---|---|
| `environment.git` | `commit`, `branch` e **`dirty`** — se havia alterações não commitadas. Um run com árvore suja não é reproduzível só pelo commit, e o artefato precisa dizer isso |
| `environment.libraries` | TensorFlow, Keras, NumPy, SciPy, scikit-learn, librosa, torch, transformers |
| `environment.pretrained_checkpoints` | ids dos pesos externos que entram no grafo: AST (`MIT/ast-finetuned-audioset-…`), WavLM, HuBERT |
| `architectures.<nome>.provenance` | `variant`, `family`, `runner`, `scope` e `result_key` do manifesto |

Os rótulos `variant` descrevem a configuração **realmente treinada** e vivem em
`benchmarks/config.py`. Eles existiam antes, mas nunca chegavam a artefato
nenhum — e vários estavam defasados após mudanças de arquitetura
(`ast_vit_base_scratch` quando o AST já partia de pesos AudioSet;
`rawgat_st_multiply_stride4` com um stride que só vale nas variantes legadas;
um `rawnet2_paper_like` que não dizia **qual** RawNet2, sendo que verificação
de locutor e baseline anti-spoofing são arquiteturas diferentes de mesmo nome).

**Ao alterar uma arquitetura, atualize o rótulo junto** — o teste
`tests/unit/test_benchmark_provenance.py` falha se um rótulo voltar a
contradizer a implementação.

As 14 arquiteturas são cobertas em **dois escopos** (`benchmarks/config.py`):

| Escopo | Modelos | Como roda |
|---|---|---|
| `official` (default) | SVM, RandomForest, RawNet2, AASIST, RawGAT-ST, Conformer, Hybrid CNN-Transformer, SpectrogramTransformer, MultiscaleCNN | hiperparâmetros de `planning.py::NEURAL_BENCHMARK_HPARAMS` |
| `official` (SSL real) | WavLM Original, HuBERT Original | runner PyTorch separado (`scripts/benchmark/run_wavlm_original_benchmark.py`) — backbones SSL reais, não o fallback TF |
| `extended` | Sonic Sleuth, EfficientNet-LSTM, Ensemble | `--experiment-scope extended`, que força `optimize_hyperparameters=False` |
| `extended` (SSL Keras) | WavLM, HuBERT | mesmos checkpoints dos "Original", mas com o backbone **portado para Keras** (`ssl_backbone.py`) e congelado; treinam só a soma ponderada de camadas e a cabeça |

!!! warning "Proveniência dos SSL no caminho Keras"
    "WavLM"/"HuBERT" (escopo `extended`) e "WavLM Original"/"HuBERT Original"
    (escopo `official`) **não são o mesmo experimento**: mesmo checkpoint,
    runners diferentes. Até 2026-07-27 os dois primeiros não constavam de
    manifesto algum e o benchmark gravava `provenance: null` justamente nos
    modelos em que a proveniência é a definição do experimento.

    O caminho Keras **degrada** para uma CNN-1D treinada do zero se o checkpoint
    não estiver acessível. Quando isso acontece, o runner publica
    `provenance.variant = "*_fallback_cnn1d_scratch_nao_e_o_ssl_real"`,
    preserva o rótulo pretendido em `declared_variant` e registra
    `ssl_backbone.pretrained = false` com o motivo — nenhum artefato alega SSL
    real onde não houve. `XFAKE_STRICT_SSL=1` aborta em vez de degradar.

!!! danger "Timeout por modelo (2026-07-28)"
    `run_models_sequential.py --timeout-min` tinha default de **60 minutos** e
    `train_by_family.py`, de **240**. Ambos são menores que o treino de
    *qualquer* modelo neural no orçamento de 100 épocas — o run seria morto
    modelo a modelo, com status `timeout`.

    Omitido, o limite agora é **derivado por arquitetura** do custo medido em
    `benchmarks.planning.EXPECTED_TRAINING_HOURS`, com fator de segurança 3× e
    escala linear em épocas e tamanho do treino. Em GPU vai de ~1,2 h (Sonic
    Sleuth) a ~162 h (RawGAT-ST); em CPU, de ~22 h a ~4.078 h. Arquitetura
    desconhecida recebe o maior valor da tabela — errar para o lado de esperar
    demais, nunca de matar um treino de dias.

    Passar `--timeout-min` explicitamente continua vencendo. **Ao mudar lote,
    precisão, janela ou arquitetura, remeça os custos**: um timeout derivado de
    número velho mata um treino bom.

!!! danger "Janela de análise do log-mel (2026-07-28)"
    O salto entre quadros é imposto pelo contrato (`ceil(T / time_steps)`), mas
    a janela era a constante **512** — e as duas eram independentes no código.
    Com 100 quadros em 3 s o salto fica em 480: janelas consecutivas se
    sobrepunham em 32 amostras, e o taper de Hann é ~0 nas duas pontas.

    Medido: o envelope de soma-e-sobreposição ia de 1,0 a **exatamente 0** —
    **27% do sinal** caía em regiões de peso desprezível, invisíveis à análise.
    E um clique de 1 ms era **9× mais ou menos visível** conforme a posição em
    que caísse (razão mín/máx 0,11), sendo transiente justamente a pista de
    síntese que a tarefa procura.

    `benchmark_frontend.resolve_n_fft()` passa a **derivar a janela do salto**,
    garantindo 50% de sobreposição (1024 para o grupo de 100 quadros). A razão
    mín/máx sobe para 0,90 **sem perda de detecção média** — a 6% de
    sobreposição o fator dominante não era resolução temporal, era o ponto
    cego. Um `n_fft` declarado pela arquitetura continua vencendo: o AST
    especifica 25 ms (400 amostras) por definição do artigo.

    Afetava Conformer, CCT, MultiscaleCNN, Sonic Sleuth e EfficientNet-LSTM —
    cinco das doze arquiteturas. O AST não era afetado, o que lhe daria
    vantagem estrutural na comparação espectral.

    O contrato passa a gravar `n_fft` e `hop_length`, e a inferência os
    respeita: verificado `max|dif| = 0` entre o espectrograma de treino e o de
    produção nas duas configurações.

!!! success "Paridade treino↔produção (2026-07-28)"
    Os modelos do benchmark são os promovidos para produção, então o
    `input_contract` de cada artefato é gravado **no próprio treino**, com o
    `feature_frontend` derivado do `input_type` efetivamente usado
    (`benchmark_frontend.frontend_for_input_type`, fonte única). É esse campo
    que faz o `FeaturePreparer` reproduzir o front-end do benchmark na
    inferência; sem ele o app cai no front-end próprio (log-magnitude-mel,
    hop 128, sem z-score) e as métricas do artigo não se transferem.

    Antes, só AASIST e RawGAT-ST declaravam o campo e as outras dez dependiam
    do passo pós-hoc `rebuild_inference_contracts.py` — que cobria nove
    arquiteturas. SVM/RandomForest não tinham sidecar algum: iam para produção
    sem front-end e sem limiar, decidindo sempre em 0,5. Agora recebem contrato
    com o vetor tabular de 63 descritores e limiar de EER derivado da
    **validação**.

Pedir um modelo do escopo estendido dentro do escopo oficial é erro de
configuração (o preflight recusa antes de treinar), não uma limitação do
harness. Dentre os modelos executados, apenas os que completam um run entram no
diretório final via
`scripts/reporting/sync_completed_benchmark_artifacts.py`.

No topo de `data/models/`, ficam os arquivos carregáveis diretamente pela
interface e pela API (`.keras`/`.pkl`) e seus respectivos
`bench_*_config.json`. Alguns modelos suportados existem apenas nessa raiz no
checkout atual; a tabela diferencia os modelos finais do artigo dos artefatos
de demonstração/suporte.

| Modelo | Artefato principal | Diretório final |
|---|---|---|
| Random Forest | `data/models/bench_randomforest.pkl` | `data/models/benchmark_final/random_forest/` |
| SVM | `data/models/benchmark_final/svm/bench_svm.pkl` | `data/models/benchmark_final/svm/` |
| CCT (Hybrid CNN-Transformer) | `data/models/bench_hybrid_cnn_transformer.keras` | `data/models/benchmark_final/cct/` |
| AST (SpectrogramTransformer) | `data/models/bench_spectrogramtransformer.keras` | `data/models/benchmark_final/ast/` |
| Res2Net (MultiscaleCNN) | `data/models/bench_multiscalecnn.keras` | `data/models/benchmark_final/res2net/` |
| Conformer | `data/models/bench_conformer.keras` | `data/models/benchmark_final/conformer/` |
| RawNet2 | `data/models/bench_rawnet2.keras` | `data/models/benchmark_final/rawnet2/` |
| AASIST | `data/models/bench_aasist.keras` | `data/models/benchmark_final/aasist/` |
| RawGAT-ST | `data/models/bench_rawgat_st.keras` | `data/models/benchmark_final/rawgat_st/` |
| WavLM Original | `data/models/benchmark_final/wavlm_original/bench_wavlm_original.pt` | `data/models/benchmark_final/wavlm_original/` |
| HuBERT Original | `data/models/benchmark_final/hubert_original/bench_hubert_original.pt` | `data/models/benchmark_final/hubert_original/` |

Artefatos carregáveis pela Gradio/API que existem na raiz `data/models/` mas não
estão no recorte final sincronizado de `benchmark_final/` neste checkout:
`bench_efficientnet_lstm.keras`, `bench_sonic_sleuth.keras` e `bench_wavlm.keras`.

WavLM Original e HuBERT Original são artefatos PyTorch/SSL completos; por isso
ficam preservados no diretório completo com o backbone (`wavlm_backbone/` ou
`hubert_backbone/`). Os demais modelos Keras/sklearn têm cópia direta no topo de
`data/models/`.

### Publicação dos modelos no Hugging Face Hub

A fonte oficial para publicação é `data/models/`, pois é a mesma pasta usada
pela Gradio/API como default. Antes de enviar, verifique o plano de upload:

```bash
python scripts/ops/upload_models_to_hf.py \
    --repo-id SEU_USUARIO/xfakesong-models \
    --dry-run
```

Depois envie para um repositório do tipo **Model**:

```bash
python scripts/ops/upload_models_to_hf.py \
    --repo-id SEU_USUARIO/xfakesong-models \
    --private
```

O script usa `HF_TOKEN` ou `HUGGINGFACE_HUB_TOKEN`, cria o repositório quando
necessário e sobe os arquivos para `models/` no Hub. Use
`--include-paper` e `--include-results` apenas quando quiser anexar o pacote
do artigo e relatórios consolidados junto ao repositório de modelos.

### Resultados numéricos usados no artigo

> Os números desta seção vêm do **artefato anterior, já apagado do disco**
> (`benchmark_audio_raw_balanced_15k_confirmatory_v2.npz`): 15.000 amostras do
> tier `medium`, split estratificado 70/15/15, 2.250 de teste, 2.769,01 MiB. Eles
> são preservados como registro do run consolidado de 2026-07-15 e **serão
> substituídos** quando o retreino sobre o `benchmark_dataset.npz` for
> executado.

O run consolidado usou 15.000 amostras em PCM linear, 16 bits, mono e
16 kHz. Os resultados consolidados anteriores usaram orçamentos e parada
antecipada heterogêneos; o protocolo corrigido executa 100 épocas completas
para todas as redes e restaura o melhor checkpoint em validação limpa; SVM e RandomForest usam GridSearchCV + ajuste
final.

> **Fonte única dos números**: a tabela de resultados usada no artigo é
> gerada automaticamente em `data/results/paper/tabelas_benchmark.tex`
> (`Tabela~\ref{tab:resultados_consolidados}` de `main.tex`) a partir de
> `data/results/<run>/benchmark_summary.json`, via
> `python scripts/reporting/consolidate_results.py <runs...> --prefer-last --copy-to data/results/paper/figures`
> seguido de `python scripts/reporting/update_tcc_latex.py`. Não duplique esses valores
> aqui à mão — copie o retrato mais recente do artigo quando precisar de
> referência rápida, mas trate `tabelas_benchmark.tex` como a fonte de
> verdade. Recorte oficial do artigo (**11 modelos**, run final consolidado
> de 2026-07-15, `data/results/final_consolidated_20260715/`, teste limpo com
> test-lock; escopo **in-domain** — ver
> [Protocolo Final de ML](final-ml-protocol.md)):

| Modelo | Accuracy | AUC ROC | EER | Acc.\ @10dB |
|---|---:|---:|---:|---:|
| Conformer | 99,82% | 1,000 | 0,18% | 98,0% |
| HuBERT Original | 99,87% | 1,000 | 0,18% | 96,8% |
| Res2Net | 99,69% | 1,000 | 0,36% | 97,5% |
| WavLM Original | 99,69% | 1,000 | 0,36% | 98,8% |
| SVM | 99,24% | 1,000 | 0,58% | 93,8% |
| CCT | 99,20% | 0,999 | 0,71% | 95,0% |
| AST | 99,02% | 0,998 | 0,98% | 93,2% |
| Random Forest | 97,82% | 0,999 | 2,09% | 92,4% |
| RawNet2 | 97,16% | 0,998 | 2,71% | 93,0% |
| AASIST | 95,02% | 0,990 | 4,89% | 88,7% |
| RawGAT-ST | 93,60% | 0,987 | 6,22% | 84,0% |

Sonic Sleuth, Ensemble e EfficientNet-LSTM são suportados pelo harness (14
arquiteturas ao todo, ver seções abaixo) mas **não** integram o recorte
oficial dos 11 modelos do artigo — Sonic Sleuth por suspeita de vazamento de
dados não auditada (`scripts/dataset/audit_dataset_leakage.py`), e Ensemble/
EfficientNet-LSTM por estarem fora do escopo consolidado
(`docs/evaluation/retraining-adjustments.md`).

## Como rodar

```bash
# 1) Verificação do harness (sintético, 1 época) — segundos:
python scripts/benchmark/run_benchmark.py --quick

# 2) Pipeline completo do TCC: download, processamento, split, treino,
#    inferência, gráficos PNG e relatórios Markdown:
python scripts/benchmark/run_tcc_pipeline.py \
    --download \
    --target-per-class 7500 \
    --full-benchmark \
    --epochs 100 \
    --device-profile gpu \
    --out data/results/tcc_full_15k \
    --npz data/datasets/benchmark_dataset.npz

# 3) Execução do TCC direto no benchmark, usando dataset real .npz já exportado:
python scripts/benchmark/run_benchmark.py \
    --full \
    --dataset data/datasets/benchmark_dataset.npz \
    --epochs 100 \
    --device-profile gpu

# 4) Benchmark neural completo, sem SVM/RF:
python scripts/benchmark/run_benchmark.py \
    --neural \
    --dataset data/datasets/benchmark_dataset.npz \
    --epochs 100 \
    --device-profile gpu \
    --out data/results/bench_neural_tcc

# 5) Sob medida:
python scripts/benchmark/run_benchmark.py \
    --archs WavLM HuBERT RawNet2 "Sonic Sleuth" AASIST RawGAT-ST Conformer \
    "Hybrid CNN-Transformer" SpectrogramTransformer EfficientNet-LSTM \
    MultiscaleCNN Ensemble SVM RandomForest \
    --dataset data/datasets/benchmark_dataset.npz \
    --epochs 100 --snr 30 20 10 5 --api --out data/results/bench_tcc

# 6) Modelo individual:
python scripts/benchmark/run_benchmark.py \
    --model AASIST \
    --dataset data/datasets/benchmark_dataset.npz \
    --epochs 100 \
    --out data/results/bench_aasist
```

Por padrão, os relatórios, métricas, CSVs e figuras ficam em `data/results/benchmark/`
ou no diretório informado por `--out`. Os modelos treinados pelo benchmark ficam
em `data/models/`, o mesmo diretório usado pela Gradio/API para inferência. Use
`--models-dir outro/diretorio` ou `DEEPFAKE_MODELS_DIR` apenas quando quiser
isolar uma execução. Caminhos relativos são resolvidos a partir da raiz do
projeto, mesmo quando o comando é chamado de outro diretório.

Para usar os modelos treinados na demonstração visual, consulte
[Frontend Gradio](../interfaces/gradio.md). Para publicar modelos e demo,
consulte [GitHub Pages e Hugging Face](../deployment/github-pages-and-hugging-face.md).

### Execução por família de ambiente

Além do preset completo, o benchmark pode ser executado por família
computacional. Essa é a rota recomendada quando o objetivo é isolar dependências
e aproveitar GPU/CPU de forma controlada.

| Família | Preset | Comando |
|---|---|---|
| Tabular clássica | `classical.yaml` | `--family classical-tabular` |
| Espectral convolucional | `spectral_convolutional.yaml` | `--family spectral-convolutional` |
| Espectral attention | `tensorflow.yaml` | `--family spectral-attention` |
| Waveform end-to-end | `pytorch.yaml` | `--family waveform-end-to-end` |
| SSL pretrained | `ssl.yaml` | `--family ssl-pretrained` |
| Exploratório | `extended.yaml` | `--family extended` |
```bash
# Revisão sem iniciar treino
python scripts/training/train_by_family.py --family spectral-attention --plan-only

# Modelo individual dentro da família TensorFlow/Keras
python scripts/training/train_by_family.py --family spectral-attention \
    --models SpectrogramTransformer \
    --epochs 100 \
    --device-profile gpu \
    --out data/results/spectrogram_transformer_retrain

# Execução Docker com GPU para SSL
docker compose -f docker/compose/train.nvidia.yml run --rm ssl-transformers

# Benchmark completo em container NVIDIA
docker compose -f docker/compose/benchmark.nvidia.yml run --rm benchmark
```

Cada wrapper delega para `scripts/benchmark/run_models_sequential.py`; por isso o padrão
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
7. exportar `data/datasets/benchmark_dataset.npz`;
8. executar o preflight (`benchmark_plan.json`/`.md`) com preset, ambiente,
   dataset e hiperparâmetros efetivos;
9. treinar, inferir e gerar relatórios/gráficos para as arquiteturas do escopo
   selecionado (ver a tabela de escopos em "Modelos treinados consolidados").

### Catálogo de fontes usado no benchmark

O catálogo único de datasets fica em `app/domain/dataset_metadata/dataset_catalog.py` e é usado
pela aba Gradio **Datasets/Download**, pela documentação e pelo exportador
`scripts/benchmark/run_tcc_pipeline.py`. Ele registra, para cada fonte: tipo (`real`,
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

O dataset é montado por **tier** (ver [docs/data/public-datasets.md](../data/public-datasets.md)). O
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
python scripts/benchmark/run_tcc_pipeline.py --download --tier medium \
    --full-benchmark --epochs 100 --device-profile gpu \
    --out data/results/tcc_medium_15k \
    --npz data/datasets/benchmark_dataset.npz
```

O script `scripts/dataset/build_dataset.py` arquiva excedentes em
`data/datasets/overflow/` por padrão, em vez de apagar os WAVs brutos. Use
`--delete-excess` apenas quando o descarte destrutivo for intencional.

Comando completo recomendado:

```bash
python scripts/benchmark/run_tcc_pipeline.py \
    --download \
    --target-per-class 7500 \
    --full-benchmark \
    --epochs 100 \
    --device-profile gpu \
    --out data/results/tcc_full_15k \
    --npz data/datasets/benchmark_dataset.npz
```

Para um ensaio rápido do roteiro sem downloads:

```bash
python scripts/benchmark/run_tcc_pipeline.py \
    --smoke \
    --epochs 1 \
    --batch-size 4 \
    --latency-runs 1 \
    --out data/results/smoke_route
```

Para revisar tudo antes de iniciar o treinamento longo:

```bash
python scripts/benchmark/run_benchmark.py \
    --full \
    --dataset data/datasets/benchmark_dataset.npz \
    --epochs 100 \
    --out data/results/tcc_full_15k \
    --plan-only
```

Para revisar um modelo individual:

```bash
python scripts/benchmark/run_benchmark.py \
    --model RawNet2 \
    --dataset data/datasets/benchmark_dataset.npz \
    --out data/results/bench_rawnet2 \
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
`benchmark_dataset.npz`. Use `--skip-download` quando os WAVs e
splits já estiverem prontos localmente.

## Preset e hiperparâmetros pré-treino

O preset oficial é `full_tcc`:

- arquiteturas: WavLM, HuBERT, RawNet2, Sonic Sleuth, AASIST, RawGAT-ST,
  Conformer, Hybrid CNN-Transformer, SpectrogramTransformer,
  EfficientNet-LSTM, MultiscaleCNN, Ensemble, SVM e RandomForest;
- dataset: `.npz` balanceado exportado do split 70/15/15;
- orçamento comum: 100 épocas completas para todas as redes e cabeças SSL;
- seleção uniforme: melhor checkpoint pela menor val_loss limpa;
- robustez: AWGN em `30`, `20` e `10` dB;
- latência: mediana de `30` execuções por arquitetura;
- API: probe habilitado no preset completo.

Para treinar apenas os modelos neurais, use o preset `neural_tcc`
(`--neural` ou `--preset neural_tcc`). No `run_benchmark.py` esse preset cobre
as arquiteturas neurais Keras do manifesto oficial direto
(`Hybrid CNN-Transformer`, `SpectrogramTransformer`, `MultiscaleCNN`,
`Conformer`, `RawNet2`, `AASIST` e `RawGAT-ST`). WavLM Original e HuBERT
Original são executados pelo runner SSL dedicado usado pelo orquestrador
sequencial, enquanto SVM/RF ficam reservados ao baseline clássico.

Antes do treino, o preflight aplica hiperparâmetros recomendados por
arquitetura e adapta o `batch_size` ao perfil de dispositivo:

- `--device-profile auto`: usa GPU se o TensorFlow detectar CUDA, senão CPU;
- `--device-profile cpu`: limita batches de modelos pesados para evitar OOM/RAM;
- `--device-profile gpu`: habilita caps conservadores de VRAM e mixed precision
  quando a arquitetura permite;
- `--no-optimize-hparams`: desliga recomendações por arquitetura e usa os
  valores globais `--epochs`/`--batch-size`.

Hiperparâmetros neurais efetivos do recorte principal:

| Arquitetura | Entrada | Batch | LR | Dropout | Regularização | Otimizador |
|---|---|---:|---:|---:|---:|---|
| RawNet2 | raw audio | 16 | 1e-4 | 0.30 | L2 1e-4 | Adam |
| AASIST | raw audio | 16 | 3e-4 | 0.20 | L2 2e-4 | AdamW |
| RawGAT-ST | raw audio | 16 | 5e-5 | 0.35 | L2 1e-3 | AdamW |
| Conformer | log-Mel | 32 | 1e-4 | 0.30 | wd 1e-4 | AdamW |
| CCT | log-Mel | 32 | 1e-3 | 0.20 | L2 1e-4 | AdamW |
| AST | log-Mel | 8 | 2e-5 | 0.25 | wd 5e-5 | AdamW |
| Res2Net | log-Mel | 32 (cap GPU) | 2e-3 | 0.50 | L2 5e-4 | AdamW |
| WavLM Original | raw audio/SSL | 128 | 1e-3 | 0.20 | wd 1e-4 | AdamW |
| HuBERT Original | raw audio/SSL | 128 | 1e-3 | 0.20 | wd 1e-4 | AdamW |

Esses valores permanecem específicos por modelo. Os controles comuns são:
100 épocas completas, early stopping desativado, restauração do checkpoint de
menor val_loss limpa, limiar 0,5, semente 42, uma cópia AWGN de treino processada em lotes de 64 formas de onda e
balanceada em 30/20/10 dB e AWGN de teste nesses mesmos níveis **mais 5 dB**
antes do frontend. Os três primeiros medem robustez em condição CASADA; 5 dB
fica deliberadamente fora do augmentation e é o único nível que mede
generalização a ruído — a tabela marca essa coluna com asterisco e o
`results.json` grava `noise_condition: matched|unseen` por nível.
SVM e Random Forest mantêm suas grades próprias de validação cruzada e não usam
o conceito de época.
O campo epochs do plano é um controle global e sobrescreve somente o orçamento
de iterações, nunca os hiperparâmetros customizados. O preset acadêmico usa
100; valores menores são destinados apenas a smoke tests ou pilotos e não
devem alimentar as tabelas finais.

O pipeline completo chama esse preflight automaticamente antes de iniciar o
benchmark. Use `--skip-benchmark-preflight` apenas para depuração local.

## Benchmark de modelo individual

Use `--model` quando quiser treinar e avaliar apenas uma arquitetura. O script
continua gerando os mesmos artefatos (`results.json`, `tcc_report.md`,
`figures/*.png`, `architectures/<modelo>/*.png`), mas restritos ao modelo
selecionado.

```bash
python scripts/benchmark/run_tcc_pipeline.py \
    --skip-download \
    --skip-preprocess \
    --model SpectrogramTransformer \
    --npz data/datasets/benchmark_dataset.npz \
    --out data/results/bench_spectrogram_transformer
```

Para múltiplos modelos, mantenha `--archs`. `--model` e `--archs` são
mutuamente exclusivos.

## Execução sequencial com timeout

Para rodar todas as arquiteturas de forma resiliente, use o orquestrador
sequencial. Ele executa um modelo por vez, cria uma subpasta por modelo, grava
`run.log`, aplica timeout e permite retomar com `--resume`.

No modo acadêmico (padrão), o runner exige `train/val/test` predefinidos e um
selo SHA-256 criado **antes** do primeiro treino. O teste legado não deve ser
selado: regenere um novo NPZ intocado e então execute:

```bash
python scripts/dataset/freeze_benchmark_test.py \
    --dataset data/datasets/benchmark_dataset.npz \
    --declare-untouched
```
```bash
python scripts/benchmark/run_models_sequential.py \
    --dataset data/datasets/benchmark_dataset.npz \
    --test-lock data/datasets/benchmark_dataset.npz.test-lock.json \
    --out data/results/sequential_15k \
    --device-profile gpu \
    --timeout-min 90
```

Rodar somente os modelos neurais:

```bash
python scripts/benchmark/run_models_sequential.py \
    --neural-only \
    --dataset data/datasets/benchmark_dataset.npz \
    --test-lock data/datasets/benchmark_dataset.npz.test-lock.json \
    --out data/results/sequential_neural_15k \
    --device-profile gpu \
    --timeout-min 90
```

Revisar planos neurais antes do treino:

```bash
python scripts/benchmark/run_models_sequential.py \
    --neural-only \
    --plan-only \
    --dataset data/datasets/benchmark_dataset.npz \
    --test-lock data/datasets/benchmark_dataset.npz.test-lock.json \
    --out data/results/sequential_neural_plan \
    --device-profile cpu
```

No `plan-only` com `.npz` real, cada subprocesso carrega o dataset para
registrar forma, balanço e metadados. Em datasets grandes, espere alguns
segundos por modelo mesmo sem iniciar treino.

Retomar somente modelos pendentes:

```bash
python scripts/benchmark/run_models_sequential.py \
    --dataset data/datasets/benchmark_dataset.npz \
    --test-lock data/datasets/benchmark_dataset.npz.test-lock.json \
    --out data/results/sequential_15k \
    --device-profile gpu \
    --timeout-min 90 \
    --resume
```

Executar um subconjunto:

```bash
python scripts/benchmark/run_models_sequential.py \
    --models AASIST RawNet2 Conformer \
    --dataset data/datasets/benchmark_dataset.npz \
    --test-lock data/datasets/benchmark_dataset.npz.test-lock.json \
    --out data/results/sequential_neural_subset
```

No Windows nativo com TensorFlow 2.11+, CUDA não é exposto ao TensorFlow. Para
`--device-profile gpu`, rode esse script dentro do WSL2/Linux com
`tensorflow[and-cuda]` instalado.

Por padrão, o CLI do benchmark imprime apenas o resumo final e avisos
importantes. Use `--verbose` para depuração detalhada de treino, registry,
factories e salvamento de modelos.

No protocolo acadêmico, o `.npz` deve conter obrigatoriamente
`X_train/y_train`, `X_val/y_val` e `X_test/y_test`; o harness preserva essas
partições e aborta se o selo não corresponder ao arquivo. Fora do modo acadêmico,
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

O protocolo de retreino aplica AWGN exclusivamente à **forma de onda canônica**,
após a divisão treino/validação/teste e antes de qualquer frontend. A mesma
realização ruidosa é então convertida para raw-audio, log-Mel ou o vetor tabular
de 63 descritores. Uma cópia ruidosa por amostra de treino distribui, de forma
balanceada e reprodutível, os SNRs de 30, 20 e 10 dB. O modo estrito rejeita
NPZs reais que contenham somente features, evitando regressão silenciosa para o
protocolo legado no espaço de entrada. O loader preserva partições explícitas
e audita duplicatas binariamente idênticas entre elas com BLAKE2b, abortando
em caso de sobreposição.

A auditoria de proveniência do corpus atual encontra as 4 fontes e os 73
identificadores disponíveis em treino, validação e teste. Esse fato deve ser
reportado como limitação do benchmark in-domain; identificadores agregados
impedem garantir disjunção retrospectiva por pessoa para todo o corpus.

## Saídas → mapeamento para as tabelas/figuras do TCC

Os artefatos de análise são gravados em `--out` (default
`data/results/benchmark/`). Os pesos/configs treinados ficam em `data/models/` por
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
| `data/models/bench_*` | Modelos e configs default carregados pela Gradio/API |
| `data/models/benchmark_final/<modelo>/` | Cópia completa do modelo final por arquitetura |
| `data/models/registry.json` | Manifesto dos modelos finais consolidados |
| `architectures/<modelo>/models/*` | Cópia do modelo dentro da execução original do benchmark |
| `architectures/<modelo>/hyperparameter_tuning.json` | Configuração do GridSearchCV, melhor score e melhores hiperparâmetros |
| `architectures/<modelo>/hyperparameter_tuning.csv` | Todos os candidatos avaliados no tuning, score médio e ranking |
| `architectures/<modelo>/*.json/csv/md/png` | Métricas, predições, robustez, resumo e figuras individuais |
| `results.csv` / `results.json` | Dados brutos (reprodutibilidade / anexos) |
| `summary.md` | Resumo legível (ambiente, dataset, tabela-resumo, API) |
| `data/results/paper/main.tex` | Artigo consolidado (fonte LaTeX) |
| `data/results/paper/figures/*.png` | Figuras finais referenciadas pelo artigo |
| `data/results/paper/paper_build_manifest.json` | Proveniência e hashes do artigo gerado |

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
data/models/
├── bench_aasist.keras
├── bench_aasist_config.json
├── bench_conformer.keras
├── bench_conformer_config.json
├── ...
├── bench_svm.pkl
├── bench_svm_config.json
├── registry.json
└── benchmark_final/
    ├── aasist/
    ├── conformer/
    ├── hubert_original/
    ├── wavlm_original/
    └── ...
```

`DetectionService`, `TrainingService`, Gradio e API usam `data/models/` como
default. Em builds Docker/Hugging Face, preserve esse diretório ou configure
`MODELS_DIR`, `DEEPFAKE_MODELS_DIR` ou `XFAKE_MODELS_DIR` apontando para uma
pasta persistente equivalente.

O pacote ativo do artigo fica separado dos modelos:

    data/results/paper/
    ├── main.tex
    ├── tabelas_benchmark.tex
    ├── paper_build_manifest.json
    ├── main.pdf
    └── figures/
        ├── benchmark_accuracy_auc.png
        ├── benchmark_eer.png
        ├── benchmark_latency.png
        ├── benchmark_robustness.png
        ├── benchmark_size.png
        ├── training_stability.png
        └── confusion_matrices/

Após o retreino, gere todo o material com um único comando:

    python scripts/reporting/build_paper_from_benchmark.py data/results/retrain_waveform_awgn

O comando rejeita modelos ausentes, histórico neural diferente de 100 épocas,
SNRs incompletos, AWGN fora da forma de onda, scores desalinhados ao teste ou
artefatos CSV/JSON ausentes. Em seguida, consolida os resultados, copia figuras,
gera tabelas_benchmark.tex, valida sua inclusão no main.tex, compila o PDF e
grava hashes SHA-256 em paper_build_manifest.json.

## Reprodutibilidade

`results.json` registra o **ambiente** (SO, Python, TensorFlow, GPU/CPU,
dispositivo), a **configuração** completa (sementes, épocas, SNRs) e o
**balanceamento** do conjunto de teste — anexe-o para garantir reprodutibilidade.
