# Retreino com Ajustes — pós `clean_benchmark_full_20260626`

> **Documento histórico (trilha de auditoria).** Registra diagnósticos,
> planos e retreinos intermediários em ordem cronológica; métricas citadas em
> seções antigas foram **supersedidas**. A metodologia e os números válidos
> da versão final estão em [final-ml-protocol.md](final-ml-protocol.md)
> (fonte: `data/results/final_consolidated_20260715/`).

Documento de rastreio dos ajustes de hiperparâmetros aplicados após o
diagnóstico do benchmark completo do harness (14 arquiteturas suportadas; 11
modelos no recorte consolidado atual). Os ajustes estão **aplicados no
código**. Os 4 modelos do escopo oficial do TCC que precisavam de retreino
(RawGAT-ST, AASIST, WavLM Original, HuBERT Original) foram **retreinados e
promovidos em 2026-07-02** — ver
["Retreino de 2026-07-02 — concluído"](#retreino-2026-07-02)
abaixo. Ensemble e EfficientNet-LSTM não fazem parte da tabela consolidada do
TCC (11 modelos) e seu retreino permanece pendente. RandomForest e SVM também
não foram retreinados com os novos grids regularizados (`random_forest.py`,
`svm.py`); os números atuais no TCC vêm do run anterior ao ajuste — a
robustez fraca sob ruído (RandomForest 68,04%, SVM 66,44% @10dB — os dois
piores do recorte atual de 11 modelos) já é discutida no texto como
limitação estrutural do vetor tabular, não como defeito de treino a corrigir,
mas o retreino com os grids ajustados ainda não foi feito e poderia mudar
esses números (overfitting diagnosticado via `mean_train_score=1.0` no
tuning do RandomForest; grid do SVM permitia C=100/gamma=1/kernel `poly`,
mesmo padrão de baixa regularização).

> **Pendência não documentada (2026-07-04):** o commit `07a654d` corrigiu um
> bug de aliasing no `SincConvLayer` (argumento do sinc `2·f·n` → `π·f̂·n`,
> que degenerava os filtros passa-banda em banco quase aleatório) usado por
> **AASIST e RawGAT-ST**. Esse fix veio *depois* do retreino de 2026-07-02
> descrito abaixo — ou seja, os modelos atualmente promovidos em
> `benchmark_final/aasist` e `benchmark_final/rawgat_st` ainda foram
> treinados com o bug. A própria mensagem do commit recomenda retreino;
> ainda não foi executado (nenhum `data/results/` mais novo que
> `tcc_consolidated_20260702` além de uma rodada de XAI).

## Resumo do diagnóstico

| Modelo | Sintoma observado | Evidência |
| --- | --- | --- |
| RawGAT-ST | Overfitting/divergência | `val_acc` cai após época 4; `val_loss` 0.39→1.85; pior limpo (0.833) |
| AASIST | Subajuste | `val_acc` travada ~0.92; recall colapsa a 0.29 @10dB |
| Ensemble | Colapso de robustez | acc 0.50 e recall ~0 @10dB (prediz tudo "real") |
| RandomForest | Overfitting | `mean_train_score=1.0` no tuning; robustez 0.98→0.68 |
| SVM | Overfitting (grid permitia C=100/gamma=1/poly) | robustez 0.966→0.672 @10dB (2º pior do recorte atual) |
| Hybrid CNN-Transformer | Robustez moderada | 0.973→0.785 @10dB |
| EfficientNet-LSTM | Acurácia limpa baixa | 0.929 (robusto, mas baixo) |
| MultiscaleCNN | Instabilidade de treino | `val_loss=NaN` épocas 4–8 (recuperou) |

Mantidos sem alteração (sólidos e robustos): **Conformer, HuBERT,
SpectrogramTransformer, WavLM, RawNet2**.
**Sonic Sleuth** (1.0 perfeito) — auditar vazamento antes de confiar
(`scripts/dataset/audit_dataset_leakage.py`), não retreinado por ora.

## Ajustes aplicados

| Modelo | Arquivo | Mudança |
| --- | --- | --- |
| RawGAT-ST | `registry.py` | dropout 0.2→0.35; l2 5e-4→1e-3; clip 0.8→0.5; aug 0.3→0.4; patience 18→25 |
| RawGAT-ST | `rawgat_st.py` | LR 1e-4→5e-5; `global_clipnorm` 1.0→0.7 |
| AASIST | `registry.py` | l2 5e-4→2e-4; aug 0.25→0.35; patience 20→25 |
| AASIST | `aasist.py` | LR 1e-4→3e-4; `weight_decay` 0.01→1e-3 |
| Ensemble | `registry.py` | `augmentation_strength` 0.3→0.45; patience 15→20 |
| Hybrid CNN-Transformer | `registry.py` | dropout 0.1→0.2; stochastic-depth 0.1→0.15; aug 0.3→0.4 |
| EfficientNet-LSTM | `registry.py` | dropout 0.3→0.25; patience 15→20 |
| MultiscaleCNN | `multiscale_cnn.py` | Adam `clipnorm=1.0` (anti-NaN) |
| RandomForest | `random_forest.py` | grid regularizado: max_depth sem `None`/30; `min_samples_leaf` [2,4,8]; `min_samples_split` [5,10,20] |
| SVM | `svm.py` | grid regularizado: remove kernel `poly`; `C` teto 100→10; `gamma` teto 1→0.1 |

Augmentation de ruído no treino (controlada por `use_augmentation`/`snr_range_db`
= (5,40) em `app/core/config/settings.py`) já cobre a faixa de robustez avaliada
(10/20/30 dB) — é a principal alavanca para Ensemble, SVM, RandomForest e AASIST.
A calibração de temperatura e o threshold automático (`auto_calibrate_temperature`)
recalibram a decisão do Ensemble pós-treino.

## Como retreinar (na máquina com GPU)

```bash
# Linux / WSL2 / Docker GPU
bash scripts/training/retrain_ajustado.sh

# Windows
scripts\training\retrain_ajustado.bat
```

Roda apenas os 8 modelos ajustados (um por vez, `--resume`, 120 épocas, SNR
30/20/10) em `data/results/retune_ajustado_<data>/`. Os modelos não ajustados não são
retreinados.

## Verificação (antes de promover)

1. `python scripts/reporting/consolidate_results.py --results data/results/retune_ajustado_<data>`
2. `python scripts/reporting/validate_artifacts.py --results data/results/retune_ajustado_<data>`
3. Comparar `accuracy`/`f1`/`eer` e a curva de robustez (10 dB) com o baseline.
4. Só então sincronizar para `data/models/benchmark_final`:
   `python scripts/reporting/sync_completed_benchmark_artifacts.py --results data/results/retune_ajustado_<data>`

> Importante: promova um modelo só se ele melhorar (ou empatar) o baseline,
> especialmente a robustez a 10 dB. Caso contrário, mantenha o artefato anterior.

## Diagnóstico do retreino de 2026-06-30 (`official_retrain_selected_20260630`)

**Achado crítico — imagem Docker desatualizada.** O `benchmark_plan.json`
gravado pela execução do RawGAT-ST registra os hiperparâmetros **antigos**
(LR 1e-4, dropout 0.2, l2 1e-4, `use_augmentation: false`), apesar de o
repositório conter os valores ajustados desde 2026-06-21 (`ba37c6f`) e de
`optimize_hyperparameters: true`. A execução rodou na imagem
`xfakesong/benchmark:nvidia` construída antes dos ajustes — ou seja, **os
ajustes deste documento nunca foram aplicados de fato** nesse run. O sintoma
original se repetiu idêntico (val_loss mínima na época 6 subindo de 0.35 para
1.20 na época 31; recall 0.08 @10dB).

Checklist obrigatório antes de qualquer novo retreino via Docker:

1. `make build-nocache` (ou rebuild explícito da imagem de benchmark);
2. `python scripts/benchmark/run_benchmark.py --plan-only` e conferir no plano gravado
   os hparams ajustados (RawGAT-ST: LR 5e-5/dropout 0.35/l2 1e-3/aug on;
   AASIST: LR 3e-4/l2 2e-4/aug on);
3. conferir que o plano registra o split por falante quando `--speaker-split`
   for passado.

**Runner SSL corrigido (WavLM/HuBERT Original).** O
`scripts/benchmark/run_wavlm_original_benchmark.py` treinava a cabeça só com áudio
limpo (AWGN apenas na avaliação), rodava as 100 épocas sem early stopping
(val_loss mínima ~época 13) e decidia com threshold 0.5 sobre scores
descalibrados — robustez colapsava (recall ~0.08 @10dB; HuBERT 0.507 de
acurácia ≈ acaso). Correções aplicadas no runner (todas ligadas por default,
com flags `--no-*` para desligar):

- `--train-augmentation` / `--train-aug-snr 30 20 10 5`: anexa cópias do
  treino com AWGN (paridade com `classical_noise_augmentation` do caminho
  Keras);
- `--early-stopping` / `--early-stopping-patience 15`: monitora val_loss
  (val com ruído) e restaura os melhores pesos;
- `--calibrate-under-noise` / `--calibration-snr 20 10`: threshold de decisão
  no EER da validação com ruído (espelha `calibrate_under_noise` do
  `settings.py`); o threshold é persistido no `.pt`, no `_config.json` e nas
  métricas.

**Escopo do TCC pendente à época deste diagnóstico** (consolidado
`tcc_consolidated_20260701`): RawGAT-ST, AASIST, WavLM Original e HuBERT
Original — `bash scripts/training/retrain_ajustado.sh --tcc-pending`
(Windows: `scripts\training\retrain_ajustado.bat tcc-pending`). Conformer, Res2Net,
AST, RawNet2, CCT e os clássicos não precisam de retreino. **Concluído em
2026-07-02** — ver
["Retreino de 2026-07-02 — concluído"](#retreino-2026-07-02)
abaixo.

## Qualidade de treino (P1) — status

Ajustes de qualidade aplicados para o retreino dos novos modelos:

1. **Augmentation de ruído casada com o teste — JÁ ATIVO.** O benchmark treina
   redes com `use_augmentation` (ruído SNR + SpecAug) e os clássicos (SVM/RF)
   com `classical_noise_augmentation` (default True), anexando cópias do treino
   com AWGN nos SNRs avaliados (`benchmarks/runner.py`). A faixa global é
   `snr_range_db=(5,40)` (`settings.py`), cobrindo 10/20/30 dB.
2. **Calibração de temperatura/threshold sob ruído — APLICADO.** `settings.py`
   ganhou `calibrate_under_noise` (default True) + `calibration_snr_db=[20,10]`;
   o `ModelTrainer` (`_build_calibration_set`) calibra temperatura, EER e OOD
   num val que inclui cópias com AWGN — corrige o ponto de operação (ex.: colapso
   de recall do Ensemble) antes medido só em áudio limpo.
3. **Early stopping `val_loss` + `restore_best_weights` — JÁ ATIVO.** Callbacks do
   `ModelTrainer` usam `monitor="val_loss"`, `restore_best_weights=True` e
   paciência da config/registry (RawGAT-ST/AASIST já com paciência maior).
4. **Split disjunto por falante (tier `large`) — disponível, mas OFF por
   padrão nos scripts de retreino do TCC.** `run_models_sequential.py` tem
   `--speaker-split`/`--group-split` (repassados ao `run_benchmark.py`), mas
   `retrain_ajustado.sh/.bat` **não** passam mais `--speaker-split` por
   padrão desde 2026-07-02: o TCC documenta particionamento estratificado
   70/15/15 para os 11 modelos da tabela, e rodar um subconjunto com split
   por locutor gera um teste menor e desbalanceado, **não comparável** ao
   baseline dos demais modelos (confirmado empiricamente: RawGAT-ST caiu de
   n=2250 balanceado para n=863 com 525/338 quando testado com
   `--speaker-split`). Use `--with-speaker-split` (`.sh`) ou
   `with-speaker-split` (`.bat`) para o protocolo exploratório disjunto por
   locutor, fora da tabela oficial do TCC.

## Retreino de 2026-07-02 — concluído {#retreino-2026-07-02}

Escopo: RawGAT-ST, AASIST, WavLM Original e HuBERT Original — os 4 modelos
pendentes do consolidado `tcc_consolidated_20260701`. Execução em duas etapas
devido a dois problemas operacionais encontrados e corrigidos durante o
processo:

1. **AASIST estourou o timeout.** A primeira chamada de
   `run_models_sequential.py` não passou `--timeout-min`, herdando o default
   de 60 min; AASIST precisa de ~160 min para 120 épocas e foi interrompido
   sem salvar artefato. `retrain_ajustado.sh/.bat` passaram a fixar
   `--timeout-min 480`.
2. **RawGAT-ST rodou com `--speaker-split` na primeira tentativa**, gerando
   um resultado não comparável (ver item 4 acima). Corrigido relançando
   RawGAT-ST e AASIST sem `--speaker-split`, com o mesmo particionamento
   estratificado 70/15/15 do restante da tabela.

Resultado final (mesmo protocolo dos demais 7 modelos, `n=2250` balanceado no
teste; `data/results/retune_ajustado_20260701_2051/` para WavLM/HuBERT Original,
`data/results/retune_ajustado_fix_20260701_2303/` para RawGAT-ST/AASIST,
consolidado em `data/results/tcc_consolidated_20260702/`):

| Modelo | Acc. limpa (antes → depois) | EER (antes → depois) | Acc. @10dB (antes → depois) |
| --- | ---: | ---: | ---: |
| AASIST | 91,69% → 92,49% | 8,31% → 7,42% | 71,38% → 88,93% |
| RawGAT-ST | 83,56% → 86,98% | 16,27% → 12,80% | 53,20% → 82,93% |
| HuBERT Original | 90,18% → 88,76% | 9,73% → 11,29% | 50,67% → 80,98% |
| WavLM Original | 86,09% → 84,67% | 13,47% → 15,24% | 52,58% → 75,91% |

Todos os 4 modelos convergiram (`converged: True`) e ganharam robustez
substancial a 10 dB, à custa de perda marginal de acurácia/EER no conjunto
limpo (WavLM e HuBERT) — *trade-off* esperado ao expor o classificador a
ruído no treino. Artefatos sincronizados para `data/models/benchmark_final/`
(11/11 modelos, `python scripts/reporting/sync_completed_benchmark_artifacts.py`) e
tabelas/figuras do TCC regeneradas (`python scripts/reporting/update_tcc_latex.py`).
`data/results/paper/main.tex` (Seção 5 — Análise dos Resultados — e Conclusão)
reescrito para refletir os números corrigidos; a narrativa de robustez a
ruído deixou de apontar RawGAT-ST/SSL como os mais frágeis e passou a
identificar SVM/Random Forest como os modelos menos robustos do conjunto.

## Retreino de 2026-07-06/07 — SincConv, CCT e grids regularizados

Escopo: RawGAT-ST, AASIST, CCT (Hybrid CNN-Transformer), Random Forest e SVM —
os 5 modelos com pendência de retreino identificados na revisão técnica de
2026-07-04 (`07a654d`, bug de aliasing no `SincConvLayer`) e nesta sessão
(grid regularizado do SVM, análogo ao já aplicado ao Random Forest). Ensemble
e EfficientNet-LSTM foram *tentados* no mesmo run, mas **falharam por design**:
`benchmarks/planning.py::_base_recommended_hparams` levanta `ValueError` para
qualquer arquitetura fora do recorte oficial do artigo (9 neurais + SVM/RF) —
achado não documentado até então. Confirmado com o usuário: os dois
permanecem fora do escopo, sem alteração de código para reabri-los.

Execução: `data/results/retune_ajustado_20260706_1933/` (120 épocas, SNR 30/20/10,
`--models RawGAT-ST AASIST Ensemble "Hybrid CNN-Transformer" EfficientNet-LSTM
RandomForest SVM` via `run_models_sequential.py` em Docker/GPU RTX 3060).

| Modelo | Acc. limpa (antes → depois) | EER (antes → depois) | Acc. @10dB (antes → depois) |
| --- | ---: | ---: | ---: |
| RawGAT-ST | 86,98% → 92,76% | 12,80% → 7,16% | 82,93% → 83,96% |
| AASIST | 92,49% → 95,82% | 7,42% → 4,18% | 88,93% → 89,20% |
| CCT | 96,04% → 97,60% | 3,91% → 2,40% | 81,20% → 89,24% |
| Random Forest | 98,18% → 98,18% (idêntico) | 1,69% → 1,69% | 68,04% → 68,04% (idêntico) |
| SVM | 96,00% → 96,00% (idêntico) | 4,31% → 4,31% | 66,44% → 66,44% (idêntico) |

**Achado relevante:** Random Forest e SVM retornaram métricas **bit-a-bit
idênticas** às anteriores, apesar dos grids regularizados (`random_forest.py`,
`svm.py`). O ótimo por CV já caía dentro da faixa restrita em ambos os casos —
a fragilidade sob ruído desses dois classificadores **não é overfitting de
hiperparâmetro**, e sim limitação estrutural do vetor tabular de 63
descritores agregados (consistente com a discussão já presente no TCC). Já
RawGAT-ST/AASIST (fix do `SincConv`) e CCT (dropout/profundidade
estocástica/augmentation mais agressivos) melhoraram de forma real, tanto
limpo quanto sob ruído — CCT deixou de ser o pior espectral (81,20% @10dB) e
passou a ficar acima do RawGAT-ST (89,24% vs. 83,96%), invertendo uma
comparação que estava no texto do TCC.

**Achado de calibração (importante para produção):** os scores brutos do
AASIST retreinado saturam quase totalmente em 0/1 (apenas 4 valores distintos
em 2250 amostras de teste — cabeça AM-Softmax + `mixed_float16`), o que faz o
recálculo ingênuo de `eer_threshold` por `scripts/reporting/rebuild_inference_contracts.py`
(cruzamento FPR=FNR sobre `predictions_clean.csv`) cair num limiar degenerado
(`0.0`, que classificaria tudo como "fake" em produção). O próprio treino já
calcula uma calibração melhor — `temperature`/`ood_threshold`/`eer_threshold`
sob ruído (`calibrate_under_noise=True`) — gravada no config raiz
(`data/models/bench_<arch>_config.json`), mas esse script a descarta ao
reconstruir o sidecar promovido (ele só preserva `feature_frontend`/
`input_shape`, não foi projetado para saber de calibração). Corrigido
manualmente para RawGAT-ST/AASIST/CCT: os sidecars promovidos agora mesclam o
`feature_frontend` correto (do rebuild) com a calibração real do treino (do
config raiz). **Se `rebuild_inference_contracts.py` for rodado de novo para
esses 3 modelos, refazer essa mesclagem** — ou, melhor, ajustar o script para
preservar `temperature`/`ood_threshold`/`eer_threshold` do config raiz quando
ele for mais recente que o smoke-test que o script foi feito para substituir.

Consolidado em `data/results/tcc_consolidated_20260707/` (11 modelos: os 5 acima +
Conformer/AST/RawNet2/Res2Net/WavLM Original/HuBERT Original, inalterados),
sincronizado para `data/models/benchmark_final/` e `data/results/paper/tabelas_benchmark.tex`
regenerado. `main.tex` revisado (ranking de robustez, decomposição de erros,
McNemar SVM×CCT — que deixou de ser empate estatístico, p≈0,0019 — e
estabilidade de treinamento) para refletir os números novos; compilação via
`latexmk -pdf` validada sem erros.

---

## 2026-07-12 — protocolo canônico de AWGN na forma de onda

Uma auditoria do benchmark identificou que a rodada consolidada até 2026-07-07
aplicava ruído em domínios diferentes: forma de onda para modelos raw/SSL,
log-Mel para redes espectrais e vetor de 63 descritores para SVM/Random Forest.
Essas perturbações não são fisicamente equivalentes. Portanto, as comparações
de robustez e as atribuições causais registradas acima são **históricas e
provisórias**; não devem ser usadas como ranking entre famílias antes do novo
retreino.

O protocolo corrigido foi implementado com as seguintes garantias:

1. divisão treino/validação/teste antes de qualquer aumento;
2. AWGN aplicado apenas à forma de onda canônica de 5 s/16 kHz;
3. SNR realizada normalizada por amostra;
4. mesma semente e mesma realização ruidosa para todas as famílias;
5. uma cópia ruidosa por amostra de treino, balanceada em 30/20/10 dB;
6. validação limpa e limiar comum de 0,5;
7. orçamento uniforme de 100 épocas completas, sem early stopping;
8. restauração do melhor checkpoint por val_loss limpa para todas as redes;
9. preservação das partições explícitas do NPZ e auditoria BLAKE2b;
7. frontends raw, log-Mel e tabular executados somente após a perturbação;
8. falha explícita, no modo estrito, quando o NPZ contém somente features;
9. aumento interno em log-Mel/features desativado na comparação principal.

O retreino integral é necessário, pois a correção altera tanto os dados de
ajuste quanto a avaliação sob ruído. Execute na raiz do repositório:

    python scripts/benchmark/run_models_sequential.py --dataset data/datasets/benchmark_dataset.npz --test-lock data/datasets/benchmark_dataset.npz.test-lock.json --epochs 100 --device-profile gpu --seed 42 --snr 30 20 10 --train-aug-snr 30 20 10 --train-noise-copies 1 --waveform-noise-batch-size 64 --waveform-train-augmentation --timeout-min 120 --out data/results/retrain_waveform_awgn

Para retomada após interrupção, repita o comando com a opção --resume. Antes
de substituir tabelas e pesos promovidos, confirme em cada
benchmark_result.json que noise_protocol.evaluation_domain seja waveform,
noise_protocol.frontend_after_noise seja true e
noise_protocol.training_augmentation_domain seja waveform.

O orquestrador grava benchmark_protocol.json com os controles comuns e mantém
os hiperparâmetros específicos de cada arquitetura no benchmark_plan.json.

---

## 2026-08-02 — SIGSEGV do MultiscaleCNN sob mixed precision

No benchmark de 2026-08-01 o MultiscaleCNN morreu com `returncode = -11`
(SIGSEGV) aos 336 s, no primeiro batch da época 1, sem deixar artefato. Não era
OOM — o OOM-killer envia SIGKILL (-9) — e o auto-JIT do XLA já estava desligado
para esta arquitetura.

**Isolamento** (repro mínimo: log-mel 100×80, batch 32, RTX 3060):

| Condição | Resultado |
| --- | --- |
| `mixed_float16`, treino | **SIGSEGV** após completar o batch 0 |
| `float32`, treino | 5 batches limpos |
| `mixed_float16`, só forward | 6 passos limpos |
| `mixed_float16` + `TF_CUDNN_USE_AUTOTUNE=0` | **SIGSEGV** igual |

Conclusão: o crash é no **backward** em fp16, não no forward nem na escolha de
kernel do autotune. A suspeita é o gradiente do split/concat hierárquico de
canais do bloco `Bottle2neck` do Res2Net, o único padrão que esta arquitetura
tem e as outras não; o build de 2026-08-01 subiu `nvidia-cudnn-cu12` de
9.1.0.70 para 9.24.0.43.

**Ajuste**: `planning.py` ganhou `_MIXED_PRECISION_UNSAFE_ARCHITECTURES`
(`rawnet2`, `multiscalecnn`), substituindo o `elif compact != "rawnet2"` que
tratava o caso do RawNet2 de forma implícita. O mecanismo já existia e é o
mesmo do RawNet2 — a chave `use_mixed_precision` chega ao `ModelTrainer`, que
devolve a política global a float32. Custo: velocidade, não resultado.

**Verificação**: `run_benchmark.py --model MultiscaleCNN --epochs 2` sobre o
dataset de 15.000 completou as duas épocas e a avaliação, com exit 0 —
92,69% de acurácia, EER 5,50%, AUC 0,989.

Dois bugs latentes do caminho **raw-audio** apareceram na instrumentação e
foram corrigidos em `layers.py`: `STFTLayer` passava float16 para
`tf.signal.stft` (RFFT aceita só float32/64) e `LogMelFromMagnitudeLayer`
multiplicava magnitude float16 pela matriz mel float32. Nenhum dos dois afetava
o benchmark, que alimenta log-mel pronto (`input_domain: spectrogram`), mas
qualquer consumidor do caminho raw sob precisão mista quebrava na CONSTRUÇÃO do
modelo. O log-mel roda em float32 de propósito: com mel em fp16 o épsilon de
1e-6 sumiria e bins nulos virariam `-inf`.

---

## 2026-07-14 — revisão pós-retreino AWGN: AST, CCT, RawNet2, Res2Net e checkpoint guardado

Diagnóstico do retreino sob o protocolo canônico de AWGN (2026-07-12) apontou
4 arquiteturas degradadas e 1 bug sistêmico de seleção de checkpoint. Mapa
diagnóstico → ajuste (todos **aplicados no código**; retreino pendente):

| Modelo | Sintoma | Causa identificada | Ajuste aplicado |
| --- | --- | --- | --- |
| AST (SpectrogramTransformer) | Degrada lentamente até chute aleatório (EER final ~51%) mesmo após o fix de `decay_steps` | Blocos **post-LN** (LayerNorm depois do residual) — instáveis a 12 blocos treinados do zero; ViT/AST reais são pre-LN. LR de pico alto p/ 87M params do zero. Cabeça não-paper (2 blocos Dense 1024/256 c/ skips, ~1M params extras) ampliava sobreajuste | `spectrogram_transformer.py`: blocos **pre-LN** (`norm_style='pre'`; `'post'` mantido só p/ desserializar modelos antigos), cabeça do paper (LN→dropout→Dense), LR 5e-5→**1e-5**, weight_decay 1e-4→**1e-5** (sincronizado em `registry.py` e `planning.py`, que ia de 2e-5→1e-5) |
| CCT (Hybrid CNN-Transformer) | Colapsa sob ruído: EER 37% limpo → 53% @10dB; AUC 0,46 (< acaso) @10dB | Compile **hardcoded** (lr=1e-3, `decay_steps=50000`): mesmo mismatch de `decay_steps` do AST — com batch 32 são ~65.700 passos reais e o LR zerava na época ~76; o `learning_rate` do plano de benchmark nunca chegava ao modelo | `hybrid_cnn_transformer.py`: otimizador parametrizado (lr/warmup/decay/wd/alpha/clipnorm); LR de pico 1e-3→**3e-4**; `planning.py`: `decay_steps=65700`; `runner.py`: parâmetros roteados ao construtor (mesmo caminho do Conformer/AST) |
| RawNet2 | EER 32,7% limpo (baseline 2,89%); 100 épocas em ~130 min | Topologia divergia do paper: MaxPool(3) só após os blocos 2 e 4 → a GRU recebia **~590 passos** temporais (paper: ~7 com recorte de 1 s); FMS multiplicativo-puro (paliativo `x·2σ`) em vez do mul+add do paper | `rawnet2.py`: MaxPool(3) após **cada** bloco residual; `layers.py`: FMS com `scale_mode='mul_add'` (`x·y + y`, forma oficial; `'mul2'` mantido como default da camada só p/ compat com modelos salvos) |
| Res2Net (MultiscaleCNN) | Overfit severo (train 100% / val 64,5%) | **Nenhuma regularização efetiva**: o `dropout_rate=0.5` e o `l2_reg_strength` do plano eram config morto (nunca chegavam ao `create_model`; valia o dropout 0,2 do registry e Adam sem weight decay) | `multiscale_cnn.py`: Adam→**AdamW** com `weight_decay` real (default 1e-2 acoplado ao LR); `registry.py`: dropout 0,2→**0,5** (agora efetivo em todos os caminhos); `planning.py`: LR 2e-3→1e-3, chaves mortas removidas |
| Todos (bug sistêmico) | "Melhor checkpoint" (val_loss) pior que a última época; no Res2Net a restauração produziu **NaN** (EER 14,9%→50%) | `training_service.py` restaurava o checkpoint às cegas; um `load_weights` que falha no meio deixa o modelo meio-carregado, e o critério val_loss pode escolher época ruim | Restauração **guardada**: snapshot dos pesos → `load_weights` → reavalia `val_loss` no val; se não-finita ou pior que os pesos em memória, reverte o snapshot (`_guarded_checkpoint_restore`) |
| Todos (reprodutibilidade) | Política de precisão dependia da ordem das arquiteturas no processo | `use_mixed_precision=True` só era aplicado DEPOIS do `create_model` (camadas capturam o dtype na construção); o caso False já era tratado antes | `training_service.py`: política (`mixed_float16`/`float32`) definida **antes** da instanciação para qualquer valor explícito |

Sem ajuste (variância normal de treino, sem sinal de bug): **Conformer**
(EER 1,24% vs 0,27%, AUC 0,996), **AASIST** (12,0% vs 4,18%) e **RawGAT-ST**
(18,3% vs 7,16%) — beneficiam-se do checkpoint guardado e podem ser
reexecutados no mesmo run para nova amostra.

Retreino: mesmo comando do protocolo de 2026-07-12 (acima), com `--models
SpectrogramTransformer "Hybrid CNN-Transformer" RawNet2 MultiscaleCNN` no
mínimo. Antes de rodar via Docker, `make build-nocache` (ver checklist de
2026-06-30) e `--plan-only` para conferir no plano: AST lr 1e-5/wd 1e-5,
CCT lr 3e-4/decay 65700, Res2Net lr 1e-3/dropout 0.5.

### Revisão sistêmica adicional (mesma data)

Auditoria de ambientes, protocolo AWGN, benchmark e inferência:

1. **Dataset errado nos composes/presets (crítico).**
   `app/datasets/benchmark_audio_raw_balanced_15k.npz` é um **stub de smoke
   com 64 amostras de 1 s** criado em 2026-07-11 com o MESMO nome do dataset
   real (15k × 5 s, com `groups`/`speaker_ids`, em `data/datasets/`).
   `docker/compose/benchmark.nvidia.yml`, `docker-compose.benchmark.yml`,
   os 5 presets de `configs/training/*.yaml` (incluindo
   `retune_ajustado.yaml`!), o comando do CLAUDE.md e os READMEs de
   `docker/environments/*` apontavam para o stub — `make benchmark-nvidia`
   rodaria 100 épocas sobre 64 amostras sem nenhum erro visível. Todos
   corrigidos para `data/datasets/`. Os runs anteriores NÃO foram afetados
   (usaram `run_models_sequential.py`/`retrain_ajustado.sh`, cujos defaults
   já eram `data/datasets/`; ex.: AST registrou 2625 passos/época = 21000
   amostras). O stub permanece em `app/datasets/` — considerar renomear
   para algo inequívoco (ex.: `benchmark_smoke_64x1s.npz`).
2. **min t-DCF incomparável no runner SSL.** O
   `run_wavlm_original_benchmark.py` calculava EER/min-tDCF com uma fórmula
   própria simplificada (`p_target=0.01`), diferente do t-DCF ASVspoof2019
   CM-only do `MetricsCalculator` usado nos outros 9 modelos. Agora delega a
   `benchmarks.evaluate.evaluate_scores` (mesmas métricas, incl.
   `accuracy_at_eer`). **Os min-tDCF históricos de WavLM/HuBERT Original não
   são comparáveis aos demais** — recalcular a partir de
   `predictions_clean.csv` ou no próximo retreino.
3. **Calibração sob ruído fora do domínio do waveform.** O
   `ModelTrainer._build_calibration_set` adicionava AWGN no ESPAÇO DE ENTRADA
   do val — para modelos espectrais/tabulares treinados pelo app isso é ruído
   em log-mel/features (o domínio errado que o protocolo de 2026-07-12
   baniu). Agora a calibração ruidosa só ocorre quando o val é forma de onda
   ((N,T)/(N,T,1) com T≥1000); caso contrário calibra com val limpo. O
   benchmark não era afetado (já desativava `calibrate_under_noise`).
4. **Conferências sem achado:** AWGN por amostra com SNR realizado
   normalizado (`add_awgn`), mesma semente/realização de ruído entre famílias
   (`seed+20000+snr` idêntico no runner Keras e no SSL), limiar comum 0,5 no
   runner SSL (calibração é ablação opt-in), frontends raw/log-mel/tabular
   aplicados somente após o ruído, `evaluate_scores` com `accuracy_at_eer`,
   inferência com paridade de frontend via `feature_frontend` no
   input_contract + temperatura/EER/OOD do contrato.

### Revisão do processo de aplicação do AWGN (mesma data)

Auditoria da MATEMÁTICA e mecânica do ruído em todos os aplicadores, com
verificação numérica (SNR realizado vs alvo por amostra):

| Aplicador | Estado | Ação |
| --- | --- | --- |
| `benchmarks/data.py::add_awgn` (canônico) | SNR realizado EXATO (normaliza a potência realizada do ruído por amostra); silêncio → sem ruído, sem NaN | nenhum ajuste |
| `add_awgn_assigned`/`balanced_snr_assignments` | atribuição 3500/3500/3500 exata e reprodutível | nenhum ajuste |
| `ModelTrainer._add_awgn` (calibração) | calibrava só a potência ESPERADA (~0,4% de desvio) apesar de prometer paridade | normalização exata — agora byte-idêntico ao canônico com a mesma semente |
| `scripts/benchmark/robustness_test.py::add_awgn` | **divergente**: `np.clip(x+ruído, ±1)` distorcia o ruído (não-gaussiano, SNR realizado sobe a 10 dB), sem normalização realizada, RNG global sem semente, e o call-site geraria a MESMA realização p/ todas as amostras | delegado ao canônico, em lote, com a convenção de semente do benchmark (`seed+20000+snr`) |
| `AudioAugmenter._add_noise` (augmentation legado) | potência esperada (desvio <1% em 16k amostras) — aceitável p/ augmentation, coberto por teste (±0,5 dB) | mantido |
| runner SSL `_add_awgn_raw` | já delegava ao canônico | nenhum ajuste |

Invariantes verificados e documentados no código: espaços de semente
treino (`seed+10000+start+1009·nível`) e avaliação (`seed+20000+snr`)
disjuntos nos defaults (batch 64, SNRs 30/20/10) — comentário no runner
fixa o contrato. Caveat inerente ao protocolo (não é bug): o SNR alvo é
GLOBAL na janela de 5 s; no recorte central de 1 s das arquiteturas raw o
SNR local pode desviar (medido: ±0,1 dB em sinal estacionário; maior em
fala real onde a energia é não-uniforme) — é o comportamento padrão de
protocolos de SNR global.

### Melhorias de pipeline (mesma data)

1. **Shuffle por época no treino (afeta TODOS os retreinos).** O Keras ignora
   `shuffle=True` quando `x` é um `tf.data.Dataset`, e o caminho sem
   augmentation (o do benchmark) não embaralhava NADA: ordem de batches fixa
   em todas as épocas e, pior, o treino do protocolo AWGN é
   `[bloco limpo | bloco ruidoso]` concatenados — cada época via primeiro só
   amostras limpas e depois só ruidosas. `ModelTrainer._array_dataset` agora
   embaralha por época (buffer completo no caminho pequeno; permutação por
   passagem no caminho generator/streaming), somente no treino (val
   preservado). Isso muda a dinâmica de TODOS os próximos treinos — mais um
   motivo para retreinar antes de comparar com números antigos.
2. **Checkpoint weights-only.** O `ModelCheckpoint` salvava o MODELO COMPLETO
   (grafo + otimizador, ~3× os pesos; ~1 GB por melhoria de época no AST) a
   cada novo melhor val_loss. O benchmark agora grava
   `best_checkpoint.weights.h5` (só pesos) e o trainer decide
   `save_weights_only` pela extensão; a restauração guardada usa
   `load_weights`, que aceita ambos os formatos.
3. **Guarda de tamanho de dataset.** `run_benchmark` loga aviso destacado
   quando um NPZ real tem <1000 amostras (defesa contra o stub do item 1 da
   revisão sistêmica).
## 2026-07-15 (tarde) — correções pós-auditoria de fidelidade aos papers

Auditoria arquitetura-por-arquitetura (AASIST, RawGAT-ST, RawNet2, CCT,
WavLM/HuBERT Original, MultiscaleCNN) contra as referências bibliográficas,
com verificação de EER/Acc/robustez e do pipeline de ruído. Nenhuma camada,
bloco ou conexão de nenhum paper foi alterada — todas as correções abaixo são
serialização, hiperparâmetro-drift ou testes obsoletos.

**Retratação importante:** a análise anterior (mesma tarde) apontou o
multicrop TTA (3 janelas na avaliação) de AASIST/RawGAT-ST como uma possível
assimetria a corrigir. Investigação mais profunda encontrou testes dedicados
(`test_p2_rawgatst_sslaasist.py`, `test_detection_model_loader_predictor.py`)
provando que é uma funcionalidade **deliberada e já testada** (janela 64.600
amostras ≈ 4,04s — o mesmo comprimento usado pelos baselines oficiais do
ASVspoof2021 para RawNet2/AASIST/RawGAT-ST — com average de 3 crops
início/centro/fim), cabeada tanto no benchmark quanto na inferência de
produção (`Predictor`). **Não foi alterada.** Recomendação anterior de
"padronizar/remover" está revogada.

### Correções aplicadas

1. **Bug de produção — Lambda não-serializável (AASIST, RawGAT-ST,
   MultiscaleCNN).** `_build_paper_aasist`/`_build_paper_rawgat` usavam
   `layers.Lambda(lambda v: tf.reduce_max(tf.abs(v), axis=N))` e
   `layers.Lambda(tf.abs, ...)` para extrair os nós espectral/temporal;
   MultiscaleCNN usava `layers.Lambda(apply_log_mel)` (closure local) no
   branch de áudio bruto. O Keras 3 **recusa por padrão** desserializar
   `Lambda` com função Python/closure (proteção contra execução de código
   arbitrário) — os modelos treinavam e salvavam normalmente, mas
   `tf.keras.models.load_model(path)` (sem `safe_mode=False`) falhava. A
   inferência de produção (`model_loader.py::TorchSSLOriginalModel`/
   `ModelLoader`) já usa `safe_mode=False` e não era afetada, mas
   `ModelTrainer.load_model` (sem essa flag) e qualquer reload externo
   quebrariam. Corrigido com camadas serializáveis dedicadas —
   `AxisMaxAbsLayer` e `LogMelFromMagnitudeLayer` (`layers.py`), reaproveitando
   `MagnitudeLayer` já existente — **mesma computação exata**, só a forma de
   serializar muda. Verificado: `model.save()` → `load_model()` (safe_mode
   padrão) funciona para os dois, zero camadas `Lambda` remanescentes.
2. **Augmentation raw-audio incompleta.** `AudioAugmenter._select_techniques`
   omitia `_volume_change` no branch raw-audio, apesar do próprio docstring
   do método listar "ruído, shift, **volume**, RawBoost, codec" como o
   conjunto esperado — a lista retornada não incluía volume. Afeta
   AASIST/RawGAT-ST (únicos consumidores do augmenter dinâmico no benchmark).
   Adicionado.
3. **Drift de hiperparâmetro silencioso (AASIST).**
   `benchmarks/planning.py` e `registry.py::default_params` tinham
   `learning_rate`/`l2_reg_strength` revertidos para 1e-4/1e-4, contradizendo
   o próprio comentário adjacente ("AJUSTE (retune): LR 1e-4->3e-4 e l2
   1e-4->2e-4") e este documento. Verificado contra o `benchmark_plan.json`
   real do retreino de 20260715: o valor EFETIVAMENTE usado já era 3e-4/2e-4
   (outra fonte, não identificada com certeza, já aplicava o valor correto em
   tempo de execução) — ou seja, **o retreino de hoje não foi afetado por
   este drift**; a correção alinha os arquivos-fonte para reprodutibilidade
   futura, sem mudar o comportamento já observado.
4. **4 testes obsoletos corrigidos** (nenhum indicava bug de comportamento,
   todos ficaram desatualizados quando `_build_paper_aasist`/janela 64.600
   viraram o default): assinatura de mock sem `crop_strategy`/`seed`;
   expectativa de janela 16.000 para AASIST (correto: 64.600); nome de
   camada `sinc_abs` da variante legada (correto no default:
   `aasist_sinc_abs`); busca por nome `res_block` em vez do tipo
   `ResidualBlock2D` (default nomeia `aasist_encoder_N`).

### Necessidade de retreino — reavaliada

Nenhuma das correções acima muda pesos, otimizador ou dado visto durante o
treino do retreino de 20260715 (item 3 confirmado sem efeito prático; itens
1 e 4 são serialização/teste, não treino). **Único item com efeito real em
um retreino futuro** é a técnica `_volume_change` adicionada (item 2) —
impacto esperado pequeno. Conclusão: **nenhum modelo precisa de retreino
por causa das correções desta seção.** AASIST/RawGAT-ST continuam sendo os
dois mais fracos da tabela (ver seção anterior) — isso é uma característica
de desempenho já registrada, não uma pendência introduzida agora. Retreiná-los
para captar o ganho marginal do volume_change é opcional, baixa prioridade.

## Retreino de 2026-07-14/15 — CONCLUÍDO (11/11 ok)

Run: `data/results/retrain_full_20260714/` (seed 42, confirmatory_v2 com test-lock,
protocolo waveform-AWGN, todos os fixes desta data ativos, +codec-eval mp3/opus,
+IC bootstrap 95%, ~16 h em RTX 3060). Consolidado em
`data/results/retrain_full_20260714/consolidated/`.

| Modelo | EER novo [IC95] | EER anterior | Acc@10dB | Situação |
| --- | --- | --- | --- | --- |
| Conformer | **0,18%** [0,00–0,36] | 0,27% | 98,0% | ✅ empata/melhora baseline; artefato de limiar sumiu (acc 99,8% @0,5) |
| Res2Net | **0,36%** [0,09–0,58] | 0,44% (13/07: 50%, NaN) | 97,5% | ✅ recuperado; overfit sanado (val 99,8%) |
| SVM | **0,58%** [0,31–1,16] | 4,31% (protocolo antigo) | 93,8% | ✅ primeiro número válido do protocolo waveform |
| CCT | **0,71%** [0,40–1,16] | 2,40% (13/07: 37%) | 95,0% | ✅ recuperado; melhor da história dele |
| AST | **0,98%** [0,53–1,47] | 1,33% (13/07: 51%) | 93,2% | ✅ recuperado; sem degradação (val máx ép. 53 mantida até 100) |
| RandomForest | 2,09% [1,47–2,71] | 1,69% | 92,4% | ≈ empate no limpo; robustez muito acima sob protocolo novo |
| RawNet2 | **2,71%** [2,09–3,47] | 2,89% (13/07: 32,7%) | 93,0% | ✅ recuperado (pooling paper); 192 min (antes: timeout) |
| HuBERT Original | **9,29%** [8,03–10,44] | 11,29% | 75,9% | ✅ melhora; t-DCF agora comparável (0,239) |
| AASIST | 9,51% [8,22–10,71] | 4,18% (INVÁLIDO: 4 scores) | 82,1% | ⚠️ primeiro EER VÁLIDO (1133 scores distintos); acc 87,9% < 95,8% histórico — investigar/nova semente antes de promover |
| RawGAT-ST | 9,56% [8,22–10,84] | 7,16% (13/07: 18,3%) | 80,8% | ⚠️ melhorou vs 13/07 mas abaixo do promovido — não promover |
| WavLM Original | **13,16%** [11,85–14,62] | 15,24% | 72,5% | ✅ melhora |

Achados de codec (novos): clássicos e espectrais quase imunes a MP3 64k;
Opus 24k triplica o EER do RandomForest (2,1→6,0%) e custa ~1–2 pp aos
espectrais. Promoção para `benchmark_final/`: pendente de decisão (critério:
melhorar/empatar baseline; AASIST/RawGAT-ST ficam de fora por ora — atenção:
o baseline do AASIST é incomparável por causa dos scores quantizados).

## 2026-07-15 — melhorias de acurácia: AASIST, RawGAT-ST, WavLM e HuBERT

Diagnóstico sobre o run 20260714 e ajustes aplicados nos 4 modelos mais
fracos da tabela:

| Modelo | Diagnóstico (evidência) | Ajuste |
| --- | --- | --- |
| AASIST | Overfit à cópia AWGN **estática** (mesma realização toda época): val_acc pico 88,8% na ÉPOCA 11 → 79,5% na 100 (train 99,2%). O run bom de 2026-07-07 (95,8%) treinava com augmentation dinâmico | Cópia estática substituída por **AudioAugmenter dinâmico na forma de onda** (domínio válido; custo por época idêntico). `training_augmentation_domain: waveform_dynamic_augmenter` no protocolo |
| RawGAT-ST | Mesmo padrão mais brando (val_acc máx 91,8% @ep30; val_loss mín @ep8) | Idem AASIST |
| WavLM Original | Via só **1 s central** dos 5 s (`_fit_length(raw, 16000)`) e **apenas a última camada** mean-pooled — as camadas intermediárias carregam os artefatos (SUPERB) | Janela 4 s (`--target-samples 64000`), **weighted-layer-sum** sobre as 13 hidden_states + **mean⊕std** pooling (`--layer-pooling weighted --time-pooling meanstd`), backbone `wavlm-base-plus` (94k h, mesmo tamanho). Contrato gravado no `.pt` (`embedding_config`) e honrado pelo wrapper de inferência via módulo compartilhado `app/domain/models/inference/ssl_head.py` (paridade por construção; artefatos antigos caem no legado) |
| HuBERT Original | Idem WavLM (1 s / última camada) | Idem (backbone mantido `hubert-base-ls960`) |

Retreino dos 4: `data/results/retrain_weak4_20260715/` (mesmo protocolo/test-lock
do run 20260714; demais 7 modelos NÃO são re-treinados — seus resultados de
20260714 permanecem os vigentes). **CONCLUÍDO** (WavLM precisou de 2 retries
por permissão do cache HF em bind-mount Docker/Windows — ver nota abaixo).

### Resultados finais (substituem as linhas de AASIST/RawGAT-ST/WavLM/HuBERT de 20260714)

| Modelo | EER novo [IC95] | EER 20260714 | EER histórico | Acc@10dB | Nota |
| --- | --- | --- | --- | --- | --- |
| AASIST | **4,89%** [3,92–5,73] | 9,51% (1º válido) | 4,18% (inválido: scores quantizados) | 88,7% | ✅ recuperado; val_acc não degrada mais (máx 95,5%@ep34 vs 88,8%@ep11→79,5% antes) |
| RawGAT-ST | **6,22%** [5,20–7,20] | 9,56% | 7,16% | 84,0% | ✅ melhora vs 20260714; ainda abaixo do baseline — não promover ainda |
| WavLM Original | **0,36%** [0,09–0,58] | 13,16% | 15,24% | 98,8% | ✅✅ ganho de 36×; validado (ver nota de shortcut) |
| HuBERT Original | **0,18%** [0,00–0,44] | 9,29% | 11,29% | 96,8% | ✅✅ ganho de 52×; validado (ver nota de shortcut) |

**Nota sobre o ganho SSL (validação anti-shortcut):** o salto de WavLM/HuBERT é
grande o bastante para exigir escrutínio — o dataset tem confounder de fonte
documentado (mlspt/ttsport 100% real, fkvoice 100% fake; só `brspeech` tem
as duas classes do MESMO locutor/canal). Isolando **apenas** as 1118 amostras
de `brspeech` no teste (onde a fonte não pode ajudar em nada), o EER
permanece baixo (HuBERT: 0,45%; WavLM: 0,45%) — confirma detecção real de
artefato de síntese, não exploração do atalho de fonte. Achado colateral: o
AASIST tem EER 5,9% no mesmo recorte `brspeech`-isolado vs. 9,5% geral —
generaliza pior para fontes não vistas misturadas no treino do que discrimina
dentro da fonte mais representada.

**Nota operacional:** WavLM (`wavlm-base-plus`, nunca baixado antes) falhou
2× com `PermissionError` ao criar `models--microsoft--wavlm-base-plus/` sob
`cache/huggingface/hub/` — o bind-mount Docker Desktop/Windows permite
leitura/escrita em diretórios pré-existentes mas não a CRIAÇÃO de novos
diretórios de topo pelo UID do container. Contornado pré-criando a árvore
(`hub/models--microsoft--wavlm-base-plus/{blobs,snapshots,refs}` e
`hub/.locks/models--microsoft--wavlm-base-plus/`) a partir do host antes de
relançar. Necessário sempre que um modelo HF **novo** (nunca baixado) for
usado pela primeira vez neste ambiente.

### Rigor acadêmico aplicado ao benchmark (mesma data)

Correções derivadas da análise metodológica para o artigo:

1. **Scores de logits clipados (bug de avaliação — afetava o AASIST).**
   `_run_neural.predict_p_fake` extraía `pred[:, 1]` direto da saída do
   modelo; para saídas LINEARES (AASIST/AMSoftmax, logits ≈[-15, 15]) o
   `_finite_scores` clipava em [0, 1], quantizando os scores em ~{0, 1}
   (os "4 valores distintos em 2250 predições") e invalidando
   EER/ROC/min-tDCF. Agora a saída linear é normalizada
   (softmax/sigmoid) antes da extração. **EER/AUC/t-DCF históricos do
   AASIST no benchmark Keras estão contaminados por esse clip** — o
   retreino/reexecução produz os primeiros números válidos.
2. **IC 95% de bootstrap** (EER/AUC/accuracy; 1000 reamostragens,
   percentil) em `evaluate_scores` — ligado por default no benchmark
   (`BenchmarkConfig.bootstrap_ci_samples`) e no runner SSL; `--bootstrap-ci`
   no CLI. Com n=2250, o IC do EER é ~±0,5–1 pp — sem ele o ranking fino
   não é interpretável.
3. **ECE (calibração, 15 bins)** reportado em toda avaliação.
4. **Curva DET** (escala probit, padrão da área) em `report.py`
   (`figures/det.png`), ao lado da ROC.
5. **Multi-sementes**: `run_models_sequential --seeds 42 43 44` roda a
   suíte completa por semente em `seed_<n>/` + `seeds_manifest.json` —
   com splits predefinidos congelados o teste é idêntico entre sementes
   (só varia RNG de treino/ruído): média±desvio e testes pareados.
6. **Cross-generator no orquestrador**: `--cross-generator fkvoice`
   repassado ao `run_benchmark` (bloqueado sob `--academic-protocol`,
   que exige o teste congelado).
7. **Robustez a codec com perdas**: `--codec-eval mp3 opus` (round-trip
   ffmpeg na forma de onda, mesmo ponto do protocolo do AWGN;
   `benchmarks/perturbations.py`; resultado em `codec_robustness` no
   results.json). MP3 64k e Opus 24k ≈ mensageria/VoIP.
8. **Auditoria de shortcut de fonte** *(histórico — não se aplica mais)*:
   `scripts/dataset/audit_source_shortcut.py` media se um RandomForest
   previa a FONTE (brspeech/fkvoice/mlspt/ttsport) a partir dos mesmos 63
   descritores — quantificava o confounder fonte↔classe (mlspt/ttsport só
   real; fkvoice só fake) e o teto de acurácia real/fake atingível sem
   detectar síntese.

   O script foi **removido em 2026-08-02**: o Protocolo de Dataset
   (CETUC × XTTS-v2 pareado) tem fonte única e pareamento enunciado a
   enunciado, então o confounder que ele media não existe por construção —
   toda frase e todo locutor aparecem nas duas classes. A garantia
   equivalente hoje é estrutural, verificada pelos oráculos de maioria
   (`source`/`speaker_id`/`text_id` = 0,5) em
   [`docs/data/dataset-protocol.md`](../data/dataset-protocol.md), e a
   checagem em tempo de execução vive na flag `--fail-on-source-shortcut`
   do `run_benchmark.py`, implementada em `benchmarks/runner.py` (nunca
   dependeu deste script).

Pendências NÃO-código do artigo (operacionais): tabela principal
speaker-disjoint (rodar com `--speaker-split`), avaliação cross-dataset
(ASVspoof/In-the-Wild), 3+ sementes no retreino, declaração de
licenças/ética do dataset.

### Consolidação de artefatos (mesma data)

Raiz canônica de dados consolidada em **`data/datasets/`** (settings já
apontava para lá; 15+ scripts e as abas Gradio hardcodavam `app/datasets`,
origem da fragmentação e do stub de 64 amostras). Removidos, após verificação
por hash MD5 (~16,4 GB liberados):

- `data/models/bench_*.{keras,pkl}` da raiz: 8 arquivos byte-idênticos aos
  promovidos em `benchmark_final/` (~690 MB). Sidecars `*_config.json` da
  raiz mantidos (proveniência de calibração).
- `data/datasets/splits/**/*.wav`: 15.000/15.000 cópias byte-idênticas de
  `real/`+`fake/` (~3,7 GB). A atribuição exata arquivo→split foi preservada
  em `splits/splits_files_manifest.json` (novo) + `splits_metadata.json`;
  regeneração via `preprocess_dataset.py --create-splits` (e os NPZs já
  embutem os splits).
- `data/results/**/best_checkpoint.keras`: 18 checkpoints intra-run de execuções
  concluídas (~12 GB; o do AST tinha 1,7 GB cada). Métricas, histórico,
  scores e figuras dos runs intactos.
- `app/datasets/benchmark_audio_raw_balanced_15k.npz` (o stub de 64
  amostras) e o antigo diretório `app/datasets/` (`doctor.py` atualizado).
- Bug pré-existente corrigido de carona: `speaker_manifest.py` resolvia o
  sidecar para `app/app/datasets/` (diretório fantasma — `BASE_DIR` subia só
  até `app/`); agora aponta para `data/datasets/speaker_manifest.json` real.

Os dois NPZs grandes NÃO são duplicatas (o `confirmatory_v2` difere em
~1,4 MB e tem rotation/test-lock próprios) — ambos mantidos.
`data/datasets/raw/` (29 GB de caches-fonte) mantido: é a origem para
regenerar `real/`+`fake/`, não uma duplicata.

## 2026-07-15/16 — AASIST e RawGAT-ST: revisão estrutural

Os nomes padrão agora selecionam as topologias espectro-temporais completas;
os builders 1D anteriores permanecem disponíveis como aasist_legacy e
rawgat_st_legacy para carregar checkpoints e executar ablações.

| Componente | AASIST | RawGAT-ST |
| --- | --- | --- |
| Janela raw | 64.600 amostras (~4,04 s), crop aleatório no treino | idem |
| Encoder | mapa Sinc 2D + 6 blocos residuais | dois encoders 2D independentes, 6 blocos cada |
| Grafos | S/T + master node; 2 pilhas de 2 HS-GAL; MGO | GAT S + GAT T, alinhamento de nós, produto e terceiro GAT |
| Head | cross-entropy em logits por padrão; AM-Softmax opcional | cross-entropy em logits |
| Otimizador | AdamW, LR 1e-4, CosineDecay até 5e-6 | idem, clip global 0,7 |
| Avaliação | média de três crops (início/centro/fim) antes do limiar EER | idem |

Augmentation raw por época passa a sortear AWGN por SNR, RawBoost completo
(LnL+ISD+SSI), simulação diferenciável de codec, RIR sintética, deslocamento
temporal e compressão dinâmica. O benchmark move LR/decay para os parâmetros
do construtor e desliga ReduceLROnPlateau, preservando o scheduler interno.

Impacto esperado, a confirmar por novo retreino multi-semente:

- menor EER por maior cobertura temporal e média multicrop;
- menor gap treino-validação por encoder/grafos fiéis e augmentation dinâmica;
- maior acurácia e recall em 10/20 dB e após codec;
- scores contínuos em logits, convertidos por softmax antes de EER/AUC.

As métricas históricas acima não foram reescritas: elas pertencem aos
checkpoints anteriores. Esta seção documenta a configuração do próximo
retreino confirmatório, não um resultado já medido.

4. **Contaminação Keras 2 via transformers (bug latente de ambiente).**
   `transformers.modeling_tf_utils` seta `TF_USE_LEGACY_KERAS=1` no
   `os.environ` do processo que o importa. Qualquer SUBPROCESSO herdado
   depois disso carrega `tensorflow.keras` como **Keras 2 (tf_keras)** — e o
   código Keras 3 do projeto quebra (ex.:
   `MultiHeadAttention.build(q, v)` do CrossAttentionFusionLayer;
   descoberto por falha ordem-dependente em
   `test_domain_imports_without_web_layer` na suíte combinada). Blindado em
   3 camadas: `run_models_sequential` pina `TF_USE_LEGACY_KERAS=0` no env
   dos filhos; `benchmarks/runner.py` faz `setdefault("0")` antes do import
   do TF; e o teste simula o Colab limpo pinando `0` no subprocesso.

## 2026-07-18 — Avaliação final dos 11 modelos promovidos: necessidade de retreino

Reavaliação pedida explicitamente pelo usuário sobre o conjunto já promovido em
`data/models/benchmark_final/` (`data/results/final_consolidated_20260715/`, que
mescla `retrain_full_20260714` + `retrain_weak4_20260715` — a fonte vigente,
ver seções acima). Antes de concluir, confirmado que o retreino confirmatório
de AASIST/RawGAT-ST de outra sessão (`data/results/retrain_aasist_rawgat_confirmatory_20260715/`,
container `xfakesong_retrain_rawgat_float32_resilient_20260718`) **não tem
resultado válido ainda** — `run_summary.json` mostra `status: running` e
`AASIST: timeout` (57.600 s = 16 h). Nenhum dado supera os números abaixo.

| Modelo | EER [IC95] | Baseline histórico | Δ | Acc | AUC | ECE | Acc@30/20/10dB | MP3/Opus EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conformer | 0,18% [0,00–0,36] | 0,27% | −0,09 | 99,82% | 1,0000 | 1,88% | 99,6/99,5/98,0% | 0,27/0,76% |
| HuBERT Original | 0,18% [0,00–0,44] | 11,29% | −11,11 | 99,87% | 1,0000 | 0,32% | 99,4/98,9/96,8% | — |
| MultiscaleCNN (Res2Net) | 0,36% [0,09–0,58] | 0,44% | −0,08 | 99,69% | 0,9999 | 0,37% | 99,2/99,0/97,5% | 0,40/0,71% |
| WavLM Original | 0,36% [0,09–0,58] | 15,24% | −14,88 | 99,69% | 1,0000 | 0,35% | 99,5/99,6/98,8% | — |
| SVM | 0,58% [0,31–1,16] | 4,31% (protocolo antigo) | −3,73 | 99,24% | 0,9998 | 0,98% | 98,0/96,0/93,8% | 1,47/0,84% |
| Hybrid CNN-Transformer (CCT) | 0,71% [0,40–1,16] | 2,40% | −1,69 | 99,20% | 0,9993 | 0,77% | 98,8/98,0/95,0% | 0,93/1,16% |
| SpectrogramTransformer (AST) | 0,98% [0,53–1,47] | 1,33% | −0,35 | 99,02% | 0,9977 | 0,97% | 97,8/96,4/93,2% | 1,16/2,00% |
| RandomForest | 2,09% [1,47–2,71] | 1,69% | +0,40 | 97,82% | 0,9989 | **11,28%** | 95,7/94,2/92,4% | 2,40/5,96% |
| RawNet2 | 2,71% [2,09–3,47] | 2,89% | −0,18 | 97,16% | 0,9977 | 0,81% | 95,5/95,3/93,0% | 4,18/4,09% |
| AASIST | 4,89% [3,92–5,73] | N/A (histórico inválido) | — | 95,02% | 0,9900 | 4,45% | 93,0/91,5/88,7% (11,4% EER@10dB) | 7,56/8,00% |
| RawGAT-ST | 6,22% [5,20–7,20] | 7,16% | −0,94 | 93,60% | 0,9865 | 3,37% | 93,3/91,1/84,0% (15,4% EER@10dB) | 8,49/8,44% |

Diagnóstico adicional (curvas treino×validação, `history.json` de cada run):

- **RandomForest — ECE 11,28%**, um patamar acima de todos os outros 10
  modelos (o segundo pior é o AASIST neural, 4,45%; a mediana dos outros 9
  fica abaixo de 1%). Acurácia/EER estão em linha com o baseline (empate
  dentro do IC), então não é um problema de discriminação — é o score de
  probabilidade não refletir a confiança real. Também é o que mais degrada
  sob Opus (2,09%→5,96% EER, ~2,9×). **Ação recomendada: recalibrar
  (Platt/isotonic ou temperatura) o RandomForest antes da próxima promoção —
  não é um retreino do zero.**
- **AASIST**: gap treino−validação de +5,42 pp e pico de val_accuracy na
  época 34/100 (val final 93,42%, abaixo do pico) — consistente com o
  diagnóstico já registrado de que este é historicamente o modelo mais
  instável dos 11, não uma regressão nova. É o número **válido** mais fraco
  do conjunto (EER quase 3× o próximo pior, RawGAT-ST) e o que mais degrada
  a 10 dB (11,4% EER) e sob codec (7,6–8,0%).
- **RawGAT-ST**: gap +3,15 pp, pico na época 42/100. Mesmo padrão, mais
  brando. É o único cujo IC95 superior (7,20%) encosta no baseline anterior
  (7,16%) — a melhoria de −0,94 pp não é estatisticamente contundente, mas
  também não há regressão. Pior robustez a 10 dB do conjunto (15,4% EER,
  quase triplica o valor limpo).
- Os outros 8 modelos: gap treino−validação ≤1,1 pp, sem sinal de overfit ou
  instabilidade, ECE baixo, degradação suave e monotônica com o ruído.

### Veredito

**Nenhum modelo tem bug pendente que invalide o resultado atual — nenhum
retreino é obrigatório.** Dois itens de acompanhamento, nenhum bloqueante:

1. **RandomForest**: recalibrar (não retreinar) por causa do ECE alto —
   prioridade média, é uma correção de pós-processamento barata.
2. **AASIST/RawGAT-ST**: continuam sendo os dois mais fracos do conjunto e os
   que mais perdem sob ruído/codec — característica de desempenho já
   registrada (não uma pendência nova). Retreiná-los é **opcional, baixa
   prioridade**, salvo se a sessão concorrente já em andamento (retreino
   confirmatório multi-semente) produzir um resultado válido que os supere —
   nesse caso, reavaliar promoção quando aquele run terminar.

Este veredito reavalia e confirma a conclusão da seção "Necessidade de
retreino — reavaliada" acima, agora sobre o conjunto final de 11 modelos já
promovidos (não apenas os 4 ajustados), com o achado novo do ECE do
RandomForest.

## 2026-07-18 — encerramento do retreino confirmatório

O retreino confirmatório foi encerrado por solicitação do usuário durante a
consolidação do projeto na branch `main`. As execuções Docker do XFakeSong
foram interrompidas e tiveram a política de reinício automático desativada.

- **AASIST:** o treino chegou a concluir e o melhor checkpoint da época 35 foi
  preservado em
  `data/results/retrain_aasist_rawgat_confirmatory_20260715/aasist/architectures/aasist/models/`.
  A avaliação confirmatória completa não foi concluída.
- **RawGAT-ST:** a execução resiliente float32 foi interrompida ainda na época
  1, no batch 108 de 656. A política híbrida (encoder em mixed precision e
  pipeline gráfico em float32) havia sido rejeitada no smoke por instabilidade.
  Portanto, não existe resultado confirmatório final válido a promover.
- A tentativa anterior interrompida por reinicialização do host permanece
  preservada em
  `data/results/retrain_aasist_rawgat_confirmatory_20260715/rawgat_st_interrupted_host_reboot_20260718/`.

Consequentemente, as métricas promovidas continuam sendo as do conjunto
`data/results/final_consolidated_20260715/`: AASIST com **95,02% de acurácia e
4,89% de EER**, e RawGAT-ST com **93,60% de acurácia e 6,22% de EER**. Esses
números não são resultados do confirmatório interrompido.

## 2026-07-19 — ressalva metodologica do o dataset anterior

As metricas promovidas acima foram medidas no artefato historico de 15 mil
amostras. A auditoria posterior demonstrou que esse conjunto e balanceado por
classe, mas confundido por fonte: MLS e TTS-Portuguese aparecem apenas como
reais, enquanto Fake Voices aparece apenas como fake. Tambem nao havia
cobertura integral de falante, texto, enunciado e gerador para sustentar todos
os protocolos de disjuncao hoje exigidos.

Por isso, os valores de AASIST (95,02% de acuracia; 4,89% de EER) e RawGAT-ST
(93,60%; 6,22%) permanecem validos somente como resultados **in-domain do
protocolo legado**. Eles nao devem ser reinterpretados como metricas do
aquele dataset nem como evidencia cross-domain. Nenhuma metrica daquele dataset foi inferida ou
fabricada nesta correcao: um novo numero so pode ser publicado depois de
reconstruir e selar o NPZ, satisfazer o oraculo de fonte e executar o
benchmark completo com bootstrap por cluster.

## 2026-08-06 — diagnóstico do `clean_benchmark_15k` (Conformer e RawGAT-ST)

Run analisado: `data/results/clean_benchmark_15k` sobre
`benchmark_dataset_15k.npz` (11/11 arquiteturas do escopo oficial, todas
`status: ok`). As métricas dos 11 foram **recalculadas a partir dos
`predictions_clean.csv`** e batem com os `metrics.json` até a 4ª casa; a
partição de teste é a mesma nos 11 (`test_split_sha256 = ab4c3a9f…`), com
test-lock v2 validado. O que segue são os dois únicos modelos com defeito de
treino — os outros nove não precisam de retreino.

### Diagnóstico

| Modelo | Sintoma | Evidência |
| --- | --- | --- |
| Conformer | **Colapso irreversível** | Divergiu na época ~14; `loss = ln 2 = 0.693` e `val_accuracy = 0.500` da época 22 à 100. Duas sessões independentes colapsaram igual. O número publicado (99,49%) vem do checkpoint da **época 10** — 10 das 100 épocas do orçamento declarado |
| RawGAT-ST | **Sobreajuste** (retune de 2026-07-02 não resolveu) | Treino 0,998 vs val 0,85 no melhor checkpoint (época 17); `val_loss` mínimo 0,511 subindo a 1,18. Acurácia 87,55%, EER 11,87%, min t-DCF 0,3149 — **abaixo de SVM e RandomForest no t-DCF**. Robustez não monotônica (77,21% a 10 dB contra 77,64% a 5 dB) |

### Ajustes aplicados

| Modelo | Parâmetro | Antes | Depois | Motivo |
| --- | --- | ---: | ---: | --- |
| Conformer | `learning_rate` | 1e-4 | **5e-5** | Pico sustentado até estourar; a topologia é pre-LN Macaron, estável por construção — o que restava era o passo |
| Conformer | `warmup_steps` | 1500 | **3000** | 1500 passos = ~2 épocas com batch 32; dobra para ~4 |
| Conformer | `decay_steps` | (omitido → 50000) | **76100** | ceil(24.324/32) = 761 passos/época × 100. Em 50.000 o cosseno zerava na época ~66 e as últimas 34 rodavam a 1e-7 |
| Conformer | `alpha` | (implícito) | **1e-7** | Explicitado junto com `decay_steps` |
| RawGAT-ST | `dropout_rate` | 0.35 | **0.5** | Sobreajuste, não subajuste |
| RawGAT-ST | `l2_reg_strength` | 1e-3 | **3e-3** | Idem (entra como `weight_decay` do AdamW) |
| RawGAT-ST | `decay_steps` | 100000 | **152100** | ceil(24.324/16) = 1.521 passos/época × 100. Mesmo desalinhamento do Conformer |

`learning_rate` do RawGAT-ST fica em 5e-5: o problema não é passo grande, é
capacidade sem freio.

Fontes editadas — as **três**, conforme a regra de sincronia do `CLAUDE.md`:
`benchmarks/planning.py::NEURAL_BENCHMARK_HPARAMS`,
`app/domain/models/architectures/rawgat_st.py::create_model` e
`app/domain/models/architectures/registry.py::default_params`. O Conformer não
carrega hiperparâmetros de otimização no `registry` (só `patience`,
`lr_patience`, `gradient_clip`, `augmentation_strength`), então o `planning.py`
cobre também o caminho da interface via `effective_hyperparameters()`.

### Estado de treino em quarentena

`data/results/_invalidado_conformer_colapso/` recebeu o `training_backup/`
(`{"epoch": 28}` — um `--resume` retomaria da sessão morta) e o
`models/` do Conformer, cujo `best.json` guardava `val_loss = 0.14744` da
sessão abortada em vez do `0.13258` que gerou as métricas publicadas. Sem
isso, o retreino não começaria limpo. Ver o README de lá.

### Não confundir com defeito de treino

- **SVM a 5 dB colapsa para exatamente 50,00%** (prediz "real" para as 1.382
  amostras). É reprodutível — o run arquivado de 40k dá 50,03%. Limitação
  estrutural do vetor tabular sob ruído fora da distribuição de treino, a
  reportar como achado, não a corrigir.
- **AASIST** tem EER 2,60% (5º melhor) mas acurácia 94,72% (8º): os scores
  saturam em 0,0099/0,9901 e o limiar de EER vai a 0,924. Com limiar ótimo
  faria 97,32%. Sob o protocolo de limiar fixo 0,5 o número está correto —
  é ressalva de texto, não retreino.

### Pendências de artefato (não exigem retreino)

- `predictions_robustness.csv` de **HuBERT/WavLM Original contém só o
  cabeçalho**; os agregados por SNR existem, mas não são reverificáveis.
- Os mesmos dois usam `bootstrap_unit = "sample"` enquanto os outros nove usam
  `"cluster"` (183 clusters) — **os ICs não são comparáveis entre si**.
- `history` truncado no `metrics.json` de **RawNet2 (17/100)** e **RawGAT-ST
  (91/100)**: só o trecho pós-retomada é persistido, então as figuras de
  convergência desses dois mostram um fragmento. A série completa está no
  `run.log`.

Os três são do runner/serialização, não do treino: bastam corrigir e
reexecutar a avaliação sobre os `.pt`/`.keras` já salvos.

### Complementos aplicados em 2026-08-06 (segunda rodada)

**1. `global_clipnorm` do RawGAT-ST deixou de ser config morto.**
`_build_paper_rawgat` compilava com o literal `global_clipnorm=0.7` enquanto o
`registry.py::default_params` declarava `gradient_clip: 0.5` — o valor do
registry nunca chegava ao otimizador. Agora é parâmetro real de
`create_model`, com **0.5** nas três fontes, e foi adicionado ao whitelist de
promoção do runner (`benchmarks/runner.py`, ramo `aasist`/`rawgatst`) — sem
isso ele renasceria morto. O AASIST divide esse ramo mas não expõe o
parâmetro; como a promoção é condicionada a `if key in train_config` e só o
plano do RawGAT-ST declara a chave, o AASIST não é afetado.

Nota de histórico: havia registro contraditório sobre esse clip — o
`retrain_ajustado.sh` dizia "clip 1.0->0.7" e o registry, "clip 0.8->0.5".
Ficou 0.5, o valor que o registry declara.

**2. Guarda de colapso (`CollapseAbort`).**
Nova em `app/domain/models/training/trainer.py`, ligada por padrão via
`TrainingConfig.abort_on_collapse`. Aborta o treino quando o modelo **já
esteve bom** (`val_accuracy >= 0.6`, o gatilho de armação) e depois caiu para
o nível do acaso (`<= 0.51`) por `collapse_patience = 15` épocas seguidas, ou
quando `val_loss` fica não-finito por `collapse_nan_patience = 3` épocas.

Não é early stopping e não conflita com `fixed_epoch_budget`: o early stopping
interrompe um modelo que ainda melhora devagar; esta guarda só dispara depois
que o modelo virou palpite constante e não voltou. O melhor checkpoint é
preservado (quem restaura continua sendo o `ResumableModelCheckpoint`), e o
aborto entra no resultado do treino como `collapsed` / `collapse_reason` —
sem isso um treino abortado passaria por treino curto qualquer no artefato.

Validação contra as séries reais de `val_accuracy` dos 11 modelos do
`clean_benchmark_15k` (simulação da lógica sobre os `history` gravados):

| Modelo | Pior época | Maior sequência <= 0.51 | Resultado |
| --- | ---: | ---: | --- |
| Conformer | 0,5000 | **84** | dispara na época 31 — pouparia 69 épocas |
| AASIST | 0,5316 | 0 | não dispara |
| RawGAT-ST | 0,6168 | 0 | não dispara |
| Hybrid CNN-Transformer | 0,6154 | 0 | não dispara |
| MultiscaleCNN | 0,7438 | 0 | não dispara |
| SpectrogramTransformer | 0,7926 | 0 | não dispara |
| HuBERT Original | 0,8407 | 0 | não dispara |
| WavLM Original | 0,9121 | 0 | não dispara |
| RawNet2 | 0,9306 | 0 | não dispara |

A separação é binária: os oito modelos saudáveis têm **zero** épocas no nível
do acaso depois de armar, contra 84 consecutivas do Conformer. `patience = 15`
é conservador de propósito — mesmo uma oscilação isolada não arma nada.

### Correções das pendências de artefato (2026-08-06)

Nenhuma exige retreino: os modelos `.pt`/`.keras` já treinados continuam
válidos. O que estava errado era a gravação.

**1. `predictions_robustness.csv` vazio em WavLM/HuBERT Original — causa raiz
encontrada.** Não era o runner SSL gravando errado: ele gravava o arquivo
correto e, três linhas depois, `benchmarks.report.write_all` — o writer
CANÔNICO dos 11 modelos — regravava por cima. O writer lê
`scores_robustness` do dicionário de resultados, e o runner SSL não colocava
essa chave lá; resultado, um CSV só com cabeçalho. A prova está no schema do
artefato final: `snr_db,sample_index,y_true,p_fake,y_pred,correct` é o do
`report.py`, não o do runner (`snr_db,idx,y_true,p_fake`).

Correções: (a) `scores_robustness` passou a entrar no dicionário de
resultados — o que também alinha o `metrics.json` com os 9 modelos Keras, que
já carregavam a chave; (b) `_write_predictions`, `_write_predictions_noisy` e
`_write_robustness` foram REMOVIDAS do runner SSL. Escreviam um schema
paralelo, eram sempre sobrescritas e criavam a ilusão de que o runner
controlava esses arquivos — foi essa duplicação que escondeu o bug.

**2. Bootstrap por amostra em vez de por cluster.** `evaluate_scores` já
aceitava `cluster_ids`; o runner SSL nunca passava. Agora os `cluster_ids` do
split de teste são extraídos em `_load_dataset` (antes de `data` ser solto
para liberar RAM) e chegam às avaliações limpa e sob ruído. Com isso os IC
95% de WavLM/HuBERT passam a ser por cluster, como nos outros nove — o
bootstrap por amostra subestima a largura porque trata amostras do mesmo
locutor/frase como independentes. Se os `cluster_ids` faltarem, o runner
agora emite WARNING em vez de degradar em silêncio.

**3. `provenance` ausente.** Era a única chave de proveniência faltando
(`input_preparation` e `noise_protocol` já vinham — a leitura anterior de que
faltavam os três estava errada, os campos apenas têm nomes diferentes dos do
runner Keras). Passou a reaproveitar `benchmarks.runner::_architecture_provenance`,
o mesmo builder do runner Keras, em vez de repetir literais — a duplicação já
tinha deixado o rótulo do WavLM defasado uma vez (declarava `base` quando o
runner treinava `base-plus`).

**4. `history` truncado em retomadas.** `model.fit()` devolve apenas as épocas
da execução corrente, então uma retomada via `BackupAndRestore` produz um
histórico parcial: RawNet2 saiu com 17 de 100 épocas e RawGAT-ST com 91 —
ambos treinaram as 100 (está nos `run.log`), mas as figuras de convergência
mostram um fragmento e a "melhor época" lida do artefato sai errada.

Novo callback `PersistentEpochHistory` (`trainer.py`): grava uma linha JSON
por época indexada pela época ABSOLUTA e reconstrói a série inteira ao fim.
O arquivo fica em `<arch>/models/epoch_history.jsonl`, **fora** do
`backup_dir` — aquele diretório é apagado ao concluir o treino
(`delete_checkpoint=True`) e levaria o histórico junto. A substituição só
ocorre se a série reconstruída for pelo menos tão longa quanto a da execução
corrente, então nunca encurta o histórico.

Validado por simulação do cenário real do RawNet2 (83 épocas → queda →
retomada na 84 → 100): 100/100 épocas, ordem absoluta preservada, valores da
sessão 1 intactos, e métrica ausente numa época vira `NaN` sem deslocar as
demais séries.

> **Estes quatro conserta a GRAVAÇÃO, não os números.** As métricas de
> WavLM/HuBERT no `clean_benchmark_15k` continuam válidas; o que faltava era
> poder reverificá-las e comparar os ICs. Para materializar os artefatos
> corrigidos basta reexecutar a avaliação sobre os `.pt` já salvos — não é
> preciso retreinar.

### Cobertura de testes das correções

`tests/unit/test_resume_guards_and_artifacts.py` — 13 testes travando os
quatro defeitos acima:

- **`CollapseAbort`** (5): dispara no padrão exato do Conformer; NÃO dispara
  num início lento (nunca armado — é a diferença para early stopping, e o que
  protege o warmup de 3.000 passos); NÃO dispara numa queda isolada que se
  recupera; pega `val_loss` não-finito; respeita `abort_on_collapse=False`.
- **`PersistentEpochHistory`** (4): reconstrói 100/100 épocas no cenário real
  do RawNet2 (83 → queda → retomada na 84); métrica ausente numa época vira
  `NaN` sem deslocar as demais séries; regravar a mesma época é idempotente;
  e o arquivo **não** cai dentro do `backup_dir` (que é apagado ao concluir).
- **Artefatos** (2): `_write_arch_predictions_noisy_csv` escreve as linhas com
  `scores_robustness` no dicionário e só o cabeçalho sem ela — o teste
  reproduz o artefato defeituoso; e uma guarda por AST impede que os writers
  paralelos do runner SSL sejam reintroduzidos.
- **Sincronia** (2): valores iguais nas três fontes para RawGAT-ST e AASIST
  (a guarda existente, `test_default_params_are_accepted_by_builder`, checa se
  a CHAVE é aceita; esta checa se o VALOR bate), e `global_clipnorm` presente
  tanto no `create_model` quanto no whitelist de promoção do runner.

---

## 2026-08-09 — fidelidade do artefato (auditoria do `clean_benchmark_15k`)

Seis defeitos encontrados relendo o run já concluído. **Nenhum altera métrica
publicada**; todos alteram o que se pode AFIRMAR a partir dela. São de gravação
e de análise, não de treino — as duas pendências de retreino (Conformer e
RawGAT-ST) continuam sendo as da seção de 2026-08-06.

### P0 — a janela do SSL declarada não era a usada

`scripts/benchmark/run_wavlm_original_benchmark.py` gravava o literal
`[16000, 1]` em **quatro** pontos (sidecar `_config.json`,
`dataset.prepared_shape`, `architectures[].input_shape` e
`input_preparation.prepared_shape`) enquanto o run usava `--target-samples`,
cujo default era 64.000. O `run.log` do `clean_benchmark_15k` registra
`target_samples: 64000`: os artefatos declaram 1 s para modelos treinados com
4 s.

A inferência **nunca** foi afetada — `TorchSSLOriginalModel` resolve a janela
pelo `embedding_config` gravado dentro do `.pt`
(`app/domain/models/inference/ssl_head.py`), e lá o valor sempre foi o real. O
que estava errado era a metadata de que sai a seção de métodos. (O model card
em Markdown não imprimia a janela de forma alguma — passou a imprimir, junto
com a estratégia de recorte.)

`crop_strategy` também mentia: dizia `"center"` de forma fixa, mas
`_fit_length` só recorta quando o clipe é MAIOR que a janela. Com clipe de
48.000 e janela de 64.000 ele **repete** o sinal (tiling).

| Item | Antes | Depois |
| --- | --- | --- |
| Literais `[16000, 1]` | 4 ocorrências fixas | `[int(args.target_samples), 1]` |
| `--target-samples` (default) | 64000 | **48000** |
| `crop_strategy` | sempre `"center"` | `identity` / `center_crop` / `tile_repeat` |
| Tiling | silencioso | `tiled_padding_samples` + `tiled_padding_ratio` + WARNING |

**Por que 48.000.** O comentário que justificava 64.000 falava em "1 s central
dos 5 s" — escrito quando os clipes tinham 5 s. O dataset canônico atual tem
**3 s** (48.000 @16 kHz), então 64.000 fazia 25% de cada entrada ser repetição
do próprio primeiro segundo, em treino e em teste. 48.000 é o clipe inteiro:
nem recorte, nem repetição, e é o que os espectrais também veem.

> Reavaliar WavLM/HuBERT com a janela nova **exige recomputar os embeddings**
> (o backbone congelado vê uma entrada diferente). Não é retreino de backbone,
> mas também não é só regravar JSON.

### P1 — `converged` não enxerga colapso

`converged` deriva só da AUC/acurácia do **checkpoint selecionado**, então
descreve o artefato promovido, não o treino. O Conformer colapsou da época ~17
à 100 e saiu `converged: True`.

Novo `benchmarks/stability.py::analyze_training_stability`: aplica à série
completa o MESMO critério do `CollapseAbort` que já roda durante o treino (um
teste trava a igualdade dos defaults, para que o veredito pós-hoc não possa
divergir da guarda). Grava `training_stability` no `metrics.json` dos dois
runners, com `status` em `stable` / `collapsed` / `recovered_collapse` /
`diverged_nonfinite` / `unknown`. `convergence_criteria` ganhou
`scope: "checkpoint_selecionado"` para não sugerir mais do que mede.

Aplicado ao run existente, separa exatamente o caso conhecido:

| Modelo | `converged` | `training_stability.status` |
| --- | --- | --- |
| Conformer | `True` | **`collapsed`** (época 17→100, 84 épocas no acaso) |
| RawNet2 | `True` | `stable` + aviso "17 épocas contra orçamento de 100" |
| RawGAT-ST | `True` | `stable` + aviso "91 épocas contra orçamento de 100" |
| Os outros 6 neurais | `True` | `stable` |
| SVM / RandomForest | `True` | `unknown` (sem histórico — honesto) |

Os avisos de RawNet2/RawGAT-ST são o histórico truncado por retomada já
descrito em 2026-08-06; o `PersistentEpochHistory` impede o caso novo, e este
aviso é a rede de segurança para quando ele ainda assim aparecer.

### P1 — a latência mistura três runtimes

O escopo oficial mede latência em **Keras/TF** (7 neurais), **PyTorch**
(WavLM/HuBERT Original) e **scikit-learn** (SVM/RandomForest). A diferença
entre pilhas é da mesma ordem da diferença entre arquiteturas — WavLM, com
94,8 M de parâmetros, mede 17,4 ms contra 52,9 ms do Conformer, de 28,6 M —, e
a figura de tradeoff apresentava as três na mesma escala sem ressalva.

`measure_latency_profile` passou a receber `runtime=` e gravar
`runtime`/`runtime_version`/`device`/`cross_runtime_comparable: false`. O
runner SSL passou a emitir o `latency_profile` completo (antes só o escalar).
Em `consolidate_results.py`, a figura de tradeoff usa **marcador por runtime**
(círculo/quadrado/triângulo) além da cor por família, com legenda própria e a
ressalva no rodapé.

### P1 — sem teste pareado, ICs sobrepostos eram lidos como empate

Conformer (EER 0,43% [0,14; 1,00]) e Hybrid CNN-Transformer (0,43%
[0,00; 0,74]) têm ICs quase coincidentes. Concluir "sem diferença" daí é
inválido: os dois veem as MESMAS 1.382 amostras, e o que decide é a
distribuição da **diferença**, não a de cada um.

Novo `benchmarks/significance.py`:

- **`mcnemar_test`** — exato (binomial), sobre as decisões duras no limiar fixo
  do protocolo. A forma qui-quadrado é ruim justamente quando o total
  discordante é pequeno, que é o caso entre os modelos do topo.
- **`paired_bootstrap_test`** — IC 95% e p-valor da diferença de EER/AUC,
  reamostrando os mesmos índices para os dois modelos.
- **`holm_adjust`** — Holm-Bonferroni. Comparar 11 modelos par a par são 55
  testes; sem correção, ~3 saem "significativos" a 5% só por acaso.
- **`compare_models`** — matriz de todos os pares, com p bruto e ajustado.

Ambos os testes agrupam/reamostram por **cluster** quando os IDs estão
disponíveis. Para isso, `dataset.test_cluster_ids` passou a ser persistido no
`results.json` dos dois runners — os IDs já eram usados nos IC por modelo, mas
não eram gravados. Runs anteriores caem para amostra e o relatório declara o
otimismo resultante em `protocol.warning`.

`consolidate_results.py` gera `benchmark_significance.json` por padrão
(`--no-significance` desliga, `--significance-bootstrap N` ajusta as
reamostragens) e imprime aviso quando algum modelo tem
`training_stability.stable == false`.

### P2 — declarações de protocolo que faltavam

**Assimetria de ajuste.** Os clássicos ajustam em **treino+validação** (25.780
amostras) porque não têm checkpoint a selecionar — a busca de hiperparâmetros
sai de CV interna sobre o treino limpo. Os neurais ajustam só no treino
(24.324, já com a cópia ruidosa) e reservam a validação para escolher a época.
Não favorece nenhum lado (os clássicos veem MAIS dados), mas o `fit_samples`
de 25.780 contra 24.324 parecia divergência de dados. Ambos os caminhos agora
declaram `fit_splits`, `validation_role` e `val_samples` no `fit_strategy`.

**`codec_robustness`.** É opt-in via `--codec-eval` e está implementado
(`benchmarks/perturbations.py`, round-trip MP3/Opus com ffmpeg), mas um `{}`
vazio não distinguia "não pedido" de "pedido e nada encontrado" de "ignorado
por domínio incompatível" — no `clean_benchmark_15k` a primeira leitura era a
verdadeira. Novo campo irmão `codec_eval_status` com `requested` e `status`
(`not_requested` / `ok` / `partial` / `skipped` / `not_supported`). O runner
SSL, que não implementa a flag, declara `not_supported` em vez de omitir a
chave.

### O que isto NÃO resolve

A **repetição de sementes** continua em `n_seeds: 1` (`benchmarks/config.py`).
O IC 95% e o teste pareado medem variância de AMOSTRAGEM DO TESTE; nenhum dos
dois mede variância de TREINO. Para isso é preciso rodar cada arquitetura N
vezes — o suporte existe (`BenchmarkConfig.n_seeds`), o custo é que não cabia
no orçamento de GPU do run. Fica declarado como limitação, não como resolvido.

### Cobertura de testes

`tests/unit/test_benchmark_reporting_fidelity.py` — 32 testes:

- **Janela SSL** (6): as três estratégias de `_input_preparation_block`;
  `_fit_length` de fato repete o começo do clipe quando a janela é maior (a
  cauda é comparada ao início, elemento a elemento); o default de
  `--target-samples` lido por AST; e uma guarda que falha se qualquer
  `[16000, 1]` voltar ao payload.
- **Estabilidade** (8): o padrão exato do Conformer; treino saudável; início
  lento que nunca arma a guarda; colapso com recuperação distinguido do
  irreversível; `val_loss` não-finito; histórico truncado virando aviso;
  clássico sem histórico devolvendo `unknown` em vez de fingir veredito; e a
  igualdade dos defaults com o `CollapseAbort`.
- **Latência** (2): o perfil declara runtime e `cross_runtime_comparable`.
- **Significância** (10): McNemar ignora acertos em comum, detecta dominância
  sistemática e produz p MAIOR por cluster que por amostra (o otimismo que o
  bootstrap por amostra introduzia); bootstrap pareado não vê diferença entre
  um modelo e ele mesmo, separa bom de ruim, e nunca devolve p=0; Holm é
  monótono, limitado a 1 e preserva a ordem de entrada; a matriz cobre C(n,2)
  pares. Inclui a regressão do arredondamento — `round(2.2e-11, 6)` = 0.0
  fazia o p ajustado sair MENOR que o bruto, e por isso p-valores passaram a
  ser arredondados por algarismos significativos.
- **Declarações** (6): `fit_splits` nos dois caminhos, `codec_eval_status`,
  `test_cluster_ids` persistido nos dois runners, a sanidade numérica do
  binomial exato contra valores calculados à mão, e a guarda de
  **fingerprint**: a comparação pareada recusa modelos vindos de conjuntos de
  teste diferentes (`test_split_sha256` divergente) em vez de produzir uma
  saída com cara de válida — é o erro que misturar os runs de 15k e 40k
  cometeria.

### Dois testes desatualizados, corrigidos de passagem

Ambos já falhavam na branch **antes** destas correções, por trabalho anterior
que mexeu no código sem atualizar a guarda:

- `tests/unit/test_benchmark.py::test_neural_benchmark_plan_uses_curated_hyperparameters`
  ainda exigia `learning_rate == 1e-4` para o Conformer, valor que o retune de
  2026-08-06 substituiu por 5e-5. Atualizado para 5e-5 e ampliado com
  `warmup_steps == 3000` e `decay_steps == 76100`, que a mesma decisão fixou e
  ninguém checava.
- `tests/unit/test_test_documentation.py::test_test_documentation_counts_match_tree`
  compara a contagem de arquivos de teste com
  `docs/development/quality-and-testing.md`. A branch tinha adicionado dois
  arquivos de unit sem atualizar o doc (61 → 63); com este, 64. Doc corrigido
  para 64 unit / **83** no total.
