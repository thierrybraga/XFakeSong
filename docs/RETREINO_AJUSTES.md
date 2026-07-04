# Retreino com Ajustes — pós `clean_benchmark_full_20260626`

Documento de rastreio dos ajustes de hiperparâmetros aplicados após o
diagnóstico do benchmark completo do harness (14 arquiteturas suportadas; 11
modelos no recorte consolidado atual). Os ajustes estão **aplicados no
código**. Os 4 modelos do escopo oficial do TCC que precisavam de retreino
(RawGAT-ST, AASIST, WavLM Original, HuBERT Original) foram **retreinados e
promovidos em 2026-07-02** — ver
["Retreino de 2026-07-02 — concluído"](#retreino-de-2026-07-02--concluído)
abaixo. Ensemble e EfficientNet-LSTM não fazem parte da tabela consolidada do
TCC (11 modelos) e seu retreino permanece pendente. RandomForest também não
foi retreinado com o novo grid regularizado (`random_forest.py`); o número
atual no TCC vem do run anterior ao ajuste — a robustez fraca sob ruído
(68,04% @10dB) já é discutida no texto como limitação estrutural do vetor
tabular, não como defeito de treino a corrigir, mas o retreino com o grid
ajustado ainda não foi feito e poderia mudar esse número (overfitting
diagnosticado via `mean_train_score=1.0` no tuning).

## Resumo do diagnóstico

| Modelo | Sintoma observado | Evidência |
| --- | --- | --- |
| RawGAT-ST | Overfitting/divergência | `val_acc` cai após época 4; `val_loss` 0.39→1.85; pior limpo (0.833) |
| AASIST | Subajuste | `val_acc` travada ~0.92; recall colapsa a 0.29 @10dB |
| Ensemble | Colapso de robustez | acc 0.50 e recall ~0 @10dB (prediz tudo "real") |
| RandomForest | Overfitting | `mean_train_score=1.0` no tuning; robustez 0.98→0.68 |
| SVM | Robustez fraca | 0.966→0.672 @10dB |
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
30/20/10) em `results/retune_ajustado_<data>/`. Os modelos não ajustados não são
retreinados.

## Verificação (antes de promover)

1. `python scripts/reporting/consolidate_results.py --results results/retune_ajustado_<data>`
2. `python scripts/reporting/validate_artifacts.py --results results/retune_ajustado_<data>`
3. Comparar `accuracy`/`f1`/`eer` e a curva de robustez (10 dB) com o baseline.
4. Só então sincronizar para `app/models/benchmark_final`:
   `python scripts/reporting/sync_completed_benchmark_artifacts.py --results results/retune_ajustado_<data>`

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
["Retreino de 2026-07-02 — concluído"](#retreino-de-2026-07-02--concluído)
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

## Retreino de 2026-07-02 — concluído

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
teste; `results/retune_ajustado_20260701_2051/` para WavLM/HuBERT Original,
`results/retune_ajustado_fix_20260701_2303/` para RawGAT-ST/AASIST,
consolidado em `results/tcc_consolidated_20260702/`):

| Modelo | Acc. limpa (antes → depois) | EER (antes → depois) | Acc. @10dB (antes → depois) |
| --- | ---: | ---: | ---: |
| AASIST | 91,69% → 92,49% | 8,31% → 7,42% | 71,38% → 88,93% |
| RawGAT-ST | 83,56% → 86,98% | 16,27% → 12,80% | 53,20% → 82,93% |
| HuBERT Original | 90,18% → 88,76% | 9,73% → 11,29% | 50,67% → 80,98% |
| WavLM Original | 86,09% → 84,67% | 13,47% → 15,24% | 52,58% → 75,91% |

Todos os 4 modelos convergiram (`converged: True`) e ganharam robustez
substancial a 10 dB, à custa de perda marginal de acurácia/EER no conjunto
limpo (WavLM e HuBERT) — *trade-off* esperado ao expor o classificador a
ruído no treino. Artefatos sincronizados para `app/models/benchmark_final/`
(11/11 modelos, `python scripts/reporting/sync_completed_benchmark_artifacts.py`) e
tabelas/figuras do TCC regeneradas (`python scripts/reporting/update_tcc_latex.py`).
`tcc_overleaf/main.tex` (Seção 5 — Análise dos Resultados — e Conclusão)
reescrito para refletir os números corrigidos; a narrativa de robustez a
ruído deixou de apontar RawGAT-ST/SSL como os mais frágeis e passou a
identificar SVM/Random Forest como os modelos menos robustos do conjunto.
