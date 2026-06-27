# 21 — Plano de Reajuste e Retreino de Modelos

> Plano técnico derivado da análise de robustez/generalização (jun/2026).
> Mapeia cada recomendação a **arquivos e mudanças concretas**. Nenhum código
> foi alterado ainda — este documento é o checklist de execução.

## 0. Causa-raiz (ler antes de tudo)

Três achados de auditoria explicam quase todos os sintomas relatados. Corrigi-los
ataca a origem em vez dos sintomas modelo a modelo.

### 0.1 Descasamento treino↔teste de ruído (origem das quedas de SNR)

- **Teste** injeta AWGN **calibrado por SNR alvo** (10/20/30 dB), `noise_std`
  derivado da potência do sinal por amostra:
  `BenchmarkData.add_awgn` — [benchmarks/data.py:177](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py).
- **Treino** adiciona ruído gaussiano de **`stddev` fixo** (`noise_factor=0.1`,
  não calibrado por SNR e não casado com os SNRs de teste):
  `AudioAugmenter._add_noise` — [app/domain/models/training/augmentation.py:104](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/augmentation.py).
- **Pior:** o runner **desativa augmentation** exatamente para os modelos
  frágeis a ruído:
  - RawNet2 → `use_augmentation=False` — [benchmarks/runner.py:331](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)
  - AASIST / RawGAT-ST → `use_augmentation=False` — [benchmarks/runner.py:339](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)
  - SpectrogramTransformer → `use_augmentation=False` — [benchmarks/runner.py:379](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)
- Já existe um helper de ruído **SNR-correto** (potência do sinal × fator),
  porém **não usado** pelo `AudioAugmenter`:
  `add_gaussian_noise` — [app/domain/models/augmentation/components/time_domain.py:7](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/augmentation/components/time_domain.py).

> Consequência: os modelos são **testados** com ruído calibrado que **nunca
> viram no treino**. Logo, qualquer "augmentation ruidoso" recomendado precisa
> ser **parametrizado por SNR** e **casado com a faixa de teste** (≈ 5–40 dB,
> cobrindo 10/20/30).

### 0.2 Split só por rótulo → risco de vazamento de fonte (P0 metodológico)

- Todos os splits hoje são **estratificados apenas por classe** (real/fake),
  sem agrupamento por falante/gerador:
  - `SecureDataSplitter._stratified_split` — [app/core/training/secure_training_pipeline.py:86](https://github.com/thierrybraga/XFakeSong/blob/main/app/core/training/secure_training_pipeline.py)
  - `BenchmarkData.stratified_split` — [benchmarks/data.py:145](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py)
  - `runner._stratified_test_labels` — [benchmarks/runner.py:267](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)
- A identidade de **falante/gerador existe apenas no nome de arquivo/diretório**
  (`brspeech`, `fkvoice/fakevoice`, `cetuc`, `cv`, `fleurs`) —
  [scripts/build_dataset.py:325](https://github.com/thierrybraga/XFakeSong/blob/main/scripts/build_dataset.py) — e **não é
  propagada por amostra** para o `.npz` (só `X_*`/`y_*` são salvos;
  `from_npz` lê apenas `X/y` — [benchmarks/data.py:43](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py)).

> Consequência: o mesmo falante/gerador pode cair em treino **e** teste →
> números 100% (Conformer, Sonic Sleuth) possivelmente **inflados**. Sem
> propagar `groups`, o reteste por falante/gerador é **impossível**. É
> pré-requisito de tudo.

### 0.3 Val não representativo (origem do colapso val→teste)

- Early stopping e restore monitoram `val_loss`
  ([app/domain/models/training/trainer.py:666](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/trainer.py)),
  com `restore_best_weights=True`. Não há como "monitorar métrica de teste"
  (seria vazamento). O colapso de 25 pp do SpectrogramTransformer é **sintoma
  de val não-disjunto por fonte** (§0.2), não de bug no early stopping.

---

## Status de implementação (jun/2026)

Ajustes de código **aplicados e testados** (`tests/unit/test_retraining_adjustments.py`):

| Item | Estado | Onde |
| --- | --- | --- |
| P2.0 ruído calibrado por SNR (casa treino↔teste) | ✅ feito | [augmentation.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/augmentation.py), [settings.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/core/config/settings.py) |
| P2 augmentation reativada (RawNet2, SpectrogramTransformer) | ✅ feito | [runner.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py) |
| P1 hiperparâmetros + restauração de checkpoint | ✅ feito | [spectrogram_transformer.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/architectures/spectrogram_transformer.py), [runner.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py) |
| P0 `groups` ponta-a-ponta + split por grupo | ✅ feito | [data.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py), [secure_training_pipeline.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/core/training/secure_training_pipeline.py) |
| P0.4 protocolo cross-generator + CLI/presets | ✅ feito | [config.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/config.py), [run_benchmark.py](https://github.com/thierrybraga/XFakeSong/blob/main/scripts/run_benchmark.py) |
| P2 SVM/RF: RASTA-PLP + augmentation ruidoso (espaço de feature) | ✅ feito | [data.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py), [runner.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py) |
| P2 Ensemble: reponderar fusão por robustez (`weights="robustness"`) | ✅ feito | [detection_service.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/services/detection_service.py) |
| P2 WavLM: harness de ablação de fine-tuning (LR/épocas) | ✅ feito | [ablate_wavlm_finetune.py](https://github.com/thierrybraga/XFakeSong/blob/main/scripts/ablate_wavlm_finetune.py) |
| P3 MultiscaleCNN: módulo de pruning por magnitude | ✅ feito | [magnitude_pruning.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/magnitude_pruning.py) |

**Achados da execução (importantes para a banca):**
- **Não há ID de falante** nos nomes (`brspeech_NNNNN`, `cvpt_NNNNN`,
  `fkvoice_NNNNN`) — só dá para agrupar por **fonte/gerador** (3 grupos). O
  split por *falante* puro é inviável com os metadados atuais.
- As 3 fontes são **correlacionadas à classe** (cvpt=real, fkvoice=fake,
  brspeech=real+fake). Logo o `group_split` puro tende a um **teste de classe
  única** (degenerado). O reteste **defensável** aqui é o **cross-generator**.
- Validação executada (SVM, subset real): cross-generator (held-out fkvoice/XTTS)
  → acurácia limpa **0,47** / AUC **0,65** vs. ~0,99 no split estratificado —
  evidência concreta de que os números altos sofrem de vazamento de fonte.

**Retreino completo:** sem GPU em Windows nativo (TF ≥2.11). A máquina tem RTX
3060 → rodar sob **WSL2**. Comandos prontos:

```bash
# Reteste cross-generator (P0.4) — o mais importante antes da defesa
python scripts/run_benchmark.py --full \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --cross-generator fkvoice --out results/bench_xgen

# Split disjunto por fonte (P0) — use se houver mais grupos/falantes no futuro
python scripts/run_benchmark.py --full \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --group-split --out results/bench_group

# Retreino padrão (in-distribution) com augmentation SNR + P1/P2 já aplicados
python scripts/run_benchmark.py --full \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --out results/bench_indist
```

**Follow-ups (uso):**

```bash
# Ablação de fine-tuning do WavLM (varre LR; baseline HuBERT)
python scripts/ablate_wavlm_finetune.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --lrs 1e-5 3e-5 1e-4 --epochs 30 --out results/ablation_wavlm

# Pruning do MultiscaleCNN (requer: pip install tensorflow-model-optimization)
#   from app.domain.models.training.magnitude_pruning import prune_and_finetune
#   pruned = prune_and_finetune(model, X_tr, y_tr, validation_data=(X_v,y_v),
#                               target_sparsity=0.5, epochs=5)

# Fusão multi-modelo reponderada por robustez (inferência)
#   DetectionService.detect_multi_model(audio, ["Conformer","AASIST","SVM"],
#                                       fusion="weighted_avg", weights="robustness")
```

- SVM/RF: `classical_noise_augmentation=True` (default) anexa cópias AWGN do
  treino nos SNRs avaliados; features agora incluem **RASTA-PLP**.
- **Pendente (menor):** limiar **adaptativo por SNR** na inferência (exige
  estimar SNR em runtime). Hoje há o threshold EER calibrado (melhor que 0,5);
  o por-SNR fica como item opcional.

---

## Prioridade 0 — Pré-requisito metodológico (split por falante/gerador)

**Objetivo:** propagar `groups` (id de falante/gerador) ponta-a-ponta e trocar os
splits por versões **agrupadas**. Sem isto, P1/P2/P3 medem números potencialmente
vazados.

### P0.1 — Capturar a fonte por amostra na extração
- [ ] Em [scripts/build_dataset.py](https://github.com/thierrybraga/XFakeSong/blob/main/scripts/build_dataset.py): derivar um
  `group_id` por arquivo (falante quando houver; senão gerador/fonte:
  `brspeech_spk###`, `xtts`, `cetuc_spk###`, `cv_###`, `fleurs_###`).
- [ ] Salvar no `.npz` um array `groups` alinhado a `X/y`
  (e opcionalmente `group_kind` ∈ {speaker, generator}).

### P0.2 — Carregar `groups` no benchmark
- [ ] `BenchmarkData` ([benchmarks/data.py:14](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py)): novo campo
  `groups: np.ndarray | None`; `from_npz` lê a chave `groups` se presente
  ([benchmarks/data.py:43](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py)).

### P0.3 — Split agrupado (disjoint por fonte)
- [ ] `BenchmarkData.stratified_split` ([benchmarks/data.py:145](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py)):
  quando `groups` existir, usar `StratifiedGroupKFold` /
  `GroupShuffleSplit` (sklearn) garantindo **falante/gerador disjunto** entre
  train/val/test. Manter o caminho atual como fallback sem `groups`.
- [ ] `runner._stratified_test_labels` ([benchmarks/runner.py:267](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)):
  espelhar a lógica agrupada para o teste held-out.
- [ ] `SecureDataSplitter` ([app/core/training/secure_training_pipeline.py:34](https://github.com/thierrybraga/XFakeSong/blob/main/app/core/training/secure_training_pipeline.py)):
  adicionar `use_group_split` + leitura de `metadata['groups']` em
  `split_data` ([:41](https://github.com/thierrybraga/XFakeSong/blob/main/app/core/training/secure_training_pipeline.py)).

### P0.4 — Protocolo cross-generator explícito
- [ ] Adicionar modo de avaliação "cross-generator": treinar sem o gerador XTTS
  (Fake Voices) e testar **só** nele — mede generalização a gerador inédito.
  Relatar separado do split in-distribution.

### P0.5 — Reavaliar TODOS com split agrupado
- [ ] Reexecutar o benchmark com P0.1–P0.3 ativos. **Foco de validação:**
  Conformer e Sonic Sleuth (hoje 100%) — confirmar se caem para nível realista.

---

## Prioridade 1 — Retreino obrigatório

### Spectrogram Transformer (colapso val 98% → teste 71,5%)
Hiperparâmetros atuais: `lr=0.0003, dropout=0.1, l2=0.0001`
([optimized_training_config.py:271](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/optimized_training_config.py)).

- [ ] **Val disjunto por fonte** (P0) — provável causa real do colapso.
- [ ] **Restaurar melhor checkpoint** — garantir `restore_best_weights=True`
  (já ativo em [trainer.py:666](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/trainer.py)) **e**
  reativar checkpoint do melhor modelo: hoje o runner faz
  `checkpoint_best` opcional ([runner.py:382](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)); torná-lo
  **default** para este modelo.
- [ ] **LR menor + weight decay maior + mais dropout**: em
  `get_recommended_hyperparameters["SpectrogramTransformer"]`
  ([optimized_training_config.py:271](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/optimized_training_config.py)) →
  `dropout_rate 0.1 → 0.3`, `weight_decay` explícito (ex. `0.01–0.05`),
  `learning_rate 3e-4 → 1e-4`.
- [ ] **Reativar `reduce_lr_on_plateau`** — hoje desligado p/ este modelo
  ([runner.py:358](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)).
- [ ] **Ligar augmentation** (hoje `False` — [runner.py:379](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py))
  com SpecAugment + ruído SNR (ver P2.0).
- [ ] **Early stopping** monitorando `val_loss` no val **disjunto** (não há
  "métrica de teste"; o val correto é o que aproxima o teste).
- [ ] Critério de aceite: gap val−teste < 5 pp.

---

## Prioridade 2 — Retreino por fragilidade a ruído

### P2.0 — Base comum: augmentation ruidoso calibrado por SNR (corrige §0.1)
Aplicar a **todos** os modelos P2 antes dos ajustes específicos.

- [ ] Reescrever `AudioAugmenter._add_noise`
  ([augmentation.py:104](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/augmentation.py)) para
  **amostrar um SNR alvo** (ex. uniforme em 5–40 dB) e derivar `noise_std` da
  potência do sinal — reutilizar a fórmula de
  [components/time_domain.py:7](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/augmentation/components/time_domain.py)
  e de [data.py:177](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py) (mesma definição usada no teste).
- [ ] Adicionar `snr_range_db` ao `augmentation_config`
  ([settings.py:183](https://github.com/thierrybraga/XFakeSong/blob/main/app/core/config/settings.py)).
- [ ] Garantir cobertura de reverberação/codec p/ raw-audio: já existem
  `_rawboost` ([augmentation.py:377](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/augmentation.py))
  e `_codec_simulation` ([:413](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/augmentation.py));
  expor RIR/reverb (MUSAN/RIR opcional) como técnica adicional.
- [ ] **Parar de desligar augmentation** no runner para os modelos P2
  ([runner.py:331](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py), [:339](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py),
  [:379](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)).

### SVM — 99% limpo → 50% em todos os SNRs (o mais frágil)
- [ ] **Features robustas (RASTA-PLP)**: já há componente PLP em
  [app/domain/features/extractors/cepstral/components/plp.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/features/extractors/cepstral/components/plp.py);
  adicionar variante RASTA e usá-la no vetor tabular clássico
  (`_to_tabular_features` — [benchmarks/data.py:268](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/data.py)).
- [ ] **Augmentation ruidoso no espaço de feature**: gerar cópias ruidosas
  (SNR 5–40 dB) **antes** da extração e concatenar ao fit clássico
  (`_run_classical` — [benchmarks/runner.py:445](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)).
- [ ] **Limiar adaptativo por SNR**: estimar SNR na inferência e selecionar
  threshold por faixa (estende a calibração EER já existente —
  [trainer.py:1019](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/trainer.py)).

### Random Forest — 98% → 55% em 10 dB
- [ ] Mesmo tratamento do SVM (augmentation ruidoso + RASTA-PLP);
  `_run_classical` é compartilhado ([runner.py:445](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)).

### RawNet2 — 96% → 50% em 10 dB
- [ ] Reativar augmentation (hoje `False` — [runner.py:331](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/runner.py)).
- [ ] Treinar com ruído SNR + codec + reverberação (P2.0; RawBoost já adequado
  a raw-audio).

### WavLM — colapsa a 51% em 10 dB e fica abaixo do HuBERT
- [ ] **Ablação de fine-tuning**: varrer `learning_rate` (ex. 1e-5/3e-5/1e-4) e
  nº de épocas/camadas descongeladas; comparar com HuBERT (mesma faixa).
  Usar o tuner existente ([app/domain/models/training/hyperparameter_tuning.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/hyperparameter_tuning.py)).
- [ ] Augmentation ruidoso (P2.0) no fine-tuning.

### HuBERT — 92% → 50% em 10 dB
- [ ] Augmentation ruidoso (P2.0). Sem mudança de arquitetura.

### Ensemble — 77,7% em 30 dB, 52% em 10 dB (fusão não robusta)
- [ ] **Reponderar ramos robustos**: a fusão hoje usa pesos/estratégias em
  `DetectionService.detect_multi_model`; calibrar pesos **por robustez a ruído**
  (favorecer Conformer/RawGAT-ST/Hybrid/AASIST/Sonic Sleuth).
- [ ] Augmentation nos ramos treináveis (P2.0).
- [ ] Reavaliar `weighted_avg` vs `soft_voting` sob SNR.

---

## Prioridade 3 — Ajuste opcional

### EfficientNet-LSTM — 91%, maior latência, 82% em 10 dB
- [ ] Retune de baixa prioridade (augmentation P2.0 + ajuste de `dropout`/`lstm_units`
  — [optimized_training_config.py:287](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/optimized_training_config.py)).

### MultiscaleCNN — 99,73% mas 82% em 10 dB e 188 MB
- [ ] Augmentation P2.0.
- [ ] **Pruning + quantização**: usar
  [app/domain/models/training/quantization_aware.py](https://github.com/thierrybraga/XFakeSong/blob/main/app/domain/models/training/quantization_aware.py)
  (QAT→tflite int8) para reduzir os 188 MB.

---

## Robustos — sem retreino por ruído (candidatos de campo)
Conformer (94% @10dB), RawGAT-ST (88%), Hybrid CNN-Transformer (88%),
AASIST (84%), Sonic Sleuth (83%).
- [ ] **Ainda assim** revalidar sob split agrupado (P0) — Conformer e Sonic
  Sleuth (100% limpo) são os principais suspeitos de vazamento de fonte.

---

## Ordem de execução sugerida
1. **P0** (split por falante/gerador) — desbloqueia medições confiáveis.
2. **P2.0** (augmentation SNR-calibrado) — corrige a causa comum das quedas.
3. **P1** (Spectrogram Transformer) — sob val disjunto + augmentation.
4. **P2** específicos (SVM, RF, RawNet2, WavLM, HuBERT, Ensemble).
5. **P3** (EfficientNet-LSTM, MultiscaleCNN).
6. Revalidação final de **todos** sob P0 + relatório cross-generator (P0.4).

## Critérios de aceite globais
- Gap val−teste < 5 pp em todos os modelos retreinados.
- Sem queda > ~15 pp de acurácia a 10 dB para modelos P2 (deixam de ser
  "inúteis em campo").
- Conformer/Sonic Sleuth: números sob split agrupado documentados (mesmo que
  menores) — defensável na banca.
