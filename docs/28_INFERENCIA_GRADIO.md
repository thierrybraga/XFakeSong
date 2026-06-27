# 28 — Verificacao Gradio/Inferencia e correcao de serving (P0)

Auditoria do caminho de inferencia
(Gradio -> `DetectionService` -> `ModelLoader` -> `FeaturePreparer` ->
`Predictor`) e a correcao P0 aplicada.

## O que esta OK (alinhamento treino/inferencia)

- **`input_contract`** salvo pelo trainer (`trainer._build_input_contract`) e a
  ponte de alinhamento: `input_shape`, `type/format`, `sample_rate`,
  `temperature` calibrada, `eer_threshold`, `ood_threshold`, `scaler_applied`,
  `label_classes`. O `FeaturePreparer` o prioriza sobre o fallback do registry e
  ja le `n_fft`/`hop_length`/`n_mels`/`n_lfcc` do contrato quando presentes.
- **Decisao calibrada** no `Predictor`: temperature scaling, **EER threshold**
  (nao 0.5 ingenuo), energy score (OOD) e MC-dropout (incerteza).
- **Carga Keras**: `custom_objects` cobre as camadas customizadas e os
  extratores SSL (WavLM/HuBERT); backbones SSL resolvidos de `benchmark_final/`.
- **Reamostragem** para o `sample_rate` do contrato (16 kHz) antes da extracao.

## Correcao aplicada — P0: servir os modelos treinados

**Problema:** `ModelLoader._discover_model_files` fazia `glob` nao-recursivo em
`app/models/*.keras|*.pkl`. Como os artefatos treinados ficam em
`app/models/benchmark_final/<arch>/bench_<arch>.*`, a descoberta achava **zero**
modelos e caia em `_create_default_models()` (MultiscaleCNN/EfficientNet-LSTM
**nao treinados**). A deteccao no Gradio rodava em pesos de demonstracao.

**Correcao:** a descoberta passou a incluir
`benchmark_final/*/bench_*.{keras,h5,pkl,pt}` (restrito ao prefixo `bench_` para
nao capturar `best_checkpoint.keras` intermediarios), com de-duplicacao por
caminho real. Verificado contra o `benchmark_final` atual: **14/14 modelos
descobertos**, cada um com seu `_config.json` (que carrega o `input_contract`),
e **0 checkpoints intermediarios** capturados.

## Itens remanescentes (proxima rodada)

- **[P1] Receita de feature dos modelos de espectrograma.** `feature_types` vem
  `None` no contrato; o `FeaturePreparer` ja le `n_mels/n_fft/hop` do contrato,
  mas eles nao sao gravados no treino. Gravar a receita completa em
  `_build_input_contract` (no retreino) elimina o mismatch potencial
  Mel/LFCC/CQT entre treino e inferencia.
- **[P2] Calibracao.** Recalibrar `temperature`/`eer_threshold` num conjunto de
  validacao representativo e **com ruido** apos o retreino.

Lista completa de melhorias de treino em `docs/RETREINO_AJUSTES.md`.
