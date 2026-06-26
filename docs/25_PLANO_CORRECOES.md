# 25 — Plano de Ação: Correções do Benchmark

> Criado em 2026-06-26. Origem: análise crítica dos resultados consolidados
> (`results/tcc_consolidated/benchmark_summary.json`, corrida `20260626`).
> Relacionado: [21_PLANO_RETREINO.md](21_PLANO_RETREINO.md),
> [22_RETREINO_WSL2.md](22_RETREINO_WSL2.md), [20_ESTUDO_EXPERIMENTAL.md](20_ESTUDO_EXPERIMENTAL.md).

Organizado por prioridade. **P0 invalida os números atuais como medida de
generalização**, então todo "resultado" depende dele. P1–P2 são correções
técnicas de modelos; P3 é texto/reprodutibilidade.

---

## P0 — Metodologia / vazamento de domínio *(bloqueante)*

O dataset `benchmark_audio_raw_balanced_15k.npz` tem um confound fonte↔classe e
foi avaliado com **split estratificado** (`group_split=false`,
`holdout_generator=null`), que compartilha fontes/geradores entre treino e
teste. As acurácias ~100% medem desempenho **in-domain**, não generalização.

### P0.0 — Auditoria forense do dataset ✅ *(concluída — sem GPU)*

Script: [`scripts/audit_dataset_leakage.py`](../scripts/audit_dataset_leakage.py).
Relatório: `results/audit_dataset_leakage/leakage_report.{json,md}`.

**Evidências obtidas:**

| Fonte | real | fake | observação |
|---|---:|---:|---|
| brspeech | 5000 | 5000 | — |
| cvpt | 2500 | 0 | **fonte pura → atalho** |
| fkvoice (XTTS) | 0 | 2500 | **fonte pura → atalho** |

1. **Atalho por fonte pura:** **32,9% do teste (741/2250)** é trivialmente
   separável só pela identidade da fonte (cvpt→real, fkvoice→fake aparecem em
   treino *e* teste).
2. **Atalho intra-fonte (brspeech):** uma regressão logística sobre ~10 features
   globais triviais (sem nenhuma pista fina de spoofing) separa real/fake do
   brspeech com **77,2% de acurácia / AUC 0,843**. O discriminador dominante é o
   **offset DC (74,8% sozinho, AUC 0,731)**, seguido de RMS/energia — assinatura
   clara de artefato sistemático de geração/pré-processamento, não de
   falsificação.

**Conclusão:** os ~100% são inflados por *dois* atalhos sobrepostos (identidade
de fonte + artefato de preço-processamento intra-brspeech). Nenhum dos dois
sobrevive a um protocolo gerador-disjunto.

**Decisão derivada (entra em P0.3/build):** neutralizar o offset DC e a loudness
no `build_dataset.py` (remoção de média + peak-normalization consistente entre
classes; trim de silêncio uniforme). Isso ataca o atalho global trivial; o
atalho de fonte só some com o protocolo cross-generator (P0.1).

### P0.1 — Re-rodar no protocolo cross-generator ⏳ *(o teste mais importante)*

```bash
python scripts/run_benchmark.py --full \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --cross-generator fkvoice \
  --out results/xgen_fkvoice_20260626 --verbose
```

Treina sem o XTTS e testa nele → generalização a gerador inédito. Infra pronta
([run_benchmark.py:101](../scripts/run_benchmark.py),
[data.py:290](../benchmarks/data.py)). **GPU/WSL2.**
**Aceite:** `results.json` com `holdout_generator="fkvoice"`; EER do Sonic Sleuth
deixa de ser 0,00% (degradar é o resultado *honesto*).

### P0.2 — Re-rodar no split disjunto por grupo ⏳

```bash
python scripts/run_benchmark.py --full \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --group-split \
  --out results/group_split_20260626 --verbose
```

**Ressalva:** 3 grupos, dois puros de classe → teste pode ficar desbalanceado.
Complementar a P0.1. Validar composição do teste antes de confiar.

### P0.3 — Ampliar diversidade de geradores ⏳ *(médio prazo)*

Hoje há 1 só gerador fake (XTTS). Cross-generator com N=1 é frágil. Adicionar ≥1
gerador a `download_datasets.py`/`build_dataset.py` + aplicar a normalização
decidida em P0.0. **Aceite:** dataset com ≥2 geradores fake; leave-one-generator-out.

---

## P1 — Instabilidade de treino (família Transformer)

**Causa-raiz identificada (não era lr/warmup):** o run `20260626` já usava o
config curado (patience 20 → parada na época 7+20=27, lr 2e-5, warmup 3000) e
**mesmo assim** colapsou (`final_val=0,5`). O elo faltante era **clip de
gradiente nunca chegando ao otimizador do AST/CCT**: `create_warmup_cosine_optimizer`
construía o AdamW sem `global_clipnorm` (ao contrário do Conformer, que threada
`clipnorm=1.0` no próprio otimizador e é estável). Um pico de gradiente na
atenção, após o warmup, explodia os pesos → val→0,5; só o `restore_best_weights`
salvava o número de teste.

### P1.1 — Spectrogram Transformer (`final_val=0,5` = colapso) ✅ *(código aplicado)*

**Ações feitas:**
- `create_warmup_cosine_optimizer(..., clipnorm=1.0)` → AdamW `global_clipnorm`
  ([optimization.py](../app/domain/models/training/optimization.py)). Fix central,
  beneficia AST **e** CCT.
- AST builder aceita `clipnorm` (default 1.0) e o repassa; `warmup_steps` default
  1000→2000 ([spectrogram_transformer.py](../app/domain/models/architectures/spectrogram_transformer.py)).
- `clipnorm: 1.0` adicionado ao plano recomendado do AST
  ([planning.py](../benchmarks/planning.py)); `EarlyStopping` em `val_loss` já era
  o monitor correto.
- Regressão: `tests/unit/test_benchmark.py` agora exige `ast["clipnorm"]==1.0`;
  verificado end-to-end que o otimizador do AST sai com `global_clipnorm=1.0`.

**Aceite (pende re-treino GPU):** `final_val` a ≤2 p.p. do `best_val` — confirmar
no próximo run cross-generator (P0.1).

### P1.2 — Hybrid CNN-Transformer (`best_val=0,972 → final_val=0,765`) ✅ *(código aplicado)*

Mesma causa, mais branda. `clipnorm=1.0` explícito no builder do CCT
([hybrid_cnn_transformer.py](../app/domain/models/architectures/hybrid_cnn_transformer.py)).
**Aceite (pende re-treino):** `final_val ≥ best_val − 3 p.p.`

---

## P2 — Calibração / robustez do Ensemble 🔄 *(diagnóstico + fixes no-GPU feitos; raiz pende re-treino)*

**Diagnóstico com evidência** (a partir de `metrics.json` do run 20260626):
1. Os scores limpos do Ensemble são **96% saturados em {0,1}** (48% <0,01, 48%
   >0,99) — fusão aprendida (gated MLP→sigmoid) grosseiramente superconfiante,
   treinada com `binary_crossentropy` puro (sem label smoothing).
2. **`use_augmentation=False`** no run → o Ensemble **nunca viu ruído** no treino.
3. Sob AWGN o score colapsa para ~0 mantendo a ORDENAÇÃO: a 30 dB **AUC 0,95**
   mas acc@0,5 = 0,739; a 10 dB AUC cai a 0,665 (degradação real).
4. `clean.eer_threshold = 0,524` (≈0,5): calibrar o limiar no limpo **não salva**
   (os scores ruidosos caem abaixo). Só limiar por-condição (oráculo) recupera:
   acc@EER ≈ 1−EER → 30 dB 0,885 · 20 dB 0,709 · 10 dB 0,612.

**Conclusão:** o "0,50" é, a 30 dB, majoritariamente artefato do limiar fixo; a
10 dB é fragilidade real (não viu ruído). Calibração de limiar sozinha não
resolve — a raiz é a fusão superconfiante + ausência de augmentation.

**Feito agora (no-GPU, testado):**
- **Métrica honesta `accuracy_at_eer`** em [evaluate.py](../benchmarks/evaluate.py):
  acurácia no ponto de EER, separando "falha de limiar" de "falha de
  separabilidade". Teste de regressão em `test_benchmark.py`
  (`test_accuracy_at_eer_separates_threshold_collapse_from_failure`).
- **`use_augmentation: True`** para o Ensemble em [planning.py](../benchmarks/planning.py)
  (corrige o `False` confirmado) — vale no próximo re-treino.

**Pende re-treino (queue com P0.1):**
- Label smoothing na loss da fusão em
  [ensemble.py:521-526](../app/domain/models/architectures/ensemble.py) (string
  `binary_crossentropy`/`sparse_categorical_crossentropy` → loss com
  `label_smoothing=0.05`) para curar a saturação {0,1}. Não feito agora por mexer
  em loss custom + serialização do `.keras` salvo, e só ter efeito com re-treino.
- **Aceite (pós re-treino):** acc@0,5 do Ensemble a 10 dB ≥ melhor membro, ou ao
  menos acc@0,5 ≈ accuracy_at_eer (scores estáveis sob ruído).

**Opção de deploy (runtime, sem re-treino):** `detect_multi_model(..., weights="robustness")`
já existe ([detection_service.py](../app/domain/services/detection_service.py)) e
funde os modelos individuais robustos (Conformer/MultiscaleCNN) ponderando por
robustez — caminho alternativo a um Ensemble treinado frágil.

---

## P3 — Texto do TCC + reprodutibilidade ⏳

- **P3.1** Reescrever o parágrafo de resultados: números in-domain rotulados como
  *teto otimista* + tabela cross-generator como estimativa honesta + ressalva do
  confound fonte↔classe; para o Spectrogram Transformer, deixar claro que o
  número depende da restauração do checkpoint.
  Arquivos: `tcc_overleaf/tcc.tex`, `tcc_overleaf/tabelas_benchmark.tex`.
- **P3.2** Documentar o protocolo em [15_BENCHMARK.md](15_BENCHMARK.md) e
  [20_ESTUDO_EXPERIMENTAL.md](20_ESTUDO_EXPERIMENTAL.md) (estratificado vs grupo
  vs cross-generator) + registrar esta auditoria.
- **P3.3** Consolidar figuras/tabelas a partir da corrida honesta:
  `python scripts/consolidate_results.py --input results/xgen_fkvoice_20260626 …`.

---

## Sequenciamento e dependências

```
P0.0 (auditoria) ✅ ─┬─> P0.1 (cross-generator) ──┐
                     └─> P0.2 (group split)        │
P1.1 + P1.2 (estabilizar) ──> [revalidar]         ├─> P3.1/P3.3 (texto+figuras)
P2 (calibrar ensemble) ─────> [revalidar]         │
                                    P3.2 (docs) ───┘
```

- **Caminho crítico:** P0.0 ✅ → P1 (estabilizar antes de re-treinar) → P0.1 → P3.
- **Custo:** P0.1/P0.2 são preset `--full` (14 arq × 100 épocas) — corrida longa.
  Rodar `--neural` primeiro para iterar; depois `--full`.
- **Esforço:** P1 ~2–3h + re-treino · P2 ~2h + re-eval · P3 ~2h.
