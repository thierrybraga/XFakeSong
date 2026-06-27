# 27 — Dataset: pipeline, consolidacao e auditoria

Referencia unica do ciclo de vida do dataset no XFakeSong — do download a criacao
do `.npz`, passando pela interface Gradio, pelo benchmark e pelo treino. Inclui o
resultado da auditoria de consistencia entre essas pecas.

Fonte unica de verdade: **`app/core/dataset_catalog.py`** (`DATASET_CATALOG` +
`DATASET_TIERS`). Esse modulo e importado por `scripts/build_dataset.py`, pela aba
Datasets do Gradio (`app/interfaces/gradio/tabs/dataset_management.py`), pelo
benchmark e pela documentacao — garantindo que tamanho, fontes e split nao
divirjam entre os componentes.

---

## 1. Catalogo de fontes (10)

| Fonte | Tipo | Idioma | Licenca | Prefixo(s) | Flag CLI | Full-bench |
|---|---|---|---|---|---|:--:|
| BRSpeech-DF | real+fake | pt-BR | Apache-2.0/CC BY 4.0 | `brspeech` | `--brspeech` | sim |
| Fake Voices (XTTS) | fake | pt-BR | MIT | `fkvoice`,`fakevoice` | `--fake-voices` | sim |
| FLEURS | real | pt-BR | consultar oficial | `fleurs` | `--fleurs` | sim |
| CETUC | real | pt-BR | livre/variavel | `cetuc` | `--cetuc` | sim |
| Common Voice PT | real | pt | CC0 | `cvpt`,`cv` | `--common-voice-pt` | sim |
| MLAAD-PT | fake | pt | CC-BY-NC 4.0 | `mlaad` | `--mlaad-pt` | nao (NC) |
| ASVspoof 2019 | real+fake | ingles | ODC-BY 1.0 | `asv2019` | `--asvspoof2019` | sim |
| WaveFake | fake | ingles/japones | CC-BY-SA 4.0 | `wavefake` | `--wavefake` | sim |
| In-the-Wild | real+fake | ingles | Apache-2.0 | `itw` | `--in-the-wild` | sim |
| ASVspoof 5 | real+fake | ingles | CC BY 4.0 | `asv5` | `--asvspoof5` | nao |

O **prefixo** do nome do arquivo (`<prefixo>_NNNNN.wav`) e o que liga cada amostra
a sua fonte/gerador em todo o sistema (catalogo, auditoria de vazamento, `groups`
no `.npz`, splits cross-generator).

---

## 2. Tiers de tamanho (`test` · `small` · `medium` · `large`)

| Tier | Por classe | Total | Split | Fontes | Habilita |
|---|---:|---:|---|---|---|
| `test` | 100 | 200 | 70/15/15 estratificado | BRSpeech-DF, Fake Voices | smoke (nada de desempenho) |
| `small` | 1.000 | 2.000 | 70/15/15 estratificado | BRSpeech-DF, Fake Voices | Classico + CNN leve |
| `medium` | 3.000 | 6.000 | 70/15/15 estratificado | + Common Voice PT, FLEURS | ate Transformer |
| `large` | 10.000 | 20.000 | **disjunto por falante** | + Common Voice PT, FLEURS | todas as 14 (Ensemble) |

`small/test` puxam a classe real inteiramente do BRSpeech-DF bonafide
(`skip_real_cv`); `medium/large` adicionam Common Voice PT + FLEURS. Apenas o
`large` ativa `speaker_aware` (manifesto de falante + split disjunto).

---

## 3. Ciclo de vida (download -> npz)

```
build_dataset.py --tier <t>
   │
   ├─ step_download()    fontes do tier (download_datasets.py por flag) -> datasets/real|fake/*.wav
   ├─ step_balance()     balanceamento 1:1 por classe (arquiva/descarta excesso)
   ├─ step_preprocess()  preprocess_dataset.py: normaliza + cria splits
   │       ├─ estratificado (StratifiedShuffleSplit)        — test/small/medium
   │       └─ disjunto por falante (StratifiedGroupKFold)   — large (usuarios nao vistos)
   └─ save_dataset_config()  grava configs/.../dataset_config.json (tier, split, speakers)

run_tcc_pipeline.py --download --tier <t> --full-benchmark
   └─ export_npz_from_splits()  WAVs -> .npz canonico em AUDIO BRUTO (samples, 1)
```

Comandos:

```bash
python scripts/build_dataset.py --tier small        # 1.000/classe
python scripts/build_dataset.py --tier large        # 10.000/classe, split por falante
python scripts/build_dataset.py --tier medium --target 4000   # override de tamanho

# ponta a ponta (download + npz + benchmark) no tier large:
python scripts/run_tcc_pipeline.py --download --tier large --full-benchmark --speaker-split
```

### Esquema do `.npz` canonico

`np.savez_compressed` grava:

| Chave | Conteudo |
|---|---|
| `X_train`,`X_val`,`X_test` | audio bruto `(amostras, 1)` por split |
| `y_train`,`y_val`,`y_test` | rotulos `0=real`, `1=fake` |
| `groups` | fonte/gerador por amostra (do prefixo) |
| `speaker_ids` | falante por amostra (tier large; cai p/ fonte quando sem manifesto) |
| `metadata_json` | splits, contagens, `paths`, `source_summary`, duracao, sample_rate |

---

## 4. Consolidacao com o benchmark

`benchmarks/data.py::BenchmarkData.from_npz` e o ponto de uniao:

- **Concatena** `X_train+X_val+X_test` (ou `X/y`) e **re-divide de forma
  estratificada e reprodutivel** (`stratified_split`, `val_frac=0.15`,
  `test_frac=0.15`, semente fixa) — o conjunto de teste do benchmark e controlado,
  independente de como o `.npz` foi originalmente splitado.
- Extrai `groups` e `speaker_ids` (ou deriva dos `paths`/`speaker_manifest`).
- Habilita os protocolos avancados via `run_benchmark.py`:
  `--speaker-split` (disjunto por falante), `--unseen-speaker <fonte:id>` (holdout
  de falante) e holdout de gerador (cross-generator, XTTS=`fkvoice`).

Ou seja: o mesmo `.npz` serve treino e benchmark; o benchmark apenas reimpoe seu
proprio split estratificado/por-falante para garantir reprodutibilidade e o
protocolo de usuario nao visto.

---

## 5. Uso no pipeline de treino

`app/domain/services/training_service.py` carrega o `.npz` exigindo
`X_train`/`y_train`; usa `X_val`/`y_val` se presentes, senao o
`SecureTrainingPipeline` cria os splits 70/15/15 com checagem de vazamento. A
config global (LR, early stopping, augmentation com `snr_range_db`, calibracao)
fica em `app/core/config/settings.py`. Hiperparametros por modelo no
`registry.py` + `create_model` de cada arquitetura.

`run_models_sequential.py` -> `run_benchmark.py --model <nome>` e o caminho usado
no benchmark/retreino, treinando um modelo por vez com o mesmo `.npz`.

---

## 6. Interface Gradio

A aba **Datasets** (`dataset_management.py`) consome o mesmo catalogo:
`get_tier`, `tier_reference_markdown`, `DATASET_CATALOG`. O usuario escolhe um
tier (radio), que pre-preenche alvo por classe + fontes; ha plano de download
balance-aware, barra de balanceamento, distribuicao por classe/fonte e um quadro
de **prontidao por familia de modelo**. Os limiares de prontidao
(`Classico 300`, `CNN leve 1.000`, `CNN/RNN 2.000`, `Transformer 4.000`,
`Ensemble 6.000` por classe) sao um eixo distinto dos tiers de tamanho — medem
quando cada modelo pode treinar, nao o tamanho do dataset.

---

## 7. Auditoria — resultado

**Consistente (OK):**

- Catalogo unico compartilhado por build, Gradio, benchmark e docs.
- Tiers identicos em todos os pontos (catalogo -> build -> Gradio -> docs).
- Split 70/15/15 coerente entre criacao (`preprocess_dataset.py`) e benchmark
  (`benchmarks/data.py`); `large` mantem disjuncao por falante nos dois.
- Esquema do `.npz` (X/y por split + `groups` + `speaker_ids` + `metadata_json`)
  e lido corretamente pelo benchmark e pelo treino.
- Rotulos padronizados `0=real`/`1=fake` em todo o fluxo.

**Pontos de atencao / a reconciliar:**

1. **Nome x tamanho do `.npz` canonico — CORRIGIDO.** O `.npz` canonico foi
   padronizado em `benchmark_audio_raw_balanced_20k.npz`, coerente com
   `target_per_class: 10000` (= 20.000 total, tier `large`). O rename foi
   aplicado em todas as referencias forward (configs, scripts, environments e
   docs). O registro historico do ultimo benchmark
   (`app/models/benchmark_final/**` e `tcc_overleaf/main.tex`) foi **preservado**
   com o nome antigo (`...15k.npz`), pois descreve a execucao ja realizada. Como
   o `.npz` nao e versionado (gerado sob demanda), renomeie/regerar o arquivo
   fisico antes do proximo treino.
2. **Dois conceitos de "tier" — CENTRALIZADO.** Tamanho de dataset (catalogo:
   100/1.000/3.000/10.000) vs. prontidao por familia de modelo (300/1.000/2.000/
   4.000/6.000) sao eixos diferentes e legitimos. A lista de prontidao deixou de
   ser hardcoded no Gradio: agora vive em `dataset_catalog.py`
   (`MODEL_READINESS_TIERS` / `ModelReadinessTier`) e `dataset_management.py` a
   consome — fonte unica, sem duplicacao.
3. **`.npz` nao versionado.** A proveniencia exata do arquivo usado no ultimo
   benchmark vem do `metadata_json` interno e do `dataset_config.json`, nao do
   git. Mantenha esses sidecars junto aos resultados ao promover artefatos.

