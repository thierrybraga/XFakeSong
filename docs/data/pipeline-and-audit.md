# Dataset: pipeline, consolidacao e auditoria

Referencia unica do ciclo de vida do dataset no XFakeSong — da aquisicao ao
`.npz`, passando pela particao, pela auditoria, pelo benchmark e pelo treino.

A metodologia (por que cada decisao foi tomada, o que foi medido) esta no
[Protocolo de Dataset](dataset-protocol.md). Aqui esta o **como**.

Fonte unica de verdade do catalogo de fontes:
**`app/domain/dataset_metadata/dataset_catalog.py`** (`DATASET_CATALOG` +
`DATASET_TIERS`), importado pelos scripts de dataset, pela aba Datasets do
Gradio, pelo benchmark e pela documentacao.

---

## 1. Ciclo de vida canonico (Protocolo de Dataset)

Quatro scripts, nesta ordem. Cada um produz um artefato auditavel e resumivel.

```
build_paired_pt_corpus.py --build
   │   baixa CETUC (bonafide) + Fake Voices/XTTS (spoof) em revisoes fixadas,
   │   alinha clone -> original por enunciado, canonicaliza e normaliza nivel
   └─> data/datasets/corpus/{real,fake}/<LOCUTOR>/*.wav
       + manifest.jsonl, sentences.json, acquisition.json, state.json

build_paired_splits.py --build
   │   particao de disjuncao DUPLA (locutor x frase), bloco diagonal;
   │   exclui duplicatas da fonte e pares mais curtos que a janela
   └─> data/datasets/splits/{train,val,test}/{real,fake}/*.wav  (hardlinks)
       + assignment.jsonl, split_manifest.json, sentence_partition.json
       + sincroniza data/datasets/metadata/speaker_manifest.json

audit_paired_corpus.py
   │   6 blocos de verificacao; sai com codigo != 0 se uma garantia falhar
   └─> data/datasets/splits/audit_report.json

export_paired_npz.py
   │   le assignment.jsonl (nao os diretorios), recorta a janela, nivela,
   │   e barra a exportacao se qualquer garantia for violada
   └─> data/datasets/benchmark_dataset.npz
```

Comandos:

```bash
python scripts/dataset/build_paired_pt_corpus.py --plan
```

```bash
python scripts/dataset/build_paired_pt_corpus.py --build
```

```bash
python scripts/dataset/build_paired_splits.py --build --min-duration-sec 3.0
```

```bash
python scripts/dataset/audit_paired_corpus.py
```

```bash
python scripts/dataset/export_paired_npz.py --out data/datasets/benchmark_dataset.npz --no-compress
```

### Propriedades de operacao

| Propriedade | Onde |
| --- | --- |
| Resumivel por locutor | `corpus/state.json`; um locutor interrompido no meio e reconstruido do zero |
| Revisoes fixadas | `REV_REAL` / `REV_FAKE` no builder; avisa se o upstream avancar |
| Poda de cache | a cada locutor — sem ela os 56 pacotes somariam ~40 GB de cache inutil no Windows |
| Custo de disco | corpus 14,4 GB · splits em hardlink (sem custo extra) · NPZ 7,4 GB |
| Tempo | ~25 min de aquisicao com `hf_xet` instalado |

---

## 2. Estrutura em disco

```
data/datasets/
├── corpus/                 # verdade unica: tudo o que as duas fontes tem em comum
│   ├── real/<CODIGO>/ptpair_<CODIGO>_<NNNN>_bonafide.wav
│   ├── fake/<CODIGO>/ptpair_<CODIGO>_<NNNN>_xttsv2.wav
│   ├── manifest.jsonl         # uma linha por amostra, com proveniencia completa
│   ├── sentences.json         # as 1000 frases canonicas + divergencias detectadas
│   └── acquisition.json       # revisoes, cobertura, politica de audio, contagens
├── splits/                 # particao (hardlinks para o corpus)
│   ├── {train,val,test}/{real,fake}/*.wav
│   ├── assignment.jsonl       # o manifesto + o campo `split` de cada amostra
│   ├── split_manifest.json    # resumo, mapa locutor->particao, cobertura
│   └── audit_report.json      # saida da auditoria
├── metadata/speaker_manifest.json
└── benchmark_dataset.npz
```

O **prefixo** do nome do arquivo (`ptpair_`) liga a amostra a sua fonte em todo o
sistema. Ele e **o mesmo nas duas classes**, de proposito: um prefixo por classe
daria um oraculo de fonte de 100% (ver [Protocolo de Dataset, §3.1](dataset-protocol.md)).
A classe vem do diretorio e do campo `label` do manifesto.

---

## 3. Esquema do `.npz`

| Chave | Conteudo |
| --- | --- |
| `X_train`,`X_val`,`X_test` | audio bruto `(N, 48000, 1)` `float32` por particao |
| `y_train`,`y_val`,`y_test` | rotulos `0=real`, `1=fake` |
| `groups`,`source_ids` | fonte por amostra (`ptpair` nas duas classes) |
| `speaker_ids`,`speaker_known` | locutor; **100% conhecido** |
| `utterance_ids`,`text_ids`,`sentence_indices` | enunciado e conteudo |
| `generator_ids`,`generator_known` | `bonafide` / `xtts_v2`; 100% conhecido |
| `cluster_ids` | unidade do bootstrap = `text_id` (997 clusters reais) |
| `content_sha256` | identidade por conteudo, para auditoria externa |
| `cetuc_official_split` | particao publicada pelo CETUC, para comparacao |
| `sample_paths` | caminho no corpus, identidade e ordem |
| `original_num_samples_*`,`window_start_*` | rastreabilidade da janela |
| `metadata_json` | contrato, contagens por particao e **auditorias embutidas** |

`metadata_json` carrega as auditorias **dentro** do artefato: quem receber apenas
o `.npz` pode verificar sobreposicoes, balanceamento e oraculos sem o
repositorio.

---

## 4. Auditoria — os seis blocos

`scripts/dataset/audit_paired_corpus.py` roda sobre o artefato, nao sobre o
plano, e **falha com codigo != 0** — pode ir para CI antes de qualquer treino.

| Bloco | Verifica | Reprova quando |
| --- | --- | --- |
| A | pareamento, duplicatas exatas, arquivos ausentes, consistencia de texto | ha enunciado sem par, duplicata **na particao** ou arquivo faltando |
| B | disjuncao de locutor, frase, texto, enunciado e hash | qualquer sobreposicao entre particoes |
| C | balanceamento por particao e por locutor | classes desiguais |
| D | oraculo de maioria por variavel de proveniencia | acima de 55% |
| E | AUC de descritor unico **na janela que o modelo recebe** | RMS acima de 0,60 ou diferenca de repeticao acima de 0,02 |
| F | quase-duplicatas espectrais: F1 interna, F2 entre particoes | qualquer par acima de 0,99 atravessando particoes |

O bloco E distingue **empacotamento** (nivel, que nao carrega informacao sobre
sintese e tem de ficar no acaso) de **sinal** (crista, centroide, rolloff — o que
um detector legitimamente usa, apenas reportado). Essa distincao existe porque
tratar tudo igual foi o que quase deixou passar um atalho de nivel com AUC 0,99.

Barreira adicional: `export_paired_npz.py` repete as verificacoes de
sobreposicao, balanceamento e oraculo **depois** da selecao e do recorte, e
interrompe a gravacao se algo violar o protocolo. A particao ja foi auditada, mas
e o `.npz` que chega ao benchmark.

---

## 5. Consolidacao com o benchmark

`benchmarks/data.py::BenchmarkData.from_npz` e o ponto de uniao:

- **concatena** `X_train+X_val+X_test` numa visao comum, preservando os indices
  predefinidos — o teste nao e redividido por seed;
- carrega os vetores hierarquicos somente quando o alinhamento e exato;
- habilita os protocolos avancados de `run_benchmark.py`: `--speaker-split`,
  `--unseen-speaker <fonte:id>` e holdout de gerador.

Protocolos por grupo/falante/holdout sao fail-closed: metadado ausente, grupo
inexistente ou impossibilidade de manter as classes interrompem a execucao em vez
de cair para um split aleatorio.

**Custo de memoria:** `from_npz` materializa tudo em `float32` e ainda concatena.
As 40.980 amostras de 3 s dao ~16 GB de pico so para abrir o arquivo. Para
hardware menor, exporte com `--max-pairs-train`, que corta **pares** (nunca
amostras isoladas) em rodizio entre locutores, preservando o balanceamento.

---

## 6. Uso no pipeline de treino

`app/domain/services/training_service.py` carrega o `.npz` exigindo
`X_train`/`y_train`; usa `X_val`/`y_val` se presentes, senao o
`SecureTrainingPipeline` cria os splits com checagem de vazamento. A config
global (LR, early stopping, augmentation com `snr_range_db`, calibracao) fica em
`app/core/config/settings.py`; hiperparametros por modelo no `registry.py` e no
`create_model` de cada arquitetura.

`run_models_sequential.py` -> `run_benchmark.py --model <nome>` e o caminho do
benchmark/retreino, um modelo por vez sobre o mesmo `.npz`.

> **Pendencia no retreino:** a janela do protocolo e de 3 s (48.000 amostras), mas o
> contrato de inferencia ainda declara 80.000. A troca precisa acontecer **junto**
> com o retreino — ver [Protocolo de Dataset, §9.2](dataset-protocol.md).

---

## 7. Interface Gradio

A aba **Datasets** (`app/interfaces/gradio/tabs/dataset_management.py`) consome o
mesmo catalogo: `get_tier`, `tier_reference_markdown`, `DATASET_CATALOG`. Mostra
distribuicao por classe/fonte, barra de balanceamento e um quadro de **prontidao
por familia de modelo**.

Os limiares de prontidao (`Classico 300`, `CNN leve 1.000`, `CNN/RNN 2.000`,
`Transformer 4.000`, `Ensemble 6.000` por classe) vivem em
`dataset_catalog.py` (`MODEL_READINESS_TIERS`) e medem **quando cada modelo pode
treinar** — eixo distinto do tamanho do dataset.

---

## 8. Scripts fora do fluxo canonico

Estes scripts nao fazem parte do protocolo e continuam no repositorio por outras
razoes:

| Script | Papel | Situacao |
| --- | --- | --- |
| `download_datasets.py` | aquisicao multi-fonte por flag (13 fontes do catalogo) | **ativo** para fontes fora do protocolo — validacao externa, teste cross-corpus |
| `build_dataset.py` | composicao por tier (`small`/`medium`/`large`) + cotas por fonte | fora do fluxo |
| `preprocess_dataset.py` | normalizacao e criacao de splits estratificados | fora do fluxo |
| `build_clean_dataset.py` | deduplicacao por impressao espectral | fora do fluxo |
| `infer_speakers.py` | inferencia de falante por embeddings WavLM | fora do fluxo — o protocolo usa locutor publicado, nao inferido |
| `export_npz_from_splits.py` | exportador generico a partir de diretorios | fora do fluxo — use `export_paired_npz.py`, que le o manifesto |

O modelo de **tiers de tamanho** (`test`/`small`/`medium`/`large`) pertence a
`build_dataset.py`, nao ao protocolo. O protocolo nao usa cotas: o tamanho e o
que as duas fontes tem em comum, e a unica reducao possivel e o corte por pares
na exportacao.

Misturar as fontes puras de classe do catalogo no treino reintroduz o atalho de
fonte — use-as como conjunto externo de validade, nunca no treino
([Protocolo de Dataset, §2.2](dataset-protocol.md)).
