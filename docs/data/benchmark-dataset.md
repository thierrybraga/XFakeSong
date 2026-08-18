# Dataset do benchmark

Descrição operacional do artefato que alimenta treino e benchmark. A metodologia
completa — por que cada decisão foi tomada e o que foi medido — está no
[Protocolo de Dataset](dataset-protocol.md).

## Resumo executivo

`data/datasets/benchmark_dataset.npz` — **CETUC pareado com clones XTTS-v2**.
Cada amostra falsa é o clone sintético do **mesmo locutor lendo a mesma frase**
que a amostra real correspondente.

- 40.980 amostras: 20.490 reais e 20.490 falsas;
- áudio bruto mono, 16 kHz, janela de 3 s — entrada `(48000, 1)`;
- treino, validação e teste **não compartilham locutor nem frase**;
- classes balanceadas por construção, não por cota;
- toda amostra é recorte central puro — nenhuma é repetida (`tile`);
- escopo científico **in-domain**: um único gerador (XTTS-v2).

## Partições

| Split | Real | Fake | Total | Locutores | Frases | Horas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| treino | 16.613 | 16.613 | 33.226 | 34 (23F/11M) | 602 | 45,47 |
| validação | 1.988 | 1.988 | 3.976 | 11 (7F/4M) | 201 | 5,49 |
| teste | 1.889 | 1.889 | 3.778 | 11 (7F/4M) | 197 | 5,17 |

Os locutores são repartidos 60/20/20 entre os 56 pareados, **estratificados por
sexo**; as frases, 600/200/resto entre os 997 grupos de conteúdo. Uma amostra só
entra numa partição quando o **locutor e a frase** pertencem àquela partição — o
bloco diagonal do grid. As combinações fora da diagonal são descartadas, e é
isso que faz o aproveitamento ser de 41,6% do corpus.

## Contrato técnico do NPZ

| Item | Valor |
| --- | --- |
| Arquivo | `benchmark_dataset.npz` |
| Tamanho | 7,99 GB (sem compressão) |
| SHA-256 | `ae3662c9e2cc904c9c2f06bc541f27fe944baea4a527591cdb127a1d00f62ad7` |
| Sample rate | 16.000 Hz |
| Janela | 48.000 amostras / 3 s, recorte central |
| Formato | `raw_audio`, `(N, 48000, 1)`, `float32` |
| Rótulos | `0=real`, `1=fake` |
| Amplitude | RMS normalizado a −26 dBFS **na janela**, teto de pico −1 dBFS |
| Semente | 42 |
| Estratégia | `speaker_x_sentence_double_disjoint_block_diagonal` |

Arrays:

- áudio e rótulos: `X_train/y_train`, `X_val/y_val`, `X_test/y_test`;
- procedência: `sample_paths`, `source_ids`, `groups`, `content_sha256`;
- identidade: `speaker_ids`, `speaker_known`, `utterance_ids`, `text_ids`,
  `sentence_indices`, `cetuc_official_split`;
- síntese: `generator_ids`, `generator_known`;
- agrupamento: `cluster_ids` (= `text_id`, 997 clusters) para bootstrap;
- janela: `original_num_samples_*`, `window_start_*`;
- contrato completo e auditorias embutidas: `metadata_json`.

**Toda a procedência está preenchida**:
`speaker_known` e `generator_known` são `True` em 100% das amostras, porque as
duas fontes publicam locutor e texto.

## Evidências de integridade

Auditoria de 26/07/2026 (`data/datasets/splits/audit_report.json`),
reproduzível com `python scripts/dataset/audit_paired_corpus.py`:

| Verificação | Resultado |
| --- | ---: |
| Locutor compartilhado entre partições | 0 |
| Frase compartilhada entre partições | 0 |
| Texto compartilhado entre partições | 0 |
| Enunciado compartilhado entre partições | 0 |
| SHA-256 de áudio compartilhado entre partições | 0 |
| Quase-duplicatas entre partições (cosseno ≥ 0,99) | 0 em 272,7 milhões de comparações |
| Redundância interna (gêmeas quase idênticas) | 8 pares em 40.980 (0,04%) |
| Enunciados sem par | 0 |
| Locutores desbalanceados | 0 |

Oráculos de maioria (acaso = 50%): `source` 50,00% · `speaker_id` 50,00% ·
`sentence_index` 50,00% · `text_id` 50,00%.

## Garantias e não garantias

Garantido e verificado no artefato:

- balanceamento 1:1 global, por partição **e por locutor**;
- zero repetição de amostra, conteúdo, locutor, frase e texto entre partições;
- nenhuma variável de procedência prediz a classe acima do acaso;
- nível de áudio neutralizado (`rms_db` AUC 0,574) e repetição de janela zerada.

**Não** garantido:

- **generalização cross-generator.** Toda a classe falsa é XTTS-v2. Esta é a
  limitação dominante e deve acompanhar qualquer métrica publicada;
- generalização cross-corpus ou para outros canais de gravação — o CETUC é
  gravação de estúdio, em condições controladas.

Um detector trivial de descritor único chega a **AUC 0,67** neste corpus (fator
de crista, artefato genuíno de vocoder). Esse é o piso de leitura: um modelo
próximo disso não aprendeu mais do que uma estatística escalar.

## Uso

```bash
python scripts/benchmark/run_models_sequential.py --dataset data/datasets/benchmark_dataset.npz --models AASIST Ensemble --epochs 100 --snr 30 20 10 --device-profile gpu --out data/results/<run> --resume
```

O corpus completo (49.264 pares, 98.528 amostras, 133,1 h) fica em
`data/datasets/corpus/` e a partição em `data/datasets/splits/`. O `.npz`
acima já contém a partição inteira; para reduzi-lo ao que couber na memória de
treino use `--max-pairs-train` no exportador, que corta **pares** e preserva o
balanceamento (ver [Protocolo de Dataset, §9.1](dataset-protocol.md)).

A janela caiu de 5 s para 3 s e o contrato de inferência (`source_samples`) já
está em 48.000 em todo o código — ver [Protocolo de Dataset, §9.2](dataset-protocol.md#92-janela-fonte-migração-concluída).

## Estado atual: nenhum modelo treinado sob este protocolo

`data/models/` está **vazio**. Os artefatos `.npz` de protocolos anteriores
foram apagados — um deles tinha composição que permitia acertar 87,6% dos
rótulos apenas identificando o corpus de origem, sem detectar síntese alguma —
e o último resíduo de modelo (`bench_svm.pkl`, incompatível com o front-end
tabular atual) foi removido de `data/models/`. `data/results/benchmark/` e
`data/results/reporting/` também estão vazios.

Isso significa que **nenhuma das 14 arquiteturas foi treinada ou avaliada sob
o protocolo `speaker_x_sentence_double_disjoint_block_diagonal` atual**. As
métricas em `data/results/paper/tabelas_benchmark.tex` (acurácia, EER, params,
etc.) são de uma rodada anterior ao protocolo vigente e **não devem ser
citadas como resultado deste dataset** — precisam ser regeneradas por um novo
run do benchmark (`scripts/benchmark/run_benchmark.py` /
`run_models_sequential.py`).
