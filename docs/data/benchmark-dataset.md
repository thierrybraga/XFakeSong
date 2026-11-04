# Dataset do benchmark

Descrição do corpus pareado CETUC–XTTS-v2 e dos artefatos de benchmark. A
referência metodológica do estudo do TCC é o [artigo canônico e inventário de
legado](../evaluation/canonical-study-and-legacy-inventory.md); o método de
construção está no [Protocolo de Dataset](dataset-protocol.md).

## Resumo executivo

!!! warning "Não confundir corpus pai e dataset avaliado no artigo"
    O artigo usa `data/datasets/benchmark_dataset_15k.npz` (15.000 amostras),
    cujo arquivo e test lock existem neste checkout local, mas não são
    distribuídos pelo Git. As seções abaixo descrevem o corpus pai de 40.980
    amostras e seus splits completos. A disponibilidade pública das fontes não
    confirma autorização para redistribuir o corpus derivado: valide os termos
    do CETUC, Fake Voices e XTTS, além da base ética para publicar áudios e
    identificadores, antes de hospedar WAVs ou NPZ.

O corpus pai CETUC pareado com clones XTTS-v2 tem 40.980 amostras após as
disjunções descritas. O arquivo integral `data/datasets/benchmark_dataset.npz`
é uma especificação/exportação histórica e **não está presente neste checkout**;
ele não é o recorte de 15 mil amostras avaliado no artigo.
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

## Contrato histórico da exportação integral do corpus pai

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
reproduzível com `python tools/dataset/audit_paired_corpus.py`:

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

## Reconstrução

O corpus de origem (49.264 pares, 98.528 amostras, 133,1 h) fica em
`data/datasets/corpus/` e as partições completas em `data/datasets/splits/`.
Os comandos para reconstruir o corpus pai e suas exportações estão no
[Protocolo de Dataset](dataset-protocol.md). Para o benchmark do TCC, siga o
NPZ de 15 mil amostras e a receita descrita no [inventário canônico](../evaluation/canonical-study-and-legacy-inventory.md).

A janela caiu de 5 s para 3 s e o contrato de inferência (`source_samples`) já
está em 48.000 em todo o código — ver [Protocolo de Dataset, §9.2](dataset-protocol.md#92-janela-fonte-migracao-concluida).

## Relação com o estudo do artigo

O artefato do artigo é `data/datasets/benchmark_dataset_15k.npz` (15.000
amostras), SHA-256
`3775b35fb05155011b21a30ff60632aaa7c9f67f031bd6f7874a5f313363400f`, com splits
12.162/1.456/1.382 e test lock correspondente. Portanto, a afirmação anterior
de que o NPZ e o lock estavam ausentes era incorreta.

A execução local `data/results/benchmark_15k_2026-08-20/` referencia o mesmo
dataset e contém 11 registros de modelo marcados `ok`, mas o resumo geral ainda
declara `running`, seus caminhos de saída apontam para uma pasta ausente, e as
métricas diferem das tabelas do artigo. As fontes dos valores consolidados
também apontam para diretórios ausentes. Assim, a existência do NPZ/lock não
fecha a rastreabilidade: **não apresentar as métricas do artigo como
confirmadas até reconciliar run, configuração, versão do código, logs,
checkpoints, predições e tabelas**. O [inventário canônico](../evaluation/canonical-study-and-legacy-inventory.md)
traz a comparação por modelo e os candidatos a legado.

O protocolo do artigo define 11 entradas oficiais, seed 42, até 100 épocas,
AWGN na forma de onda (treino 30/20/10 dB; teste 30/20/10/5 dB), seleção no
split de validação e 30 medidas de latência. Preserve os resultados históricos;
uma nova campanha deve gerar um diretório próprio e manifesto completo.
