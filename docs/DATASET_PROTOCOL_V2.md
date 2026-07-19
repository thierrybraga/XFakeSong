# Protocolo acadêmico de dataset v2

## Escopo

Este documento define as garantias exigidas para novos benchmarks do
XFakeSong. O arquivo histórico
`data/datasets/benchmark_audio_raw_balanced_15k.npz` não adquire essas
garantias retroativamente: ele deve ser regenerado a partir da camada bruta,
auditado e selado novamente como `xfakesong-test-lock-v2`.

Até essa regeneração, resultados obtidos com o artefato histórico devem ser
descritos como **legados e in-domain**. Eles não sustentam generalização para
falantes, textos, geradores ou domínios não vistos.

## Camadas e imutabilidade

- `data/datasets/real` e `data/datasets/fake`: aquisição bruta, imutável;
- `data/datasets/processed`: WAV mono PCM16 canônico e artefatos de auditoria;
- `data/datasets/splits`: cópias derivadas dos processados;
- NPZ: janela temporal e vetores de proveniência alinhados às amostras.

O pré-processamento nunca sobrescreve nem remove a aquisição. Reexecuções usam
um cache endereçado por SHA-256, tamanho e versão do pipeline.

## Áudio canônico

| Etapa | Política v2 |
|---|---|
| Decodificação | mono, `float32` |
| Reamostragem | `soxr_hq`, 16 kHz |
| Duração válida | 1 a 30 segundos |
| VAD | desativado; nenhum recorte de fala é declarado sem ser executado |
| Amplitude | preservada; apenas atenuação anti-clipping acima de 1,0 |
| Janela curta | repetição (`tile`), sem zero-padding |
| Janela longa | recorte central |
| Janela exportada | 5 s por padrão, com comprimento original e início registrados |

A mesma política temporal é usada pelo frontend de benchmark/runtime. Não é
permitida normalização de pico exclusiva de uma das rotas.

## Proveniência hierárquica

Cada amostra pode registrar:

- fonte e revisão imutável do repositório;
- falante;
- enunciado e texto/conteúdo;
- gerador e vocoder;
- codec e canal;
- rótulo.

IDs ausentes permanecem explicitamente desconhecidos. Eles não podem ser
substituídos por uma fonte coletiva e chamados de “falante”. `speaker_split` e
`holdout_speaker` falham se a cobertura `speaker_known` não for completa.

O NPZ v2 contém `sample_paths`, `source_ids`, `speaker_ids`, `speaker_known`,
`utterance_ids`, `text_ids`, `generator_ids`, `generator_known` e
`cluster_ids`. O cluster prefere texto, enunciado, falante e, sem metadados, a
própria amostra.

## Seleção, balanceamento e splits

- o alvo por classe é exato; falta de dados interrompe o build;
- quotas por fonte são contratuais e verificadas mesmo se o total já for exato;
- a seleção dentro de cada fonte é aleatória com semente registrada;
- textos/enunciados repetidos são mantidos em uma única partição;
- splits por falante/grupo falham quando não há grupos suficientes;
- nenhum fallback aleatório pode quebrar a disjunção solicitada;
- cada partição deve conter as duas classes.

O tier medium mantém as quotas históricas (3.750 BRSpeech reais, 1.875 MLS,
1.875 TTS-Portuguese; 3.750 BRSpeech falsos e 3.750 Fake Voices), mas isso não
elimina o confundimento fonte-classe. Por isso ele é um benchmark in-domain.

## Duplicatas e leakage

1. duplicatas exatas usam SHA-256 do PCM mono decodificado;
2. áudio idêntico com rótulos opostos interrompe o pipeline;
3. quase-duplicatas são auditadas por fingerprint log-Mel e similaridade
   cosseno; conflitos entre rótulos interrompem o pipeline;
4. candidatos do mesmo rótulo são relatados, não removidos automaticamente;
5. sobreposição de conteúdo entre train/val/test deve ser zero;
6. o benchmark repete uma auditoria byte a byte das partições.

## Atalho por fonte

O pipeline calcula a acurácia de um oráculo que prevê, para cada fonte, sua
classe majoritária. Valores acima de 55% marcam confundimento. No protocolo
acadêmico isso é erro fatal (`--fail-on-source-shortcut`); fora dele, o
resultado recebe escopo explícito `in-domain`.

Uma composição com fontes exclusivas de apenas uma classe não pode ser usada
para afirmar generalização cross-domain, ainda que o balanceamento global seja
50/50.

## Avaliação

- os intervalos de confiança usam bootstrap por `cluster_ids`, não IID por
  segmento;
- são reportadas métricas por fonte/gerador, macro e worst-group;
- aumento comum é o padrão: mesma cópia AWGN estática para todas as
  arquiteturas;
- aumento específico de AASIST/RawGAT-ST é uma ablação opt-in e não pode ser
  misturado à tabela comparativa principal;
- o selo v2 inclui identidade de teste, clusters, fontes e caminhos.

## Reconstrução

```powershell
python scripts/dataset/build_dataset.py --tier medium --seed 42
python scripts/dataset/export_npz_from_splits.py `
  --out data/datasets/benchmark_audio_raw_balanced_15k_v2.npz
python scripts/dataset/freeze_benchmark_test.py `
  --dataset data/datasets/benchmark_audio_raw_balanced_15k_v2.npz `
  --declare-untouched
```

O selo só pode ser criado antes de o novo teste orientar qualquer decisão de
arquitetura, hiperparâmetro, limiar ou correção.

Essa reconstrucao do tier `medium` continua deliberadamente **in-domain**:
ela serve para comparacao controlada com a composicao historica, mas falha ao
ativar `--academic-protocol` porque o oraculo de fonte excede 55%. Para uma
alegacao cross-domain, e necessario construir um conjunto em que cada fonte
contribua com ambas as classes, ou reservar fontes inteiras pareadas para o
teste. O pipeline recusa a execucao estrita enquanto essa condicao nao for
satisfeita.
