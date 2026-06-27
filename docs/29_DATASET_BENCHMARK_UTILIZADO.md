# Dataset utilizado no treino e benchmark

Data da consolidação local: **27/06/2026**.

Este documento registra o dataset efetivamente preparado em
`app/datasets/splits/` para o treino e benchmark dos modelos do XFakeSong. Ele
deve ser lido como manifesto operacional do experimento local: os diretórios
brutos (`app/datasets/real/` e `app/datasets/fake/`) podem conter excedentes,
mas o conjunto de treino/validação/teste é definido pelos splits.

## Resumo executivo

| Item | Valor |
| --- | ---: |
| Diretório ativo | `app/datasets/splits/` |
| Total de amostras ativas | **14.508 WAV** (7.254 real + 7.254 fake) |
| Treino | 10.155 WAV (5.077 real + 5.078 fake) |
| Validação | 2.176 WAV (1.088 + 1.088) |
| Teste | 2.177 WAV (1.089 + 1.088) |
| `.npz` canônico | `app/datasets/benchmark_audio_raw_balanced_15k.npz` (~2,7 GB) |
| Janela no `.npz` | 5,0 s → `(80000, 1)` por amostra |
| Taxa de amostragem | 16 kHz |
| Canais | mono |
| Codificação | PCM 16-bit |
| Split | 70/15/15 estratificado, `random_state=42` |
| Separação por falante | não garantida no split atual |
| Arquivo de configuração | `app/datasets/dataset_config.json` |
| Pacote para builds | `dataset_pt_deepfake_15k.zip` (raiz do projeto) |

> **Nota de tamanho:** alvo nominal 15k (7.500/classe); o conjunto efetivo ficou
> em **14.508 (7.254/classe)** porque as fontes PT-BR de *fake* atingiram o teto
> de únicos — BRSpeech-spoof satura em ~5.007 no streaming e o Fake Voices XTTS
> contribuiu 2.247 (alguns falantes pendentes por instabilidade de rede). Para
> 7.500/classe: retome o Fake Voices (mais falantes) ou some WaveFake/MLAAD.

## Composição efetiva por fonte (conjunto ativo)

| Classe | Fonte/prefixo | Amostras |
| --- | --- | ---: |
| real | BRSpeech-DF bonafide (`brspeech_`) | 4.754 |
| real | Common Voice PT (`cvpt_`) | 2.500 |
| **real total** | | **7.254** |
| fake | BRSpeech-DF spoof (`brspeech_`) | 5.007 |
| fake | Fake Voices XTTS (`fkvoice_`) | 2.247 |
| **fake total** | | **7.254** |

Os splits 70/15/15 são estratificados por classe (`random_state=42`); cada split
preserva ~1:1 real/fake e a proporção das fontes acima. O `.npz` traz `groups`
(fonte/gerador por amostra) e `speaker_ids` (cai p/ fonte sem manifesto).

## Diretórios brutos e excedentes

Os diretórios brutos refletem o material baixado antes do balanceamento e dos
splits. No estado local atual:

| Diretório | Amostras WAV | Tamanho |
| --- | ---: | ---: |
| `app/datasets/real/` | 15.000 | 2.134,6 MB |
| `app/datasets/fake/` | 20.759 | 1.822,4 MB |
| `app/datasets/overflow/` | 759 | 115,7 MB |
| `app/datasets/raw/` | cache bruto/downloads | 20.470,0 MB |

O diretório `overflow/` contém amostras arquivadas pelo processo de
balanceamento/seleção ativa. Ele não entra no treinamento enquanto os scripts
usarem `app/datasets/splits/`.

## Fontes utilizadas

### BRSpeech-DF

- **Uso no experimento:** fonte principal, com amostras reais e sintéticas
  identificadas pelo prefixo `brspeech_`.
- **Repositório:** `AKCIT-Deepfake/BRSpeech-DF`
- **URL:** <https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF>
- **Idioma informado:** pt-BR.
- **Licença registrada no catálogo local:** Apache-2.0/CC BY 4.0 (metadados
  divergentes no Hugging Face).
- **Artigo/paper:** não há, no manifesto local, um artigo acadêmico canônico
  associado ao dataset card utilizado. A referência reprodutível adotada é o
  dataset card do Hugging Face.
- **Amostras no conjunto ativo:** 10.000.
- **Duração no conjunto ativo:** 1.232,8 min (20,55 h).
- **Observação:** como a mesma fonte fornece amostras reais e falsas, existe
  risco de aprendizado de artefatos de origem se a separação por falante/fonte
  não for aplicada em experimentos futuros.

### Common Voice PT

- **Uso no experimento:** reforço de fala real, prefixo `cvpt_`.
- **Projeto:** Mozilla Common Voice.
- **URL:** <https://commonvoice.mozilla.org/datasets>
- **Licença:** CC0.
- **Artigo/paper:** *Common Voice: A Massively-Multilingual Speech Corpus*
  (Ardila et al., LREC 2020).
- **Amostras no conjunto ativo:** 2.500.
- **Duração no conjunto ativo:** 545,0 min (9,08 h).
- **Observação:** o catálogo local registra que mirrors antigos no Hugging Face
  podem estar legados/vazios; a fonte canônica é a Mozilla Data Collective.

### Fake Voices XTTS

- **Uso no experimento:** fala sintética independente, prefixo `fkvoice_`.
- **Repositório:** `unfake/fake_voices`
- **URL:** <https://huggingface.co/datasets/unfake/fake_voices>
- **Licença:** MIT.
- **Gerador:** XTTS v2 (Coqui TTS), conforme manifesto local.
- **Artigo/paper:** não há, no manifesto local, artigo canônico associado ao
  dataset card. A referência reprodutível adotada é o dataset card do Hugging
  Face.
- **Amostras no conjunto ativo:** 5.000.
- **Duração no conjunto ativo:** 382,8 min (6,38 h).
- **Falantes informados pela fonte/catalogação local:** 101 falantes no conjunto
  completo; o split atual não consolidou IDs reais de falantes por arquivo.

### FLEURS

FLEURS aparece no catálogo e nos scripts como fonte real prevista para tiers
`medium/large`, porém **não aparece no conjunto ativo atual** pelos prefixos dos
WAVs em `app/datasets/splits/`. O documento do experimento, portanto, não conta
FLEURS na composição efetivamente usada.

- **Repositório:** `google/fleurs`
- **URL:** <https://huggingface.co/datasets/google/fleurs>
- **Licença:** CC BY 4.0.
- **Artigo/paper:** *FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech*.

## Falantes

O arquivo `dataset_config.json` atual registra apenas falantes inferidos por
fonte (`brspeech`, `cvpt`, `fkvoice`) e não IDs reais por amostra. Portanto:

- o número de falantes distintos **não está consolidado** para o conjunto ativo;
- o split atual deve ser descrito como **estratificado por classe**, não como
  disjunto por falante;
- métricas finais devem mencionar essa limitação, pois amostras correlacionadas
  por fonte ou falante podem atravessar treino, validação e teste.

## Uso no treinamento e benchmark

Para treinar/avaliar os 14 modelos, o fluxo recomendado é exportar os splits
para `.npz` e executar o benchmark sequencial com retomada:

```powershell
docker compose -f docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
  python scripts/export_npz_from_splits.py `
    --out app/datasets/benchmark_audio_raw_balanced_15k.npz `
    --sample-rate 16000 `
    --duration-sec 5.0

docker compose -f docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
  python scripts/run_models_sequential.py `
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz `
    --out results/large_benchmark_full `
    --epochs 100 `
    --batch-size 32 `
    --device-profile gpu `
    --timeout-min 240 `
    --latency-runs 30 `
    --snr 30 20 10 `
    --resume
```

## Limitações documentadas

- O conjunto ativo atual tem 14.508 amostras (7.254/classe), não 15.000.
- O split atual não é disjunto por falante.
- O número real de falantes por arquivo ainda não foi consolidado.
- A composição efetiva não inclui arquivos `fleurs_` no diretório de splits,
  apesar de FLEURS constar como fonte prevista no catálogo.
- As licenças devem ser preservadas e revisadas antes de redistribuir o
  dataset consolidado fora do ambiente local.
