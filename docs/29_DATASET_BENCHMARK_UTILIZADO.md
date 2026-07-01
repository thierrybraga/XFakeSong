# Dataset utilizado no treino e benchmark

Data da revisão local: **28/06/2026**.

O benchmark canônico do XFakeSong usa o tier **`medium`**. O alvo consolidado é
**15.000 amostras** balanceadas: 7.500 reais e 7.500 falsas. Em 27/06/2026,
Common Voice/FLEURS no Hugging Face ficaram indisponíveis para completar os
reais PT-BR estritos; portanto, a consolidação final usa MLS Portuguese e
TTS-Portuguese Corpus como reforço real fora do HF.

Histórico da decisão:

- **PT-BR estrito validado:** 9.008 amostras, 4.504 reais + 4.504 falsas.
- **15k viável:** completa reais com MLS Portuguese/TTS-Portuguese, registrando
  explicitamente que MLS é português amplo/LibriVox, não PT-BR estrito.

O arquivo canônico gerado por esse tier é:

`app/datasets/benchmark_audio_raw_balanced_15k.npz`

Os WAVs ativos consolidados ficam em `app/datasets/real/` e
`app/datasets/fake/`. O `.npz` é derivado dos splits em
`app/datasets/splits/` e padroniza cada amostra em janela de 5,0 s.

O tier `small` fica reservado para execução rápida de 10k, e o tier `large`
fica reservado para execução estendida de 20k com protocolo de falantes não
vistos.

## Tiers Consolidados

| Tier | Real | Fake | Total | Fontes | Split | Uso |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `test` | 100 | 100 | 200 | BRSpeech-DF + Fake Voices | 70/15/15 estratificado | smoke |
| `small` | 5.000 | 5.000 | 10.000 | BRSpeech-DF + Fake Voices | 70/15/15 estratificado | iteração robusta |
| `medium` | 7.500 | 7.500 | 15.000 | BRSpeech-DF + MLS Portuguese + TTS-Portuguese + Fake Voices | 70/15/15 estratificado | **benchmark canônico 15k viável** |
| `large` | 10.000 | 10.000 | 20.000 | BRSpeech-DF + MLS Portuguese + TTS-Portuguese + Fake Voices | disjunto por falante quando possível | auditoria estendida |

## Contrato do NPZ Canônico

| Item | Valor |
| --- | --- |
| Tier | `medium` |
| Arquivo | `app/datasets/benchmark_audio_raw_balanced_15k.npz` |
| Amostras alvo | 15.000 |
| Classes | 7.500 real + 7.500 fake |
| Split alvo | 10.500 treino + 2.250 validação + 2.250 teste |
| Taxa de amostragem | 16 kHz |
| Janela exportada | 5,0 s |
| Entrada raw-audio | `(80000, 1)` por amostra |
| Tamanho do `.npz` | 2.769,01 MiB (`2.903.517.797` bytes) |
| Duração efetiva no `.npz` | 1.250,00 min / 20,83 h |
| WAVs ativos | 15.000 arquivos; 3.746,26 MiB; 2.045,61 min / 34,09 h |
| Formato dos WAVs ativos | WAV PCM linear, 16 bits, mono, 16 kHz, sem compressão |
| Modulação/codificação | PCM (`Pulse-Code Modulation`) linear em arquivo RIFF/WAV |
| Arrays | `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `groups`, `speaker_ids`, `metadata_json` |
| Diretório de splits | `app/datasets/splits/` |
| Tabela de falantes | `app/datasets/speaker_table.csv` |
| Manifesto de falantes | `app/datasets/speaker_manifest.json` |

Com janela padronizada de 5 s, o tier medium representa aproximadamente
**1.250 min** ou **20,83 h** de áudio exportado no `.npz`.

Os WAVs ativos preservam a duração validada após VAD, normalização e descarte de
arquivos inválidos; por isso a duração bruta ativa (**2.045,61 min**) é maior
que a duração efetiva do `.npz`, que usa exatamente 5 s por amostra.

## Composição Consolidada do Dataset Ativo

| Classe | Fonte | Arquivos | MiB | Minutos | Horas | Duração média | Falantes/chaves | Status de ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| real | BRSpeech-DF bonafide | 3.750 | 847,84 | 462,95 | 7,72 | 7,41 s | 1 fallback | fallback por fonte |
| real | MLS Portuguese | 1.875 | 894,37 | 488,40 | 8,14 | 15,63 s | 21 | ID real derivado do leitor/caminho |
| real | TTS-Portuguese Corpus | 1.875 | 645,12 | 352,28 | 5,87 | 11,27 s | 1 | `ttsport_single_speaker` |
| fake | BRSpeech-DF spoof | 3.750 | 836,65 | 456,84 | 7,61 | 7,31 s | 1 fallback | fallback por fonte |
| fake | Fake Voices XTTS | 3.750 | 522,27 | 285,15 | 4,75 | 4,56 s | 50 | ID real do falante/ZIP |
| **total** | **ativo** | **15.000** | **3.746,26** | **2.045,61** | **34,09** | — | **73 chaves** | 7.500 IDs reais + 7.500 fallback |

Resumo por classe:

| Classe | Arquivos | MiB | Minutos | Horas |
| --- | ---: | ---: | ---: | ---: |
| real | 7.500 | 2.387,33 | 1.303,63 | 21,73 |
| fake | 7.500 | 1.358,92 | 741,98 | 12,37 |

Resumo dos splits materializados:

| Split | Real | Fake | Total |
| --- | ---: | ---: | ---: |
| treino | 5.250 | 5.250 | 10.500 |
| validação | 1.125 | 1.125 | 2.250 |
| teste | 1.125 | 1.125 | 2.250 |

## Fontes e Metadados

| Fonte | Classe no benchmark | Papel | ID de falante |
| --- | --- | --- | --- |
| BRSpeech-DF | real + fake | fonte principal PT-BR com bonafide/spoof | Parquets locais não expõem coluna explícita de falante; usa fallback por fonte |
| MLS Portuguese | real | reforço real fora do HF para fechar 15k | leitor LibriVox derivado do caminho local quando disponível |
| TTS-Portuguese Corpus | real | reforço real PT-BR, limitado por 1 falante | `ttsport_single_speaker` |
| Common Voice PT | real legado | indisponível/vazio no HF; usar somente se já existir localmente | `client_id` quando preservado pelo downloader |
| FLEURS PT-BR | real legado | trava no streaming HF local; usar somente se já existir localmente | campo de falante quando existir no registro da fonte |
| Fake Voices XTTS | fake | gerador sintético independente | nome do ZIP/falante em `unfake/fake_voices` |

O downloader registra IDs reais em `speaker_manifest.json` sempre que a fonte
expõe esse identificador. Quando uma fonte não expõe falante por arquivo, o
pipeline não inventa IDs: `speaker_ids` usa fallback por fonte (`brspeech`,
`cvpt`, `fleurs`, etc.) e a tabela marca `id_status=fallback_source`.

## Tabela Consolidada de Falantes

A tabela deve ser gerada após o download/splits:

```powershell
python scripts/rebuild_speaker_manifest.py --dataset-dir app/datasets
python scripts/export_speaker_table.py --dataset-dir app/datasets --scope all
```

Campos principais em `app/datasets/speaker_table.csv`:

| Campo | Descrição |
| --- | --- |
| `file` | nome do WAV |
| `relative_path` | caminho no projeto |
| `split` | `train`, `val`, `test`, `active` ou `overflow` |
| `class` | `real` ou `fake` |
| `source` | prefixo/fonte (`brspeech`, `mlspt`, `ttsport`, `fkvoice`) |
| `speaker_id` | ID real quando disponível |
| `speaker_key` | `fonte:speaker_id` ou fallback `fonte` |
| `id_status` | `real_id` ou `fallback_source` |
| `duration_sec` | duração do WAV |
| `sample_rate` | taxa de amostragem detectada |
| `channels` | canais detectados |
| `size_bytes` | tamanho do arquivo |

## Comandos Recomendados

Reconstrução canônica do dataset medium 15k:

```powershell
$env:DOCKER_TRAIN_CPU_LIMIT='8'
docker compose -f docker\compose\benchmark.nvidia.yml --env-file .env run --rm benchmark `
  python scripts/run_tcc_pipeline.py `
    --download `
    --tier medium `
    --full-benchmark `
    --epochs 100 `
    --batch-size 32 `
    --device-profile gpu `
    --npz app/datasets/benchmark_audio_raw_balanced_15k.npz `
    --out results/benchmark_15k_medium
```

Auditoria de falantes:

```powershell
docker compose -f docker\compose\benchmark.nvidia.yml --env-file .env run --rm benchmark `
  python scripts/audit_speaker_manifest.py --dataset-dir app/datasets --scope splits `
    --json-out app/datasets/speaker_audit.json
```

Benchmark sequencial sobre o NPZ canônico:

```powershell
docker compose -f docker\compose\benchmark.nvidia.yml --env-file .env run --rm benchmark `
  python scripts/run_models_sequential.py `
    --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz `
    --out results/benchmark_15k_medium `
    --epochs 100 `
    --batch-size 32 `
    --device-profile gpu `
    --timeout-min 240 `
    --latency-runs 30 `
    --snr 30 20 10 `
    --resume
```

## Limitações e Regras de Interpretação

- `medium` é o benchmark canônico de 15k, mas não promete split disjunto por
  falante; ele é estratificado.
- `large` é o tier correto quando a pergunta experimental exige usuários não
  vistos ou auditoria mais forte de vazamento.
- IDs de falante são usados somente quando vêm da fonte ou de metadado local
  rastreável.
- Amostras sem ID real continuam válidas no benchmark, mas devem ser reportadas
  como fallback por fonte na tabela de falantes.
