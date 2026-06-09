# Datasets Públicos de Deepfake de Áudio

Este guia reúne os principais datasets públicos para detecção de deepfakes de áudio e documenta **como baixá-los e balanceá-los** no XFakeSong, tanto pela interface Gradio quanto pela linha de comando.

## Boas Práticas
- Respeite as licenças e termos de uso de cada dataset.
- Não redistribua arquivos quando a licença proibir.
- Mantenha separação clara entre splits de treino, validação e teste.
- Documente pré-processamentos (resample, normalização, duração, canais).
- **Balanceie as classes** (`real` ≈ `fake`) — o ratio ideal fica entre 0.8 e 1.25.

---

## Download via XFakeSong

O XFakeSong oferece **dois caminhos** para popular `app/datasets/real/` e `app/datasets/fake/`:

### 1. Interface Gradio (recomendado) — Download balanceado

Abra a interface (`python main.py --gradio`) → aba **Dataset → Download**.

O sistema é **balance-aware**: você define um *alvo por classe* e seleciona as fontes; o XFakeSong calcula automaticamente quantas amostras baixar de cada fonte para chegar a um dataset equilibrado.

Fluxo:
1. **Barra de balanço** no topo mostra o estado atual (real vs fake, ratio, badge ✅/⚠️/❌).
2. Escolha um **preset** (ex.: "PT-BR Completo", "Internacional Padrão") ou marque fontes manualmente.
3. Ajuste o **alvo por classe** (slider, 100–10 000). O plano de download recalcula automaticamente.
4. A tabela **Plano de Download** mostra quantas amostras `real`/`fake` virão de cada fonte e o ratio estimado pós-download.
5. O painel **Prontidão para Treinamento** indica quais dos 14 modelos ficarão habilitados (Clássico ≥300, CNN Leve ≥1 000, CNN/RNN ≥2 000, Transformer ≥4 000, Ensemble ≥6 000 por classe).
6. Clique em **Iniciar Download Balanceado**. A barra e a prontidão atualizam ao vivo após cada fonte.

**Presets disponíveis:**

| Preset | Fontes | Uso |
|--------|--------|-----|
| PT-BR Rápido | BRSpeech-DF + Fake Voices | Começar rápido em português |
| PT-BR Completo | + CETUC + MLAAD-PT | Cobertura PT-BR completa |
| Internacional Padrão | ASVspoof 2019 + WaveFake + In-the-Wild | Benchmark anti-spoofing |
| Máxima Cobertura | PT-BR + Internacional | Máxima diversidade |
| Só Reforçar Real | FLEURS + CETUC + Common Voice PT | Corrigir déficit de reais |
| Só Reforçar Fake | Fake Voices + MLAAD-PT + WaveFake + ASVspoof 5 | Corrigir déficit de fakes |

> **Dica:** combine sempre ≥1 fonte `both` (BRSpeech-DF, ASVspoof 2019) com fontes especializadas (Fake Voices para fakes, CETUC para reais) para diversidade máxima.

### 2. Linha de comando — `scripts/download_datasets.py`

```bash
# Atalhos
python scripts/download_datasets.py --all --max-samples 2000        # PT-BR (brspeech+cetuc+fake-voices)
python scripts/download_datasets.py --all-intl --max-samples 2000   # Internacional (asvspoof2019+wavefake+in-the-wild)

# Fontes individuais
python scripts/download_datasets.py --brspeech --max-samples 1000
python scripts/download_datasets.py --cetuc --max-samples 1000
python scripts/download_datasets.py --fake-voices --max-speakers 20
python scripts/download_datasets.py --fleurs --max-samples 500
python scripts/download_datasets.py --mlaad-pt --max-samples 500
python scripts/download_datasets.py --common-voice-pt --max-samples 1000
python scripts/download_datasets.py --asvspoof2019 --max-samples 2000
python scripts/download_datasets.py --wavefake --max-samples 2000
python scripts/download_datasets.py --in-the-wild --max-samples 1000
python scripts/download_datasets.py --asvspoof5 --max-samples 2000
```

| Flag | Tipo | Prefixo arquivo | Licença |
|------|:----:|-----------------|---------|
| `--brspeech` | both | `brspeech_` | ODC-BY |
| `--fake-voices` | fake | `fkvoice_` | MIT |
| `--cetuc` | real | `cetuc_` | livre |
| `--fleurs` | real | `fleurs_` | CC BY 4.0 |
| `--mlaad-pt` | fake | `mlaad_` | CC-BY-NC 4.0 |
| `--common-voice-pt` | real | `cvpt_` | CC0 |
| `--asvspoof2019` | both | `asv2019_` | ODC-BY |
| `--wavefake` | fake | `wavefake_` | MIT |
| `--in-the-wild` | both | `itw_` | CC BY 4.0 |
| `--asvspoof5` | both | `asv5_` | CC BY 4.0 |

**Parâmetros:**
- `--max-samples N` — máximo **por classe** (fontes `both` dividem em N/2 real + N/2 fake).
- `--max-speakers N` — só para Fake Voices (≈80 amostras/falante).

**Garantias do pipeline de download** (`process_audio` + `safe_write_wav`):
- Reamostragem para 16 kHz mono.
- Sanitização de **NaN/Inf** (arquivos corrompidos são descartados, não salvos).
- Rejeição de áudios silenciosos (peak < 1e-6) e fora de duração (1–30 s).
- Rejeição de magnitudes absurdas (int16 não-normalizado).
- Normalização de pico para 0.95 (evita clipping).
- Indexação livre de colisão (`next_index` — não sobrescreve arquivos existentes ao re-rodar).

---

## Pós-download: validação e splits

Sempre rode o pré-processamento **antes de treinar**:

```bash
python scripts/preprocess_dataset.py --validate         # relatório de integridade
python scripts/preprocess_dataset.py --normalize        # reamostra/normaliza + remove corrompidos
python scripts/preprocess_dataset.py --remove-duplicates # MD5
python scripts/preprocess_dataset.py --create-splits    # train/val/test estratificado
python scripts/preprocess_dataset.py --full             # tudo acima
```

Pela UI: aba **Dataset → Preprocessamento → Pipeline Completo**.

> **`--validate` agora detecta NaN/Inf** explicitamente. Se aparecer a linha
> `NaN/Inf (graves): N — REMOVA antes de treinar!`, rode `--normalize` (que
> sanitiza ou remove esses arquivos) antes de iniciar o treino — caso contrário
> o treino falha com `loss: nan` na 1ª época.

---

## Solução de Problemas (Download)

Os scripts agora dão **mensagens acionáveis**. Os erros mais comuns:

### `ImportError: ... please install 'torchcodec'` (datasets 4.x)
A biblioteca `datasets` **4.x** passou a exigir o `torchcodec` (torch + FFmpeg)
para decodificar áudio e **quebra o download** — mesmo com `Audio(decode=False)`.
O projeto fixa `datasets>=2.14,<4.0` (onde o decode é via `soundfile`). Se você
estiver com a 4.x (ex.: ambiente novo, Colab):
```bash
pip install 'datasets>=2.14,<4.0'
```
Os notebooks de treino já fazem isso na célula de download.

### "requer AUTENTICAÇÃO no HuggingFace" (401)
```bash
# 1. Crie um token: https://huggingface.co/settings/tokens
# 2. Autentique:
huggingface-cli login
# 3. Tente novamente.
```
Afeta: Common Voice PT, ASVspoof 5 (e qualquer dataset privado).

### "é GATED — você precisa aceitar os termos" (403)
1. Acesse a página do dataset no HuggingFace (o link aparece no erro).
2. Clique em **"Agree and access repository"**.
3. `huggingface-cli login` e tente novamente.

Afeta principalmente: `mozilla-foundation/common_voice_*`.

### "falha de REDE"
- Verifique sua conexão.
- HuggingFace/Zenodo podem estar instáveis — aguarde e tente de novo.
- Atrás de proxy corporativo? Configure `HF_ENDPOINT` ou `HTTPS_PROXY`.

### "Dependências faltando"
```bash
pip install datasets>=4.0 huggingface_hub pandas requests tqdm
# ou simplesmente:
pip install -r requirements.txt
```

### "To support decoding audio data, please install 'torchcodec'"
A partir de `datasets` >= 4.0, o decode automático de áudio em streaming passou a
exigir o pacote `torchcodec` (que depende de torch + FFmpeg e é problemático no
Windows). **Você NÃO precisa instalar torchcodec** — o XFakeSong desativa o decode
automático (`Audio(decode=False)`) e decodifica os bytes com `soundfile`
internamente. Se ainda vir esse erro, atualize o repositório para a versão com a
correção (`cast_no_decode` / `extract_audio` em `scripts/download_datasets.py`).

### MLAAD demora muito / não encontra PT
MLAAD é multilíngue (~160 K amostras). O script tem um **cap de iterações**; se
houver poucos exemplos PT, ele avisa e para. Prefira WaveFake/Fake Voices para
fakes se precisar de volume rapidamente.

---

## Lista de Datasets (Referência)

### ASVspoof 2019 (LA/PA)
- Descrição: Base de referência para anti-spoofing — Logical Access (síntese/VC) e Physical Access (replay).
- Download oficial: [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)
- Mirror HF (usado pelo script): [LanceaKing/asvspoof2019](https://huggingface.co/datasets/LanceaKing/asvspoof2019) — ODC-BY 1.0
- Paper: [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)

### ASVspoof 2021 (Speech Deepfake)
- Descrição: Sub-desafio focado em deepfakes de fala em condições desconhecidas.
- Site: [asvspoof.org/index2021](https://www.asvspoof.org/index2021.html)
- Plano: [Evaluation Plan (PDF)](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)

### ASVspoof 5 (2024)
- Descrição: Edição mais recente; 20+ tipos de ataque, incluindo TTS/VC com LLMs de voz.
- Mirror HF: [jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5) — CC BY 4.0 (requer login)

### WaveFake (NeurIPS 2021)
- Descrição: Deepfakes de áudio com 6 vocoders (MelGAN, PWG, HiFi-GAN, WaveGlow, MB-MelGAN, FB-MelGAN); ~175 h.
- Repositório: [GitHub — WaveFake](https://github.com/RUB-SysSec/WaveFake)
- Download: [Zenodo 5642694](https://zenodo.org/records/5642694) (~28.9 GB) — MIT/CC BY-SA 4.0
- Idiomas: Inglês e Japonês (LJSpeech/JSUT).
- O script tenta espelhos HF primeiro e cai para o Zenodo automaticamente.

### In-the-Wild (Deepfake de Áudio)
- Descrição: Deepfakes e áudios autênticos de figuras públicas; foco em generalização realista.
- Página: [deepfake-total.com/in_the_wild](https://deepfake-total.com/in_the_wild) — CC BY 4.0
- Tamanho: ~20.8 h bonafide + ~17.2 h spoof.

### BRSpeech-DF (PT-BR)
- Descrição: Corpus PT-BR com bonafide + spoof (459 K arquivos). Principal fonte em português.
- Mirror HF: [AKCIT-Deepfake/BRSpeech-DF](https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF) — ODC-BY

### Fake Voices (XTTS, PT-BR)
- Descrição: ~140 h geradas por XTTS, 101 falantes; ZIPs por falante.
- Mirror HF: [unfake/fake_voices](https://huggingface.co/datasets/unfake/fake_voices) — MIT

### MLAAD v9 (subset PT)
- Descrição: Multi-Language Anti-spoofing; o script filtra o subconjunto PT (fake only).
- Mirror HF: [OU-CSAIL/MLAAD](https://huggingface.co/datasets/OU-CSAIL/MLAAD) — CC-BY-NC 4.0

### FLEURS / CETUC / Common Voice PT (reais PT-BR)
- FLEURS: [google/fleurs](https://huggingface.co/datasets/google/fleurs) (`pt_br`) — CC BY 4.0
- CETUC: via Common Voice/OpenSLR 132 (fallback em cascata) — livre
- Common Voice PT: [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) (`pt`) — CC0 (requer login/aceite)

### FakeAVCeleb (Áudio-Vídeo)
- Descrição: Multimodal (vídeo + áudio clonado sincronizado). Acesso via formulário dos autores.
- Site: [fakeavcelebdash-lab](https://sites.google.com/view/fakeavcelebdash-lab/)

### Fake-or-Real (FoR) — Kaggle
- Descrição: Detecção de deepfakes de voz; fonte adicional de avaliação.
- Link: [Kaggle — FoR](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) (requer login Kaggle)

---

## Organização no Projeto

O XFakeSong usa estrutura de dois estágios em `app/datasets/`:

```
app/datasets/
├── real/           # Áudios genuínos (16 kHz mono PCM-16)
├── fake/           # Áudios sintéticos/deepfake
├── raw/            # Caches de download (ZIPs, parquet) antes da normalização
├── splits/         # Gerado por preprocess_dataset.py --create-splits
│   ├── train/{real,fake}/
│   ├── val/{real,fake}/
│   ├── test/{real,fake}/
│   └── splits_metadata.json
└── features/       # Features extraídas (numpy/JSON)
```

**Fluxo recomendado de ponta a ponta:**
1. **Baixar** (balanceado): UI **Dataset → Download** ou `python scripts/download_datasets.py --all --max-samples 2000`
2. **Validar + normalizar + splits**: `python scripts/preprocess_dataset.py --full`
3. **Treinar**: UI **Treinar** (wizard) — escolha um modelo compatível com o tamanho do seu dataset.

Mantenha um `metadata.json` por dataset externo com: fonte (URL), licença, data de obtenção e comandos de pré-processamento.

---

## Referências
- ASVspoof 2019 — DataShare: https://datashare.ed.ac.uk/handle/10283/3336
- ASVspoof 2019 — Paper: https://arxiv.org/abs/1911.01601
- ASVspoof 2021 — Site: https://www.asvspoof.org/index2021.html
- ASVspoof 5 — HF: https://huggingface.co/datasets/jungjee/asvspoof5
- WaveFake — GitHub: https://github.com/RUB-SysSec/WaveFake
- WaveFake — Zenodo: https://zenodo.org/records/5642694
- In-the-Wild — Dataset: https://deepfake-total.com/in_the_wild
- BRSpeech-DF — HF: https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF
- Fake Voices — HF: https://huggingface.co/datasets/unfake/fake_voices
- MLAAD — HF: https://huggingface.co/datasets/OU-CSAIL/MLAAD
- FLEURS — HF: https://huggingface.co/datasets/google/fleurs
- Common Voice — HF: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
- FakeAVCeleb — Site: https://sites.google.com/view/fakeavcelebdash-lab/
- Fake-or-Real — Kaggle: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
