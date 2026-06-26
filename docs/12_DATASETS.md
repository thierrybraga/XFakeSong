# Datasets Públicos de Deepfake de Áudio

Este guia reúne os principais datasets públicos para detecção de deepfakes de áudio e documenta **como baixá-los e balanceá-los** no XFakeSong, tanto pela interface Gradio quanto pela linha de comando.

---

## Tiers de Dataset (test · small · medium · large)

O XFakeSong organiza a montagem do dataset em **quatro tiers** com finalidade
bem definida. Eles são a **fonte única de verdade** de tamanho/finalidade
(`app/core/dataset_catalog.py` → `DATASET_TIERS`), compartilhada por
`scripts/build_dataset.py`, pela interface Gradio, pelo benchmark e por esta
documentação — escolher um tier pré-configura tamanho, fontes e estratégia de
split de forma consistente em todo o sistema.

| Tier | Por classe | Total | Finalidade | Fontes | Split | Falante |
|------|-----------:|------:|------------|--------|-------|:------:|
| **test** | 100 | 200 | **Smoke** — validar que treino e modelo funcionam de ponta a ponta. Não mede desempenho. | BRSpeech-DF + Fake Voices | 70/15/15 estratificado | — |
| **small** | 1.000 | 2.000 | **Treino rápido** para iteração. Habilita Clássico (SVM/RF) + CNN leve. | BRSpeech-DF + Fake Voices | 70/15/15 estratificado | — |
| **medium** | 3.000 | 6.000 | **Treino + teste completos** com diversidade real adicional. Habilita até Transformer. | BRSpeech-DF + Common Voice PT + FLEURS + Fake Voices | 70/15/15 estratificado | — |
| **large** | 10.000 | 20.000 | **Completo** com falantes identificados e **usuários não vistos**. Habilita todas as 14 arquiteturas (Ensemble). | BRSpeech-DF + Common Voice PT + FLEURS + Fake Voices | **disjunto por falante** + cross-generator | **sim** |

### Descrição detalhada

- **`test` — mínimo (100/classe).** Conjunto de validação. Serve só para
  confirmar que download → pré-processamento → treino → inferência rodam sem
  erro (acurácia não é significativa neste tamanho). Real vem inteiramente do
  BRSpeech-DF bonafide; fake do BRSpeech-DF spoof + Fake Voices XTTS. Download em
  segundos. Use antes de qualquer execução pesada.
- **`small` — rápido (1.000/classe).** Para ciclos rápidos de experimentação.
  Atinge os limiares de prontidão de **SVM/RandomForest** e **CNNs leves**
  (RawNet2, Sonic Sleuth, MultiscaleCNN). PT-BR via BRSpeech-DF + Fake Voices.
- **`medium` — completo (3.000/classe).** Treino **e** teste mais robustos, com
  diversidade real adicional (Common Voice PT + FLEURS) e fake independente
  (Fake Voices XTTS). Atinge os limiares de **CNN/RNN** e **Transformers**.
- **`large` — completo + falantes não vistos (10.000/classe).** O tier de
  referência do TCC. Além do volume que habilita **todas as 14 arquiteturas**
  (incluindo o Ensemble, ≥6.000/classe), ele:
  - **identifica falantes** num sidecar `app/datasets/speaker_manifest.json`
    (Fake Voices por falante do ZIP, Common Voice por `client_id`, In-the-Wild
    por celebridade, ASVspoof pelo `speaker` do protocolo). Fontes que não
    expõem falante caem para o nível de **fonte** (`<prefixo>`);
  - cria splits **disjuntos por falante** — nenhum falante aparece em treino e
    teste ao mesmo tempo, medindo generalização a **usuários não vistos**;
  - combina com o **protocolo cross-generator** (segura o XTTS=`fkvoice` fora do
    treino).

> **Identificação de falante é aditiva** — não altera os nomes de arquivo
> (`<fonte>_NNNNN.wav`), então todos os parsers de prefixo (catálogo, auditoria,
> benchmark) continuam funcionando. O manifesto é gravado automaticamente pelos
> downloaders das fontes que expõem falante.

### Como usar

**Interface Gradio** — aba **Datasets → Download → "1. Escolher Tier"**. Escolher
um tier pré-preenche o **alvo por classe** e as **fontes**, e mostra a descrição
+ o protocolo de split. O restante do fluxo balance-aware continua igual.

**Linha de comando:**

```bash
# Monta o dataset de um tier (download + balanceamento + splits + config)
python scripts/build_dataset.py --tier test       # smoke, ~segundos
python scripts/build_dataset.py --tier small       # 1.000/classe
python scripts/build_dataset.py --tier medium      # 3.000/classe
python scripts/build_dataset.py --tier large       # 10.000/classe, split por falante

# Override manual do tamanho de um tier (mantém fontes/split do tier):
python scripts/build_dataset.py --tier medium --target 4000
```

O `dataset_config.json` resultante registra `tier`, `split_strategy`,
`speaker_aware` e um resumo `speakers` (total, identificados, por fonte). Para o
tier `large`, o `splits_metadata.json` registra `unseen_in_test` (falantes só no
teste) e a verificação de vazamento `leakage_overlap_train_test`.

**Benchmark (protocolo de usuário não visto):** o `.npz` exportado por
`scripts/run_tcc_pipeline.py` inclui o array `speaker_ids`. Ative o protocolo com:

```bash
# split disjunto por falante (usuários não vistos)
python scripts/run_benchmark.py --full --dataset SEU_large.npz --speaker-split

# holdout de um falante específico (teste = só ele + reais reservados)
python scripts/run_benchmark.py --full --dataset SEU_large.npz --unseen-speaker "fkvoice:<id>"

# pipeline ponta a ponta no tier large
python scripts/run_tcc_pipeline.py --download --tier large --full-benchmark --speaker-split
```

> Quando os falantes de uma fonte são correlacionados à classe (ex.: fonte pura
> só-real ou só-fake), o split disjunto por falante na criação dos splits pode
> degenerar para classe única; nesse caso o pipeline **cai para o estratificado**
> e avisa — o protocolo de usuário não visto fica melhor servido pelo benchmark
> (`--unseen-speaker`, que reserva reais para manter o teste balanceado).

---

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

Abra a interface (`python main.py --gradio`) → **Gerenciar → Datasets/Download**.

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
| Benchmark Robusto Recomendado | BRSpeech-DF + Fake Voices + FLEURS + Common Voice PT + ASVspoof 2019 + WaveFake + In-the-Wild | Treino/benchmark com PT-BR, fontes reais adicionais e validação internacional |
| Só Reforçar Real | FLEURS + CETUC + Common Voice PT | Corrigir déficit de reais |
| Só Reforçar Fake | Fake Voices + MLAAD-PT + WaveFake + ASVspoof 5 | Corrigir déficit de fakes |

> **Dica:** combine sempre ≥1 fonte `both` (BRSpeech-DF, ASVspoof 2019) com fontes especializadas (Fake Voices para fakes, CETUC para reais) para diversidade máxima.

### Catálogo usado pela Gradio e pelo benchmark

A aba Gradio **Datasets/Download** e o pipeline de benchmark usam o catálogo
central `app/core/dataset_catalog.py`. Esse catálogo documenta tipo de classe,
comando de download, licença, duração, falantes e uso recomendado. Quando uma
fonte não publica um valor estável por idioma, o campo fica marcado como
“não informado” ou “variável”; nesses casos, o manifesto do benchmark registra
as contagens realmente baixadas.

| Dataset | Tipo | Flag | Prefixo | Idioma | Arquivos/duração | Falantes | Licença | Uso no benchmark |
|---|:---:|---|---|---|---|---|---|---|
| BRSpeech-DF | both | `--brspeech` | `brspeech_` | pt-BR | 459.137 amostras; duração não informada pela fonte | não informado | Apache-2.0/CC BY 4.0 (HF divergente) | Fonte principal PT-BR para treino balanceado |
| Fake Voices | fake | `--fake-voices` | `fkvoice_` | pt-BR | ~140 h; ~30,5 GB | 101 falantes | MIT | Fake PT-BR independente para teste cross-generator |
| FLEURS | real | `--fleurs` | `fleurs_` | pt-BR | `pt_br` ~4,1 mil linhas; duração variável | não consolidado no catálogo local | CC BY 4.0 | Reforço de fala real PT-BR |
| CETUC | real | `--cetuc` | `cetuc_` | pt-BR | variável conforme fallback | variável | livre/variável | Completar déficit de amostras reais |
| MLAAD-PT | fake | `--mlaad-pt` | `mlaad_` | pt | subconjunto PT filtrado em streaming | não informado | CC-BY-NC 4.0 | Reforço fake; uso condicionado à licença NC |
| Common Voice PT | real | `--common-voice-pt` | `cvpt_`, `cv_` | pt | variável por release/configuração; HF v17 legacy vazio | variável por release/configuração | CC0 | Diversidade de fala real; usar Mozilla Data Collective |
| ASVspoof 2019 | both | `--asvspoof2019` | `asv2019_` | inglês | protocolo oficial LA/PA | derivado do VCTK; consultar protocolo oficial | ODC-BY 1.0 | Referência externa padrão anti-spoofing |
| WaveFake | fake | `--wavefake` | `wavefake_` | inglês/japonês | ~175 h; download completo ~28,9 GB | LJSpeech/JSUT; poucos falantes base | CC-BY-SA 4.0 | Reforço fake internacional e cross-vocoder |
| In-the-Wild | both | `--in-the-wild` | `itw_` | majoritariamente inglês | ~20,8 h real + ~17,2 h fake | 58 celebridades/falantes | Apache-2.0 | Validação externa realista |
| ASVspoof 5 | both | `--asvspoof5` | `asv5_` | inglês/multifonte | protocolo ASVspoof 5; mirror HF ~142 GB | ~2.000 falantes | consultar README/LICENSE oficiais | Referência moderna; pode exigir login/aceite |

O `.npz` exportado por `scripts/run_tcc_pipeline.py` inclui:

- `metadata_json.source_summary`: contagem por fonte, classe e horas estimadas
  pelas janelas exportadas;
- `metadata_json.dataset_catalog`: snapshot do catálogo usado na execução;
- `groups`: vetor alinhado às amostras, derivado do prefixo de origem, usado
  por `--group-split` e `--cross-generator`.

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
| `--brspeech` | both | `brspeech_` | Apache-2.0/CC BY 4.0 (HF divergente) |
| `--fake-voices` | fake | `fkvoice_` | MIT |
| `--cetuc` | real | `cetuc_` | livre |
| `--fleurs` | real | `fleurs_` | CC BY 4.0 |
| `--mlaad-pt` | fake | `mlaad_` | CC-BY-NC 4.0 |
| `--common-voice-pt` | real | `cvpt_` | CC0 |
| `--asvspoof2019` | both | `asv2019_` | ODC-BY |
| `--wavefake` | fake | `wavefake_` | MIT |
| `--in-the-wild` | both | `itw_` | Apache-2.0 |
| `--asvspoof5` | both | `asv5_` | consultar README/LICENSE oficiais |

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
- Mirror HF: [jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5) — verificar README/LICENSE oficiais (pode exigir login)

### WaveFake (NeurIPS 2021)
- Descrição: Deepfakes de áudio com 6 vocoders (MelGAN, PWG, HiFi-GAN, WaveGlow, MB-MelGAN, FB-MelGAN); ~175 h.
- Repositório: [GitHub — WaveFake](https://github.com/RUB-SysSec/WaveFake)
- Download: [Zenodo 5642694](https://zenodo.org/records/5642694) (~28.9 GB) — CC-BY-SA 4.0
- Idiomas: Inglês e Japonês (LJSpeech/JSUT).
- O script tenta espelhos HF primeiro e cai para o Zenodo automaticamente.

### In-the-Wild (Deepfake de Áudio)
- Descrição: Deepfakes e áudios autênticos de figuras públicas; foco em generalização realista.
- Página: [deepfake-total.com/in_the_wild](https://deepfake-total.com/in_the_wild) — Apache-2.0
- Tamanho: ~20.8 h bonafide + ~17.2 h spoof.

### BRSpeech-DF (PT-BR)
- Descrição: Corpus PT-BR com bonafide + spoof (459 K arquivos). Principal fonte em português.
- Mirror HF: [AKCIT-Deepfake/BRSpeech-DF](https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF) — metadado HF Apache-2.0; card cita CC BY 4.0

### Fake Voices (XTTS, PT-BR)
- Descrição: ~140 h geradas por XTTS, 101 falantes; ZIPs por falante.
- Mirror HF: [unfake/fake_voices](https://huggingface.co/datasets/unfake/fake_voices) — MIT

### MLAAD v9 (subset PT)
- Descrição: Multi-Language Anti-spoofing; o script filtra o subconjunto PT (fake only).
- Mirror HF: [OU-CSAIL/MLAAD](https://huggingface.co/datasets/OU-CSAIL/MLAAD) — CC-BY-NC 4.0

### FLEURS / CETUC / Common Voice PT (reais PT-BR)
- FLEURS: [google/fleurs](https://huggingface.co/datasets/google/fleurs) (`pt_br`) — CC BY 4.0
- CETUC: via Common Voice/OpenSLR 132 (fallback em cascata) — livre
- Common Voice PT: [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets) (`pt`) — CC0; o mirror HF v17 está legacy/vazio desde 2025

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
├── real/                   # Áudios genuínos (16 kHz mono PCM-16)
├── fake/                   # Áudios sintéticos/deepfake
├── raw/                    # Caches de download (ZIPs, parquet) antes da normalização
├── splits/                 # Gerado por preprocess_dataset.py --create-splits
│   ├── train/{real,fake}/
│   ├── val/{real,fake}/
│   ├── test/{real,fake}/
│   └── splits_metadata.json   # inclui split_strategy + estatística de falantes
├── dataset_config.json     # tier, split_strategy, speaker_aware, fontes, formato
├── speaker_manifest.json   # arquivo → falante (tier large; aditivo, opcional)
└── features/               # Features extraídas (numpy/JSON)
```

**Fluxo recomendado de ponta a ponta:**
1. **Escolher um tier e montar**: UI **Datasets → Download → Escolher Tier** ou
   `python scripts/build_dataset.py --tier <test|small|medium|large>` (faz
   download balanceado + validação/normalização + splits + `dataset_config.json`).
   Para fontes avulsas: `python scripts/download_datasets.py --all --max-samples 2000`.
2. **(Opcional) re-criar splits**: `python scripts/preprocess_dataset.py --full`
   (adicione `--speaker-disjoint` para o protocolo de usuários não vistos).
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
