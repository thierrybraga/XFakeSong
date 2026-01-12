# Datasets Públicos de Deepfake de Áudio

Este guia reúne os principais datasets públicos para pesquisa em detecção de deepfakes de áudio. As informações estão em português, com links oficiais, licenças, tamanhos e instruções de obtenção. Use estes recursos para treinar, validar e comparar modelos de detecção.

## Boas Práticas
- Respeite as licenças e termos de uso de cada dataset.
- Não redistribua arquivos quando a licença proibir.
- Mantenha separação clara entre splits de treino, validação e teste.
- Documente pré-processamentos (resample, normalização, duração, canais).

## Lista de Datasets

### ASVspoof 2019 (LA/PA)
- Descrição: Base pública de referência para anti-spoofing com cenários de Logical Access (síntese/voice conversion) e Physical Access (replay).
- Download oficial: University of Edinburgh DataShare (DOI) [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) [fonte]
- Licença: Disponível via cadastro/aceite; consulte a página do DataShare.
- Tamanho/Formato: FLAC/PCM; múltiplos splits (train/dev/eval).
- Uso recomendado: Treino e validação de detectores; generalização para ataques conhecidos e desconhecidos.
- Referências:
  - Paper: [ASVspoof 2019](https://arxiv.org/abs/1911.01601) [fonte]
- Observação: Alguns espelhos em Hugging Face existem para conveniência, mas verifique compatibilidade e licença.

### ASVspoof 2021 (Speech Deepfake)
- Descrição: Sub-desafio focado em detecção de deepfakes de fala em condições desconhecidas.
- Site oficial: [ASVspoof 2021](https://www.asvspoof.org/index2021.html) [fonte]
- Plano de avaliação: [Evaluation Plan (PDF)](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf) [fonte]
- Licença/Obtenção: Requer registro e concordância com regras; avaliação libera novos dados (train/dev derivados de 2019).
- Uso recomendado: Benchmark de robustez e generalização em deepfake de fala.

### WaveFake (NeurIPS 2021)
- Descrição: Dataset de deepfakes de áudio com múltiplas arquiteturas (MelGAN, PWG, HiFi-GAN, WaveGlow etc.); 175 horas de áudio gerado.
- Repositório: [GitHub — WaveFake](https://github.com/RUB-SysSec/WaveFake) [fonte]
- Download: Zenodo [WaveFake 1.2.0](https://zenodo.org/records/5642694) (generated_audio.zip ~28.9 GB) [fonte]
- Licença: CC BY-SA 4.0 [fonte]
- Idiomas: Inglês e Japonês (LJSpeech/JSUT).
- Exemplo de download:

```bash
wget "https://zenodo.org/records/5642694/files/generated_audio.zip?download=1" -O wavefake_generated_audio.zip
unzip wavefake_generated_audio.zip -d datasets/wavefake
```

### FakeAVCeleb (Áudio-Video multimodal)
- Descrição: Dataset multimodal com deepfakes de vídeo e áudios clonados sincronizados (lip-sync). Útil para pesquisa conjunta áudio+vídeo.
- Repositório: [GitHub — FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) [fonte]
- Site: [Dataset Site / Download](https://sites.google.com/view/fakeavcelebdash-lab/) [fonte]
- Licença/Obtenção: Acesso via formulário de solicitação (Google Form) e script fornecido pelos autores.
- Uso recomendado: Estudos multimodais; avaliação de detectores em cenários realistas e vieses.

### In-the-Wild (Deepfake de Áudio)
- Descrição: Coleta de deepfakes e áudios autênticos de figuras públicas, visando generalização a cenários realistas.
- Página oficial: [In-the-Wild Dataset](https://deepfake-total.com/in_the_wild) [fonte]
- Licença: Apache 2.0 [fonte]
- Tamanho: ~20.8h (bonafide) e ~17.2h (spoof).
- Uso recomendado: Testes de generalização em condições reais fora de laboratório.

### SWAN-DF (Áudio-Video deepfakes)
- Descrição: Dataset público de alta fidelidade com faces e vozes clonadas; também inclui amostras de áudio-only (LibriTTS-DF).
- Site: [SWAN-DF](https://swan-df.github.io/) [fonte]
- Uso recomendado: Pesquisa conjunta de vulnerabilidades em reconhecimento de identidade e detecção AV.
- Observação: Verifique política de acesso e citação conforme instruções dos autores.

### ASVspoof 2019 (mirror em Hugging Face)
- Descrição: Espelho comunitário do ASVspoof 2019 (LA/PA) para carga via `datasets`.
- Link: [Hugging Face — asvspoof2019](https://huggingface.co/datasets/LanceaKing/asvspoof2019) [fonte]
- Licença (mirror): ODC BY 1.0 (segundo card do mirror) [fonte]
- Exemplo (Python):

```python
from datasets import load_dataset
la = load_dataset("LanceaKing/asvspoof2019", "LA")
pa = load_dataset("LanceaKing/asvspoof2019", "PA")
```

### Fake-or-Real (FoR) — Kaggle
- Descrição: Conjunto focado em detecção de deepfakes de voz; útil como fonte adicional de avaliação.
- Link: Kaggle — Fake-or-Real (FoR) [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) [fonte]
- Licença/Obtenção: Disponível via Kaggle; requer login e aceite dos termos.
- Observação: Verifique integridade e balanceamento antes de uso em produção.

## Organização Sugerida no Projeto
- Armazene datasets sob `datasets/` com subpastas por nome e split:
  - `datasets/<nome>/train`, `datasets/<nome>/dev`, `datasets/<nome>/test`
- Para datasets multimodais, crie pastas separadas:
  - `datasets/<nome>/audio/...`, `datasets/<nome>/video/...`
- Mantenha um `metadata.json` com:
  - fonte (URL), licença, data de obtenção, comandos de pré-processamento.

## Referências
- ASVspoof 2019 — DataShare: https://datashare.ed.ac.uk/handle/10283/3336 [fonte]
- ASVspoof 2019 — Paper: https://arxiv.org/abs/1911.01601 [fonte]
- ASVspoof 2021 — Site: https://www.asvspoof.org/index2021.html [fonte]
- ASVspoof 2021 — Evaluation Plan: https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf [fonte]
- WaveFake — GitHub: https://github.com/RUB-SysSec/WaveFake [fonte]
- WaveFake — Zenodo: https://zenodo.org/records/5642694 [fonte]
- FakeAVCeleb — GitHub: https://github.com/DASH-Lab/FakeAVCeleb [fonte]
- FakeAVCeleb — Site: https://sites.google.com/view/fakeavcelebdash-lab/ [fonte]
- In-the-Wild — Dataset: https://deepfake-total.com/in_the_wild [fonte]
- SWAN-DF — Site: https://swan-df.github.io/ [fonte]
- ASVspoof 2019 — Mirror: https://huggingface.co/datasets/LanceaKing/asvspoof2019 [fonte]

