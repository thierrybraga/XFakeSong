# utils

Utilitarios compartilhados.

## Responsabilidade

Agrupa helpers de audio, arquivos, sistema, VAD e suporte a Colab.

## Quando usar

Use para funcoes pequenas, reutilizaveis e sem regra de negocio. Se um helper
passar a coordenar fluxo de dominio, promova-o para `domain/services`.

## Arquivos

- `audio_utils.py`: leitura, conversao e validacoes de audio.
- `file_utils.py`: operacoes seguras de arquivo e extracao.
- `helpers.py`: helpers gerais de path, JSON, hash e retry.
- `system_utils.py`: diagnostico e manutencao local.
- `silero_vad.py`: integracao VAD.
- `colab.py`: suporte isolado para Google Colab.
