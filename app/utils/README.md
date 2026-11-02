# utils

Utilitarios compartilhados.

## Responsabilidade

Agrupa helpers de audio, arquivos, VAD e suporte a Colab. O pacote nao
re-exporta nomes: importe do submodulo.

## Quando usar

Use para funcoes pequenas, reutilizaveis e sem regra de negocio. Se um helper
passar a coordenar fluxo de dominio, promova-o para `domain/services`.

## Arquivos

- `audio_utils.py`: leitura, conversao e validacoes de audio.
- `file_utils.py`: fonte unica de operacoes de arquivo (diretorios, hash, nomes
  seguros `safe_filename`/`sanitize_filename`, extracao protegida).
- `helpers.py`: helpers gerais de JSON, formatacao, timing e retry.
- `file_storage.py`: escrita atômica, hardlinks e cópia segura de artefatos (sem numpy).
- `silero_vad.py`: integracao VAD.
- `colab.py`: suporte isolado para Google Colab.
