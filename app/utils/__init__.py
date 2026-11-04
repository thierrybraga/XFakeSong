"""Utilitários centrais: áudio, arquivos, sistema, VAD e o helper de Colab.

Sem re-exports: importe do submódulo (``from app.utils.helpers import
ensure_directory``). Assim ``app.utils.file_storage`` continua importável sem
numpy, que ``helpers`` carrega no topo.
"""
