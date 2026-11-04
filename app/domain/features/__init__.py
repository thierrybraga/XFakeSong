"""Extração de características de áudio.

Os extratores são carregados sob demanda pelo
``app.domain.services.feature_extraction.extractor_loader`` (caminhos de
módulo explícitos). Sem re-exports aqui: ``import
app.domain.features.benchmark_frontend`` não deve carregar os doze
extratores.
"""
