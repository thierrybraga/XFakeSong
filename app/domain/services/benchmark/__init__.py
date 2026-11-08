"""Benchmark do TCC visto pela aplicação: catálogo, resultados, artigo e execução.

Tudo aqui lê arquivos (sidecars, ``results.json``, ``run_summary.json``, o
artigo) ou lança o orquestrador ``benchmarks/run_models_sequential.py`` como
processo separado. Nada carrega pesos de modelo nem lê a partição de teste: os
leitores devolvem só métricas agregadas.
"""
