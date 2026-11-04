"""Serviços de domínio.

Sem imports eager: ``detection_service`` puxa TensorFlow e
``feature_extraction_service`` puxa librosa e todos os extratores. Importar
qualquer um deles aqui tornava essas bibliotecas obrigatórias para todo
consumidor de ``app.domain.services.*`` — inclusive o ambiente Docker
"classical-ml". Importe direto do módulo do serviço.
"""
