"""Modelos de ML do domínio: ``architectures/``, ``training/`` e ``inference/``.

As entidades persistidas (ORM) ficam em ``app.core.db.models``. Sem
re-exports aqui: importar ``app.domain.models.architectures.svm`` não deve
carregar SQLAlchemy nem abrir o banco.
"""
