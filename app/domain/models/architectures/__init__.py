"""Módulo de arquiteturas de modelos.

Import eager restrito a submódulos livres de TensorFlow (svm, random_forest,
registry). Os módulos Keras/TF individuais (aasist, conformer, etc.) e os
helpers que dependem de `tensorflow.keras` (architecture_patcher,
safe_normalization, transforms) NUNCA devem ser importados aqui: qualquer
consumidor deste pacote — incluindo o ambiente Docker "classical-ml"
(SVM/RandomForest puros, sem TF instalado) — herdaria TensorFlow como
dependência obrigatória.

`create_model_by_name`/`ArchitectureRegistry.create_model` (registry.py) já
carregam esses módulos sob demanda via `__import__`/import local. Precisa de
`SafeInstanceNormalization`, `ArchitecturePatcher` etc.? Importe direto do
submódulo: `from app.domain.models.architectures.safe_normalization import ...`.
"""
from . import random_forest, registry, svm

__all__ = [
    "svm",
    "random_forest",
    "registry",
    # Registry functions
    "architecture_registry",
    "get_available_architectures",
    "create_model_by_name",
    "create_safe_model_by_name",
    "get_architecture_info",
    "validate_architecture_input",
]

# Importar registry functions para conveniência (registry.py não depende de TF).
from .registry import (  # noqa: E402
    architecture_registry,
    create_model_by_name,
    create_safe_model_by_name,
    get_architecture_info,
    get_available_architectures,
    validate_architecture_input,
)
