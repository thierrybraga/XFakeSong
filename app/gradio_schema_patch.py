"""Patch para compatibilizar Pydantic v2 + Gradio v4 schema generation.

Gradio 4.x + Pydantic 2.5+ podem gerar schemas com valores booleanos em
posições onde dicts são esperados. Este módulo intercepta e corrige
antes de Gradio processar.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def sanitize_schema(schema: Any) -> Any:
    """Recursivamente sanitiza schemas removendo bools que deveriam ser dicts.
    
    Problema: Em certos casos, Pydantic v2 gera:
        {"metadata": True}  # ← ERRADO, deveria ser {"metadata": {"type": "object"}}
    
    Este sanitizador detecta e corrige padrões conhecidos.
    """
    if not isinstance(schema, dict):
        return schema
    
    sanitized = {}
    for key, value in schema.items():
        # Se o valor é True/False em certos campos conhecidos,
        # substitui por schema vazio de dict
        if isinstance(value, bool) and key in {
            "metadata", "config", "details", "per_model", 
            "per_fold", "parameters", "training_metrics",
            "additionalProperties"
        }:
            if value is True:
                sanitized[key] = {"type": "object"}
            elif value is False:
                sanitized[key] = {}
        elif isinstance(value, dict):
            sanitized[key] = sanitize_schema(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def patch_gradio_schema_validator():
    """Monkey-patches Gradio para usar nosso sanitizador antes de processar schemas.
    
    Chamador: deve ser invocado NO INÍCIO de gradio_app.py, ANTES
    de criar a interface Gradio.
    """
    try:
        from gradio_client import utils as gradio_utils
        
        original_json_schema_to_python_type = (
            gradio_utils.json_schema_to_python_type
        )
        
        def patched_json_schema_to_python_type(schema: Dict) -> type:
            """Wrapper que sanitiza schema antes de processar."""
            try:
                # Sanitizar o schema de entrada
                cleaned_schema = sanitize_schema(schema)
                
                # Chamar o original com o schema limpo
                return original_json_schema_to_python_type(cleaned_schema)
            except Exception as e:
                logger.warning(
                    f"Erro ao processar schema (usando fallback): {e}. "
                    f"Schema: {json.dumps(schema, default=str, indent=2)[:500]}"
                )
                # Fallback: tentar com dict vazio se tudo falhar
                try:
                    return original_json_schema_to_python_type({})
                except Exception:
                    return dict  # Último recurso
        
        # Aplicar monkey-patch
        gradio_utils.json_schema_to_python_type = patched_json_schema_to_python_type
        logger.info("Gradio schema validator patched successfully")
        
    except Exception as e:
        logger.warning(
            f"Não foi possível aplicar Gradio schema patch: {e}. "
            f"Continuando sem patch (pode ter erros de schema)."
        )


if __name__ == "__main__":
    # Teste unitário simples
    test_schema = {
        "metadata": True,  # ← ERRADO
        "config": False,   # ← ERRADO
        "nested": {
            "value": "ok",
            "broken": True
        }
    }
    
    cleaned = sanitize_schema(test_schema)
    print("Original:", test_schema)
    print("Cleaned:", cleaned)
    assert cleaned["metadata"] == {"type": "object"}
    assert cleaned["config"] == {}
    print("Test passed!")
