"""Artefatos de inferência do treino: o ``.keras`` sem otimizador e o
``input_contract`` que a inferência lê para reproduzir o pré-processamento.

Saiu de ``trainer.py`` (P2-15).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


def save_inference_keras(model: "tf.keras.Model", path) -> None:
    """Salva um artefato de INFERÊNCIA (.keras) SEM o estado do otimizador.

    No Keras 3, `include_optimizer=False` é IGNORADO para o formato `.keras`: o
    estado do Adam (2 momentos por peso, ~2× os pesos) é sempre serializado,
    deixando o arquivo ~3× maior que o necessário para inferência.

    Removemos o otimizador ANTES de salvar e o restauramos DEPOIS. O grafo e os
    pesos ficam idênticos — a saída do modelo NÃO muda (neutro em acurácia) —,
    apenas o estado de treino deixa de ser gravado. Ganho típico: 561 MB → 188 MB
    (MultiscaleCNN), com load proporcionalmente mais rápido.

    Obs.: reconstruir via from_config+set_weights foi descartado por alterar a
    saída (NaN em modelos com BatchNormalization/camadas custom).
    """
    from app.utils.file_storage import break_hardlink

    path = str(path)
    break_hardlink(Path(path))
    saved_opt = getattr(model, "optimizer", None)
    try:
        try:
            model.optimizer = None
        except Exception as e:
            logger.debug(f"Não foi possível remover o otimizador: {e}")
        model.save(path)
    finally:
        # Restaura o otimizador para não afetar usos posteriores (ex.: continuar
        # o treino, avaliar com o mesmo objeto de modelo).
        if saved_opt is not None:
            try:
                model.optimizer = saved_opt
            except Exception:
                pass


def build_input_contract(
    model: tf.keras.Model,
    metadata: Optional[Dict] = None,
    *,
    scaler_applied: bool,
    temperature: float = 1.0,
    ood_threshold: Optional[float] = None,
    eer_threshold: Optional[float] = None,
    eer_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Constrói contrato de entrada para garantir consistência train/inference.

    O input_contract é salvo junto ao modelo e lido pelo FeaturePreparer
    na inferência para garantir que as mesmas features/formato sejam usados.
    """
    metadata = metadata or {}
    model_input_shape = list(model.input_shape[1:]) if model.input_shape else None

    # Determinar tipo de input a partir da arquitetura ou shape
    architecture = metadata.get("architecture", "")
    input_type = metadata.get("input_type", "features")
    input_format = metadata.get("input_format", "tabular")

    # Heurística: modelos com input (N, 1) provavelmente recebem áudio raw
    if (
        model_input_shape
        and len(model_input_shape) == 2
        and model_input_shape[-1] == 1
    ):
        input_type = "audio"
        input_format = "raw"
    # Modelos com input (H, W) ou (H, W, C) provavelmente usam spectrogram
    elif (
        model_input_shape
        and len(model_input_shape) in (2, 3)
        and (model_input_shape[-1] != 1 if len(model_input_shape) == 2 else True)
    ):
        if any(dim and dim > 10 for dim in model_input_shape[:2]):
            input_type = "features"
            input_format = "spectrogram"

    # Feature types se disponíveis
    feature_types = metadata.get("feature_types", None)
    if feature_types is None:
        feature_types_attr = getattr(model, "feature_types_used", None)
        if feature_types_attr:
            feature_types = list(feature_types_attr)

    contract = {
        "type": input_type,
        "format": input_format,
        "input_shape": model_input_shape,
        "architecture": architecture,
        "feature_types": feature_types,
        "sample_rate": metadata.get("sample_rate", 16000),
        "scaler_applied": scaler_applied,
        # Temperatura calibrada (Sprint 1.4) — aplicada na inferência pelo Predictor
        "temperature": float(temperature),
    }

    # Persiste o contrato canônico da arquitetura (frontend, crop e janela).
    # Sem esta fusão, AASIST/RawGAT-ST perdiam a política multicrop ao salvar.
    if architecture:
        try:
            from app.domain.models.architectures.registry import (
                get_architecture_info,
            )

            architecture_info = get_architecture_info(architecture)
            requirements = (
                dict(architecture_info.input_requirements)
                if architecture_info is not None
                else {}
            )
            for key in (
                "input_type",
                "feature_frontend",
                "target_sequence_length",
                "source_samples",
                "crop_strategy",
                "sample_rate",
                "preprocessing",
            ):
                if key in requirements:
                    contract[key] = requirements[key]
        except Exception as exc:
            logger.debug(
                "Contrato do registry não pôde ser incorporado: %s", exc
            )

    # Sprint 2.5: OOD threshold (energy-based). None se desabilitado/falhou.
    ood_t = ood_threshold
    if ood_t is not None:
        contract["ood_threshold"] = float(ood_t)

    # A inferencia precisa saber se a ULTIMA camada emite logits crus
    # (AASIST/AM-Softmax) ou ja probabilidades. Sem este campo, o Predictor
    # adivinhava pela faixa de valores enquanto o benchmark decidia pela
    # ativacao da camada — criterios diferentes, que divergem quando logits
    # caem por acaso em [0, 1] e somam ~1.
    try:
        from app.domain.services.detection.predictor import model_emits_logits

        is_logits = model_emits_logits(model)
        if is_logits is not None:
            contract["output_is_logits"] = bool(is_logits)
    except Exception as exc:  # noqa: BLE001 — contrato sem o campo ainda serve
        logger.debug("output_is_logits indisponivel: %s", exc)

    # Sprint 4.5: EER threshold (Equal Error Rate) — alternativa adaptativa
    # ao threshold 0.5 fixo. Predictor pode usar via flag use_eer_threshold.
    eer_t = eer_threshold
    eer_v = eer_value
    if eer_t is not None:
        contract["eer_threshold"] = float(eer_t)
    if eer_v is not None:
        contract["eer_value"] = float(eer_v)
    return contract
