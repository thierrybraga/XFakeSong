"""Pontuação centrada no limiar: o número mostrado nunca contradiz o veredito.

SUJEITO: `app/domain/services/detection/operating_point.py`. Um "real" com P(fake) de 0,80
e corte em 0,94 aparecia como "probabilidade de ser sintética: 80%" e "confiança: 20%".
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.services.detection.operating_point import (
    fake_score,
    fuse_scores,
    verdict_confidence,
)

# Os limiares calibrados vão de 0,05 (SVM) a 0,99 (Hybrid CNN-Transformer).
LIMIARES = [0.047, 0.39, 0.482, 0.5, 0.61, 0.941, 0.988]


def test_caso_relatado_fica_coerente():
    assert fake_score(0.802, 0.941) == pytest.approx(0.426, abs=1e-3)
    assert verdict_confidence(0.802, 0.941) == pytest.approx(0.574, abs=1e-3)


@pytest.mark.parametrize("limiar", LIMIARES)
def test_limiar_vai_para_o_centro_e_os_extremos_ficam(limiar):
    assert fake_score(limiar, limiar) == pytest.approx(0.5)
    assert fake_score(0.0, limiar) == 0.0
    assert fake_score(1.0, limiar) == 1.0


@pytest.mark.parametrize("limiar", LIMIARES)
def test_veredito_e_pontuacao_nunca_discordam(limiar):
    for p in np.linspace(0.0, 1.0, 2001):
        assert (fake_score(p, limiar) >= 0.5) == (p >= limiar)


@pytest.mark.parametrize("limiar", LIMIARES)
def test_escala_preserva_a_ordem_dos_audios(limiar):
    """Monotônica: AUC e EER calculados sobre ela são os mesmos da escala bruta."""
    pontuacoes = [fake_score(p, limiar) for p in np.linspace(0.0, 1.0, 2001)]
    assert all(a < b for a, b in zip(pontuacoes, pontuacoes[1:]))


@pytest.mark.parametrize("limiar", LIMIARES)
def test_confianca_vai_de_meio_a_um(limiar):
    for p in np.linspace(0.0, 1.0, 501):
        assert 0.5 <= verdict_confidence(p, limiar) <= 1.0
    assert verdict_confidence(limiar, limiar) == pytest.approx(0.5)


def test_limiar_de_meio_e_a_identidade():
    """Modelo sem calibração não muda: é o caso dos testes e dos artefatos antigos."""
    for p in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert fake_score(p, 0.5) == pytest.approx(p)
        assert fake_score(p) == pytest.approx(p)
    assert verdict_confidence(0.8) == pytest.approx(0.8)
    assert verdict_confidence(0.2) == pytest.approx(0.8)


@pytest.mark.parametrize("limiar", [0.0, 1.0])
def test_limiar_degenerado_nao_divide_por_zero(limiar):
    assert fake_score(0.3, limiar) == pytest.approx(0.3)


def test_valor_fora_da_faixa_e_limitado():
    assert fake_score(1.2, 0.9) == 1.0
    assert fake_score(-0.1, 0.9) == 0.0


# ───────────────────────── fusão entre modelos ─────────────────────────


@pytest.mark.parametrize("fusao", ["weighted_avg", "soft_voting", "max_conf", "majority_vote"])
def test_fusao_nao_inverte_modelos_unanimes(fusao):
    assert fuse_scores([0.1, 0.45, 0.3], fusao) < 0.5
    assert fuse_scores([0.9, 0.55, 0.7], fusao) >= 0.5


def test_fusao_ponderada_respeita_os_pesos():
    assert fuse_scores([0.2, 0.8], "weighted_avg", [3.0, 1.0]) == pytest.approx(0.35)


def test_max_conf_pega_o_mais_distante_do_limiar():
    assert fuse_scores([0.55, 0.05, 0.9], "max_conf") == pytest.approx(0.05)


def test_voto_majoritario_nunca_contradiz_a_maioria():
    """A troca antiga pela média perto de 0,5 podia inverter o voto."""
    assert fuse_scores([0.55, 0.51, 0.02], "majority_vote") == pytest.approx(0.53)
    assert fuse_scores([0.45, 0.49, 0.98], "majority_vote") == pytest.approx(0.47)


def test_empate_no_voto_e_decidido_pela_media():
    assert fuse_scores([0.9, 0.4], "majority_vote") == pytest.approx(0.65)


def test_fusao_desconhecida_e_erro():
    with pytest.raises(ValueError):
        fuse_scores([0.5], "mediana")


def _servico_com_vereditos(vereditos):
    """`detect_single` falso: nome do modelo -> (P(fake), limiar), ou None para falhar."""
    from app.core.contracts.audio import DeepfakeDetectionResult
    from app.core.contracts.base import ProcessingResult, ProcessingStatus
    from app.domain.services.detection_service import DetectionService

    servico = DetectionService.__new__(DetectionService)

    def detect_single(_audio, model_name=None, use_tta=False):
        if vereditos[model_name] is None:
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=["falhou"])
        p, limiar = vereditos[model_name]
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=DeepfakeDetectionResult(
                is_fake=p >= limiar,
                confidence=verdict_confidence(p, limiar),
                probabilities={"fake": p, "real": 1.0 - p},
                model_name=model_name,
                features_used=[],
                metadata={
                    "fake_score": fake_score(p, limiar),
                    "classification_threshold": limiar,
                },
            ),
        )

    servico.detect_single = detect_single
    return servico


def test_dois_modelos_que_dizem_real_nao_viram_fake_na_fusao():
    """Somar P(fake) dava 0,61: acima de 0,5, embora os dois tivessem dito "real"."""
    servico = _servico_com_vereditos(
        {"aasist": (0.80, 0.941), "conformer": (0.42, 0.482)}
    )
    resultado = servico.detect_multi_model(
        None, ["aasist", "conformer"], fusion="soft_voting"
    )
    assert resultado.data.is_fake is False
    assert resultado.data.metadata["fake_votes"] == 0
    assert resultado.data.metadata["classification_threshold"] == 0.5
    assert resultado.data.metadata["fake_score"] == pytest.approx(
        resultado.data.probabilities["fake"]
    )
    assert resultado.data.confidence >= 0.5


def test_modelo_que_falha_nao_desalinha_os_pesos():
    """Os pesos eram cortados por posição, e o de um modelo ia para o seguinte."""
    servico = _servico_com_vereditos(
        {"a": None, "b": (0.9, 0.5), "c": (0.1, 0.5)}
    )
    resultado = servico.detect_multi_model(
        None, ["a", "b", "c"], fusion="weighted_avg", weights=[0.8, 0.1, 0.1]
    )
    assert resultado.data.metadata["weights"] == pytest.approx([0.5, 0.5])
    assert resultado.data.probabilities["fake"] == pytest.approx(0.5)


def test_resultado_sem_pontuacao_usa_o_limiar_do_modelo():
    servico = _servico_com_vereditos({"a": (0.8, 0.9)})
    original = servico.detect_single

    def sem_pontuacao(*args, **kwargs):
        resultado = original(*args, **kwargs)
        del resultado.data.metadata["fake_score"]
        return resultado

    servico.detect_single = sem_pontuacao
    resultado = servico.detect_multi_model(None, ["a"], fusion="soft_voting")
    assert resultado.data.is_fake is False
    assert resultado.data.probabilities["fake"] == pytest.approx(fake_score(0.8, 0.9))
