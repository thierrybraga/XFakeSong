"""Treino do assistente pelo serviço de domínio: treinar → salvar → recarregar → detectar.

SUJEITO: `app.domain.services.interactive_training`. O pipeline saiu da aba
do Gradio (P1-4) e nada exercitava `_run_training` de ponta a ponta — o smoke
`test_wizard_pipeline.py` reimplementa o pré-processamento. Aqui o serviço
roda de verdade, num dataset sintético minúsculo, e o modelo salvo volta pela
mesma `DetectionService` que a aba Detectar usa. Nada toca `data/models`.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("tensorflow")
soundfile = pytest.importorskip("soundfile")

from app.domain.services import interactive_training as it  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    """6 bonafide (senoides) + 6 spoof (ruído), 1 s a 16 kHz."""
    rng = np.random.default_rng(3)
    t = np.arange(16000) / 16000.0
    for i in range(6):
        real = 0.3 * np.sin(2 * np.pi * (180 + 40 * i) * t)
        fake = 0.3 * rng.standard_normal(16000)
        for classe, onda in (("real", real), ("fake", fake)):
            pasta = tmp_path / "ds" / classe
            pasta.mkdir(parents=True, exist_ok=True)
            soundfile.write(
                str(pasta / f"{classe}_{i}.wav"), onda.astype("float32"), 16000
            )
    return tmp_path / "ds"


def _eventos(gerador):
    return list(gerador)


def _detectar(models_dir, nome, arquivo):
    from app.domain.services.detection_service import DetectionService

    servico = DetectionService(models_dir=str(models_dir))
    assert nome in servico.get_available_models()
    return servico.detect_from_file(arquivo, model_name=nome)


def test_treino_neural_emite_eventos_e_o_modelo_salvo_volta_na_deteccao(
    dataset, tmp_path
):
    # Pedido de parada pendente de um treino anterior: não pode matar este.
    parada = threading.Event()
    parada.set()
    progresso = []

    eventos = _eventos(
        it.InteractiveTrainingService().train(
            "Sonic Sleuth",
            str(dataset),
            epochs=2,
            batch_size=4,
            lr=1e-3,
            use_class_weighting=True,
            stop_event=parada,
            on_progress=lambda f, d: progresso.append((f, d)),
        )
    )

    tipos = [type(e).__name__ for e in eventos]
    assert tipos[:3] == ["DatasetLoading", "DatasetReady", "FitStarted"]
    # Com o pedido antigo não limpo, o Keras pararia ao fim da 1ª época.
    assert tipos.count("EpochCompleted") == 2
    assert tipos[-1] == "FitCompleted", eventos[-1]
    pronto = eventos[1]
    assert pronto.binary_map == {0: 1, 1: 0}  # fake -> 1, real -> 0
    concluido = eventos[-1]
    assert concluido.trained.kind == "keras"
    assert concluido.trained.input_contract["feature_frontend"] == "lfcc"
    assert len(concluido.y_true) == len(concluido.y_pred) > 0
    assert progresso[0] == (0.05, "Carregando dataset...")

    salvo = it.save_trained_model(concluido.trained, "Sonic Teste!", tmp_path / "m")
    assert salvo.slug == "sonic_teste"
    assert salvo.model_path.exists() and salvo.config_path.exists()
    with pytest.raises(FileExistsError):
        it.save_trained_model(concluido.trained, "sonic teste", tmp_path / "m")

    r = _detectar(tmp_path / "m", "sonic_teste", next((dataset / "real").glob("*.wav")))
    assert r.status.value == "success", r.errors


def test_treino_classico_extrai_ajusta_e_salva_com_scaler(dataset, tmp_path):
    eventos = _eventos(
        it.InteractiveTrainingService().train(
            "SVM",
            str(dataset),
            epochs=1,
            batch_size=4,
            lr=1e-3,
            use_class_weighting=False,
        )
    )

    tipos = [type(e).__name__ for e in eventos]
    assert tipos[:2] == ["DatasetLoading", "ClassicalStarted"]
    assert "ClassicalExtractionProgress" in tipos and "ClassicalFitting" in tipos
    concluido = eventos[-1]
    assert isinstance(concluido, it.ClassicalCompleted), eventos[-1]
    assert concluido.trained.kind == "sklearn"
    assert concluido.trained.scaler is not None
    assert concluido.feature_importances is None  # só o Random Forest tem
    assert concluido.n_real == concluido.n_fake == 6

    salvo = it.save_trained_model(concluido.trained, "svm teste", tmp_path / "m")
    assert (tmp_path / "m" / "svm_teste_scaler.pkl").exists()
    r = _detectar(tmp_path / "m", salvo.slug, next((dataset / "fake").glob("*.wav")))
    assert r.status.value == "success", r.errors


def test_dataset_sem_uma_das_classes_e_recusado(tmp_path):
    vazio = tmp_path / "so_real" / "real"
    vazio.mkdir(parents=True)
    soundfile.write(str(vazio / "a.wav"), np.zeros(16000, dtype="float32"), 16000)

    eventos = _eventos(
        it.InteractiveTrainingService().train(
            "Random Forest",
            str(tmp_path / "so_real"),
            epochs=1,
            batch_size=4,
            lr=1e-3,
            use_class_weighting=False,
        )
    )

    assert isinstance(eventos[-1], it.ClassicalDatasetInsufficient)
    assert (eventos[-1].n_real, eventos[-1].n_fake) == (1, 0)
