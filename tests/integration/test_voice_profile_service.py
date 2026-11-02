"""`VoiceProfileService`: perfil → amostras → validação → treino → verificação.

SUJEITO: o fluxo público do serviço de perfis de voz, que não tinha teste
nenhum. Fixa o comportamento antes de o módulo (1.445 linhas: CRUD, features,
fakes sintéticos, seis construtores Keras, treino e detecção) ser repartido na
Fase 4 do plano de refatoração (P1-10): os mesmos testes têm de passar antes e
depois. Banco SQLite isolado em `tmp_path`; nada toca `data/`.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.db import models  # noqa: E402,F401  (registra as tabelas)
from app.core.db.session import Base  # noqa: E402
from app.domain.services import voice_profile_service as vps  # noqa: E402


@pytest.fixture
def servico(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'perfis.db').as_posix()}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(vps, "SessionLocal", sessionmaker(bind=engine))
    yield vps.VoiceProfileService(profiles_dir=str(tmp_path / "perfis"))
    engine.dispose()


def _voz(f0: float, segundos: float, semente: int) -> bytes:
    """WAV 16 kHz com harmônicos e vibrato — algo que o pyin reconhece como voz."""
    sr = 16000
    t = np.arange(int(sr * segundos)) / sr
    vibrato = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    fase = 2 * np.pi * np.cumsum(vibrato) / sr
    onda = sum(np.sin(k * fase) / k for k in range(1, 6))
    onda += 0.01 * np.random.default_rng(semente).standard_normal(len(t))
    buffer = io.BytesIO()
    soundfile.write(buffer, (0.3 * onda / np.abs(onda).max()).astype("float32"), sr,
                    format="WAV")
    return buffer.getvalue()


def _amostras(n: int):
    return [(f"frase_{i}.wav", _voz(110 + 15 * i, 2.0 + 0.6 * i, i)) for i in range(n)]


def test_perfil_completo_com_svm(servico):
    perfil = servico.create_profile("Ana", email="ana@exemplo.org", architecture="svm")
    assert perfil.status == vps.ProfileStatus.CREATED.value

    adicionadas = servico.add_audio_samples(perfil.id, _amostras(4))
    assert adicionadas["success"] and adicionadas["total_samples"] == 4
    assert servico.get_profile(perfil.id).status == vps.ProfileStatus.COLLECTING.value
    assert servico.remove_audio_sample(perfil.id, "frase_3.wav")
    info = servico.get_dataset_info(perfil.id)
    assert [f["filename"] for f in info["files"]] == [
        "frase_0.wav", "frase_1.wav", "frase_2.wav",
    ]

    validacao = servico.validate_dataset_for_training(perfil.id)
    assert validacao["valid"], validacao
    assert validacao["warnings"]  # abaixo do recomendado

    mensagens = []
    resultado = servico.train_profile_model(
        perfil.id, progress_callback=mensagens.append
    )
    assert resultado["success"], resultado
    metricas = resultado["metrics"]
    assert set(metricas) == {
        "accuracy", "val_accuracy", "precision", "recall", "f1_score", "eer",
        "confusion_matrix", "loss", "val_loss", "epochs_trained",
    }
    assert any("Extraindo features" in m for m in mensagens)
    assert any("SVM" in m for m in mensagens)

    treinado = servico.get_profile(perfil.id)
    assert treinado.status == vps.ProfileStatus.READY.value
    assert treinado.model_path.endswith("profile_model.pkl")
    cfg = treinado.training_config
    assert cfg["architecture"] == "svm"
    assert cfg["num_real_samples"] == 3
    assert cfg["num_fake_samples"] >= 3
    assert cfg["feature_dim"] == 144
    assert cfg["stratified_split"] is True

    deteccao = servico.detect_with_profile(
        perfil.id, str(servico._samples_dir(perfil.id) / "frase_0.wav")
    )
    assert deteccao["success"], deteccao
    assert deteccao["details"]["model_type"] == "classical_ml"
    assert deteccao["details"]["feature_dim"] == 144
    assert deteccao["is_fake"] != deteccao["is_authentic"]

    assert servico.delete_profile(perfil.id)
    assert servico.get_profile(perfil.id) is None
    assert not servico._profile_dir(perfil.id).exists()


def test_perfil_treina_rede_keras_e_verifica(servico):
    perfil = servico.create_profile("Bia", architecture="sonic_sleuth")
    servico.add_audio_samples(perfil.id, _amostras(3))

    resultado = servico.train_profile_model(perfil.id, epochs=2, batch_size=4)

    assert resultado["success"], resultado
    metricas = resultado["metrics"]
    assert 1 <= metricas["epochs_trained"] <= 2
    assert {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(metricas["history"])
    treinado = servico.get_profile(perfil.id)
    assert treinado.model_path.endswith("profile_model.h5")
    assert treinado.training_config["epochs"] == 2

    deteccao = servico.detect_with_profile(
        perfil.id, str(servico._samples_dir(perfil.id) / "frase_1.wav")
    )
    assert deteccao["success"], deteccao
    assert deteccao["details"]["model_type"] == "deep_learning"
    assert 0.0 <= deteccao["raw_score"] <= 1.0


def test_dataset_abaixo_do_minimo_nao_treina(servico):
    perfil = servico.create_profile("Caio")
    servico.add_audio_samples(perfil.id, _amostras(1))

    resultado = servico.train_profile_model(perfil.id)

    assert resultado["success"] is False
    assert "Mínimo de 3 amostras" in resultado["error"]
    assert servico.get_profile(perfil.id).status == vps.ProfileStatus.COLLECTING.value


@pytest.mark.parametrize(
    ("arquitetura", "nome_do_modelo"),
    [
        ("aasist", "attention_profile"),
        ("conformer", "transformer_profile"),
        ("rawnet2", "conv1d_profile"),
        ("efficientnet_lstm", "recurrent_profile"),
        ("multiscale_cnn", "multihead_profile"),
    ],
)
def test_cada_familia_ganha_o_construtor_do_perfil(arquitetura, nome_do_modelo):
    from app.domain.services.voice_profile.models import build_profile_model

    modelo = build_profile_model(144, arquitetura)
    assert modelo.name == nome_do_modelo
    assert modelo.output_shape == (None, 1)


def test_arquitetura_desconhecida_cai_no_mlp():
    from app.domain.services.voice_profile.models import build_profile_model

    modelo = build_profile_model(144, "nao_existe")
    assert modelo.output_shape == (None, 1)
    assert modelo.name.startswith("sequential")
