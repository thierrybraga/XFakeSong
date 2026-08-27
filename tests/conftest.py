import os
import pathlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.contracts.base import ProcessingResult, ProcessingStatus
from app.dependencies import (
    get_detection_service,
    get_training_service,
    get_upload_service,
)
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_service import TrainingService
from app.domain.services.upload_service import AudioUploadService
from app.interfaces.web.main_fastapi import app
from app.interfaces.web.schemas.api_models import DatasetMetadata

_TEST_CATEGORIES = ("unit", "api", "functional", "integration", "smoke")


def pytest_collection_modifyitems(config, items):
    """Marca cada teste pela categoria da pasta onde ele vive."""
    for item in items:
        parts = set(pathlib.Path(str(item.fspath)).parts)
        for cat in _TEST_CATEGORIES:
            if cat in parts:
                item.add_marker(getattr(pytest.mark, cat))
                break

# Configurar API Key para testes
os.environ["XFAKESONG_API_KEY"] = "test-api-key"

# SEGFAULT INTERMITENTE NA SUÍTE (2026-08-17): uma execução em três terminava
# com `Segmentation fault (core dumped)` e ZERO testes falhando — o crash
# acontecia na FINALIZAÇÃO do interpretador, com o traço apontando para o
# thread alimentador das filas do `loky` (joblib) sobre módulos do scipy.
#
# Causa: SVM e RandomForest têm `n_jobs=-1` por default
# (`architectures/{svm,random_forest}.py`), então todo teste que os ajusta sobe
# workers `loky` em todos os cores — dentro de um processo que já carregou o
# TensorFlow. Os dois runtimes disputando o desligamento é uma combinação
# conhecida por travar ou crashar no `atexit`.
#
# Sequencial no teste NÃO muda o que está sob contrato: nenhum teste afirma
# nada sobre paralelismo, e o `n_jobs=-1` do código de produção segue intacto —
# esta variável só afeta o processo do pytest.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")


@pytest.fixture(autouse=True, scope="session")
def _isolate_models_dir(tmp_path_factory):
    """Nenhum teste escreve em ``data/models`` — o diretório é de PRODUÇÃO.

    `benchmarks/runner.py::_models_dir` cai em `cfg.models_dir`, cujo default é
    `data/models`: um diretório GLOBAL chaveado só pela arquitetura. Onze
    `BenchmarkConfig` da suíte não passam `models_dir`, então cada execução de
    `run_benchmark` com SVM gravava por cima de `data/models/bench_svm.pkl`.

    Foi exatamente assim que o artefato do `clean_benchmark_15k` (63 features,
    3,6 MB) virou um `.pkl` de smoke com 47 KB e 8 amostras — o
    `model_artifact_fingerprint` do run registrou `integrity: size_mismatch`, e
    o modelo que produziu as métricas publicadas deixou de existir.

    `_models_dir` consulta as variáveis nesta ORDEM, parando na primeira
    definida: `MODELS_DIR`, `DEEPFAKE_MODELS_DIR`, `XFAKE_MODELS_DIR`, depois
    `XFAKE_STORAGE_DIR`/`DEEPFAKE_STORAGE_DIR`, e só então `cfg.models_dir`.
    Por isso a fixture escreve em TODAS elas.

    Definir só `XFAKE_MODELS_DIR` não bastava — e essa foi a primeira versão
    desta fixture, que não protegia nada: o `.env` do projeto declara
    `DEEPFAKE_MODELS_DIR=./data/models` e o `python-dotenv` o carrega no import
    de `app.*`, que este próprio conftest faz no topo. A variável do `.env`
    vence por vir antes na cadeia, e o diretório de produção seguia exposto.

    Os testes que exercitam a resolução de caminho apagam essas variáveis via
    `monkeypatch.delenv` e passam `models_dir` explícito — continuam válidos,
    porque o `monkeypatch` restaura no teardown.
    """
    sandbox = tmp_path_factory.mktemp("models_dir_sandbox")
    variaveis = (
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
    )
    anteriores = {nome: os.environ.get(nome) for nome in variaveis}
    for nome in variaveis:
        os.environ[nome] = str(sandbox)
    yield sandbox
    for nome, valor in anteriores.items():
        if valor is None:
            os.environ.pop(nome, None)
        else:
            os.environ[nome] = valor


@pytest.fixture(autouse=True)
def _reset_tf_mixed_precision_policy():
    """BUG FIX (flakiness sistêmica por estado global do Keras): a suíte

    completa expôs testes que passam isolados mas falham quando rodados
    depois de outro teste que chama
    ``tf.keras.mixed_precision.set_global_policy("mixed_float16")`` sem
    reverter — a política é global de PROCESSO, não de teste, então vaza
    para qualquer teste seguinte na mesma sessão pytest.

    Dois sintomas já confirmados (causa raiz idêntica, reproduzida
    isoladamente): ``test_trainer_preserves_precompiled_loss_and_optimizer``
    (o Keras encapsula o otimizador precompilado num ``LossScaleOptimizer``
    ao chamar ``model.compile()`` sob a política vazada) e
    ``test_architecture_builds_and_forwards`` para AASIST/RawGAT-ST/Ensemble
    (camadas customizadas — ``SincConvLayer`` e afins — produzem saída
    não-finita sob ``float16``). Reseta para ``float32`` antes de cada teste
    e restaura a política anterior depois, para não mascarar os testes que
    *de fato* querem exercitar mixed precision (ex.:
    ``test_sinc_layers_mixed_precision.py``, que já gerencia a própria
    política com ``try/finally``).
    """
    try:
        import tensorflow as tf
    except ImportError:
        yield
        return
    original = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    yield
    tf.keras.mixed_precision.set_global_policy(original)


@pytest.fixture
def api_key_headers():
    return {"X-API-Key": "test-api-key"}


@pytest.fixture
def mock_detection_service():
    mock = MagicMock(spec=DetectionService)
    mock.default_model = "test_model"
    mock.loaded_models = {"test_model": MagicMock()}
    mock.get_available_models.return_value = ["test_model"]
    mock.get_available_architectures.return_value = ["AASIST", "RawNet2"]
    return mock


@pytest.fixture
def mock_upload_service(tmp_path):
    mock = MagicMock(spec=AudioUploadService)
    mock.upload_directory = tmp_path / "uploads"
    mock.upload_directory.mkdir()
    mock.SUPPORTED_FORMATS = AudioUploadService.SUPPORTED_FORMATS
    mock.MAX_FILE_SIZE = AudioUploadService.MAX_FILE_SIZE

    # Mock create_dataset return.
    # O serviço REAL retorna ProcessingResult[DatasetMetadata] (não o
    # DatasetMetadata cru), e o router faz result.status / result.data.
    # O mock precisa espelhar esse contrato senão result.status estoura
    # AttributeError.
    def side_effect_create(name, type, desc):
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=DatasetMetadata(
                name=name,
                dataset_type=type,
                description=desc or "",
                file_count=0,
                total_size=0,
                total_duration=0.0,
                created_at=None,
                file_paths=[]
            ),
            metadata={"message": f"Dataset {name} criado com sucesso"},
        )
    mock.create_dataset.side_effect = side_effect_create

    # Mock delete_dataset return
    mock.delete_dataset.return_value = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        metadata={"message": "Dataset deleted"}
    )

    return mock


@pytest.fixture
def mock_training_service():
    mock = MagicMock(spec=TrainingService)
    # Mock train_model return
    mock_result = MagicMock()
    mock_result.status = ProcessingStatus.SUCCESS
    mock_result.data = MagicMock()
    mock_result.data.metrics = {"accuracy": 0.95}
    mock_result.data.path = "/tmp/model.h5"
    mock.train_model.return_value = mock_result
    return mock


@pytest.fixture
def client(
    mock_detection_service,
    mock_upload_service,
    mock_training_service
):
    # Override dependencies
    app.dependency_overrides[get_detection_service] = \
        lambda: mock_detection_service
    app.dependency_overrides[get_upload_service] = \
        lambda: mock_upload_service
    app.dependency_overrides[get_training_service] = \
        lambda: mock_training_service

    with TestClient(app) as c:
        yield c

    # Clean up
    app.dependency_overrides = {}
