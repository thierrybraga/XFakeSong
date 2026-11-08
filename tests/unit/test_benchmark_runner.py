"""Lançador do benchmark do artigo: comando, travas, processo e progresso.

SUJEITO: `app/domain/services/benchmark/runner.py`. O orquestrador real (dias
de GPU) nunca roda aqui: o ciclo iniciar → acompanhar → parar usa um processo
Python curto no lugar dele.
"""

from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from app.domain.services.benchmark import runner as rn


@pytest.fixture
def spec(tmp_path):
    paths = SimpleNamespace(
        root=tmp_path / "repo",
        data=tmp_path / "data",
        results=tmp_path / "data" / "results",
        models=tmp_path / "data" / "models",
    )
    paths.results.mkdir(parents=True)
    paths.root.mkdir()
    return rn.ArticleRunSpec.from_paths(paths, run_name="tcc_reproducao_20261107-1200")


def _dataset(spec, contagens=None, tamanho=None):
    spec.dataset.parent.mkdir(parents=True, exist_ok=True)
    spec.dataset.write_bytes(b"x" * 16)
    spec.test_lock.write_text(
        json.dumps(
            {
                "dataset_size_bytes": 16 if tamanho is None else tamanho,
                "split_counts": contagens or rn.ARTICLE_SPLIT_COUNTS,
            }
        ),
        encoding="utf-8",
    )


def test_comando_e_o_do_artigo(spec):
    comando = rn.build_command(spec)
    assert comando[2:6] == [
        "--dataset",
        str(spec.dataset),
        "--test-lock",
        str(spec.test_lock),
    ]
    flags = comando[6:-2]
    assert flags == list(rn.ARTICLE_FLAGS)
    assert "--experiment-scope" not in comando, "o orquestrador recusa essa flag"
    assert ["--scope", "official"] == flags[:2]
    assert "--checkpoint-monitor" in flags and "val_eer" in flags
    assert comando[-2:] == ["--out", str(spec.out_dir)]
    assert rn.build_command(spec, resume=True)[-1] == "--resume"


def test_nome_de_run_so_com_prefixo_e_sem_barras(tmp_path):
    paths = SimpleNamespace(
        root=tmp_path, data=tmp_path, results=tmp_path, models=tmp_path
    )
    for ruim in ("benchmark", "tcc_reproducao_../x", "outro"):
        with pytest.raises(ValueError):
            rn.ArticleRunSpec.from_paths(paths, run_name=ruim)
    assert rn.ArticleRunSpec.from_paths(paths).run_name.startswith(rn.RUN_PREFIX)


def test_modelos_do_run_vao_para_pasta_propria(spec):
    ambiente = rn.child_env(spec, base={"MODELS_DIR": "data/models", "PATH": "x"})
    for chave in ("MODELS_DIR", "DEEPFAKE_MODELS_DIR", "XFAKE_MODELS_DIR"):
        assert ambiente[chave] == str(spec.models_dir)
    assert spec.models_dir.name == spec.run_name


def test_comando_docker_isola_modelos_e_pasta(spec):
    ps = rn.docker_command(spec, "powershell")
    bash = rn.docker_command(spec, "bash")
    for texto in (ps, bash):
        assert f"MODELS_DIR=/app/data/models/{spec.run_name}" in texto
        assert f"data/results/{spec.run_name}" in texto
        assert "run --rm" in texto
    assert ps.startswith("$env:") and bash.startswith("XFAKE_BENCHMARK_DATASET=")


def _todas_ok(spec, **kw):
    return rn.preflight(
        spec,
        gpu_available=lambda: True,
        auth_enabled=True,
        env={rn.ENABLE_FLAG: "1"},
        **kw,
    )


def test_preflight_passa_com_tudo_pronto(spec, monkeypatch):
    _dataset(spec)
    monkeypatch.setattr(rn, "MIN_FREE_GB", 0.0)
    checks = _todas_ok(spec)
    assert rn.can_start(checks), [c for c in checks if not c.ok]


@pytest.mark.parametrize(
    "mudanca, nome",
    [
        ({"env": {}}, "Execução habilitada"),
        ({"auth_enabled": False}, "Login ativo"),
        ({"gpu_available": lambda: False}, "GPU visível ao TensorFlow"),
    ],
)
def test_preflight_bloqueia(spec, monkeypatch, mudanca, nome):
    _dataset(spec)
    monkeypatch.setattr(rn, "MIN_FREE_GB", 0.0)
    argumentos = {
        "gpu_available": lambda: True,
        "auth_enabled": True,
        "env": {rn.ENABLE_FLAG: "1"},
        **mudanca,
    }
    checks = rn.preflight(spec, **argumentos)
    assert not rn.can_start(checks)
    assert [c.name for c in checks if not c.ok] == [nome]


def test_preflight_bloqueia_dataset_e_selo_errados(spec, monkeypatch):
    monkeypatch.setattr(rn, "MIN_FREE_GB", 0.0)
    falhas = {c.name for c in _todas_ok(spec) if not c.ok}
    assert {"Dataset do artigo", "Selo do teste"} <= falhas
    _dataset(spec, contagens={"y_train": 1, "y_val": 1, "y_test": 1}, tamanho=99)
    falhas = {c.name for c in _todas_ok(spec) if not c.ok}
    assert {"Dataset do artigo", "Selo do teste"} <= falhas


def test_preflight_recusa_pasta_existente_e_retomada_de_run_alheio(spec, monkeypatch):
    _dataset(spec)
    monkeypatch.setattr(rn, "MIN_FREE_GB", 0.0)
    spec.out_dir.mkdir(parents=True)
    assert "Pasta de saída nova" in {c.name for c in _todas_ok(spec) if not c.ok}
    assert "Run criado pela aba" in {
        c.name for c in _todas_ok(spec, resume=True) if not c.ok
    }


def test_trava_de_processo_morto_e_descartada(spec):
    trava = spec.results_root / rn.LOCK_NAME
    trava.write_text(json.dumps({"pid": 999999999, "run_name": "x"}), encoding="utf-8")
    assert rn.active_run(spec.results_root) is None
    assert not trava.exists()


def test_start_recusa_com_checagem_falha(spec):
    with pytest.raises(RuntimeError):
        rn.start(spec, [rn.Check("GPU", False, "sem GPU")])


def test_ciclo_iniciar_acompanhar_parar(spec):
    modelo = spec.out_dir / "aasist"
    script = (
        "import json, pathlib, time, sys\n"
        f"m = pathlib.Path({str(modelo)!r}); m.mkdir(parents=True, exist_ok=True)\n"
        "(m / 'run_fingerprint.json').write_text('{}')\n"
        "(m / 'run.log').write_text('[TRAIN] AASIST epoch=12/100 epoch_s=60 eta_min=88 val_eer=0.05\\n')\n"
        "print('orquestrador de teste no ar', flush=True)\n"
        "time.sleep(120)\n"
    )
    trava = rn.start(
        spec, [rn.Check("ok", True, "")], command=[sys.executable, "-c", script]
    )
    try:
        limite = time.time() + 20
        estado = rn.run_status(spec.results_root)
        while time.time() < limite and (
            estado.current is None or estado.current.epoch is None
        ):
            time.sleep(0.3)
            estado = rn.run_status(spec.results_root)
        assert estado.active and estado.run_name == spec.run_name
        assert (estado.current.model, estado.current.epoch, estado.current.epochs) == (
            "aasist",
            12,
            100,
        )
        assert estado.current.eta_min == 88.0
        marcador = json.loads((spec.out_dir / rn.UI_MARKER).read_text(encoding="utf-8"))
        assert marcador["events"][0]["event"] == "start"
        assert (spec.models_dir).is_dir()
    finally:
        parada = rn.stop(spec.results_root)
    assert parada["pid"] == trava["pid"]
    limite = time.time() + 15
    while time.time() < limite and rn.pid_alive(trava["pid"]):
        time.sleep(0.2)
    assert not rn.pid_alive(trava["pid"])
    assert rn.active_run(spec.results_root) is None
    eventos = json.loads((spec.out_dir / rn.UI_MARKER).read_text(encoding="utf-8"))[
        "events"
    ]
    assert [e["event"] for e in eventos] == ["start", "stop"]


def test_estado_de_run_parado_le_o_resumo(spec):
    spec.out_dir.mkdir(parents=True)
    (spec.out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "models": [{"model": "SVM", "status": "ok", "elapsed_s": 3200}],
            }
        ),
        encoding="utf-8",
    )
    (spec.out_dir / rn.ORCHESTRATOR_LOG).write_text(
        "a\nb\n[OK] SVM\n", encoding="utf-8"
    )
    estado = rn.run_status(spec.results_root, spec.out_dir)
    assert not estado.active and estado.summary_status == "partial"
    assert [(m.model, m.status) for m in estado.models] == [("SVM", "ok")]
    assert estado.log_tail[-1] == "[OK] SVM"


def test_consolidacao_so_de_run_da_aba_e_dentro_dele(spec, monkeypatch):
    with pytest.raises(RuntimeError):
        rn.consolidate(spec)
    spec.out_dir.mkdir(parents=True)
    (spec.out_dir / rn.UI_MARKER).write_text("{}", encoding="utf-8")
    chamadas = []

    def falso(comando, cwd, env, log):
        chamadas.append(comando)
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(rn, "_spawn", falso)
    assert rn.consolidate(spec) == 1
    (comando,) = chamadas
    destino = str(spec.out_dir / "consolidated")
    assert comando[comando.index("--out") + 1] == destino
    assert comando[comando.index("--copy-to") + 1].startswith(destino)
    assert "paper" not in " ".join(comando)
