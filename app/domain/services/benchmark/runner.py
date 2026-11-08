"""Execução do benchmark do artigo a partir da aplicação, com travas.

Reproduz o estudo do TCC exatamente como o artigo o descreve: o recorte de 15
mil amostras com o selo do teste, as 11 entradas oficiais e as flags abaixo
(``ARTICLE_FLAGS``, a fonte única que a aba, a API e os documentos usam). O
orquestrador ``benchmarks/run_models_sequential.py`` roda como processo
separado e destacado, e sobrevive a um reinício do app.

Travas, cada uma contra um dano medido:

- **pasta nova a cada run** (``data/results/tcc_reproducao_<data>``). Um
  ``--resume`` numa pasta existente (por exemplo ``data/results/benchmark``)
  move tudo para ``_superseded`` e retreina;
- **modelos em pasta própria** (``data/models/tcc_reproducao_<data>``), por
  ``MODELS_DIR``/``DEEPFAKE_MODELS_DIR``/``XFAKE_MODELS_DIR`` no ambiente do
  processo filho. Os padrões gravariam por cima dos modelos da detecção;
- **um run por vez**: arquivo de trava com o pid; trava de processo morto é
  descartada;
- **pré-condições**: GPU visível ao TensorFlow (sem ela o plano cai em silêncio
  para CPU, com lotes diferentes dos do artigo), ``XFAKE_ENABLE_BENCHMARK_RUN=1``,
  login do Gradio ativo (o app pode estar exposto à internet), NPZ e selo do
  teste conferidos, disco livre;
- **consolidação dentro do run**, nunca nos padrões, que gravam em
  ``data/results/paper``.

O progresso sai dos arquivos que o orquestrador grava: ``run_summary.json`` ao
fim de cada modelo e ``<modelo>/run.log`` ao vivo.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess  # nosec B404 - lança só o orquestrador do benchmark e o taskkill
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

ARTICLE_DATASET = "benchmark_dataset_15k.npz"
ARTICLE_SPLIT_COUNTS = {"y_train": 12162, "y_val": 1456, "y_test": 1382}
#: Flags do artigo (``main.tex``, comando de reprodução; inventário canônico).
#: ``--scope`` e não ``--experiment-scope``: o orquestrador recusa a segunda.
ARTICLE_FLAGS: tuple[str, ...] = (
    "--scope",
    "official",
    "--epochs",
    "100",
    "--batch-size",
    "32",
    "--seed",
    "42",
    "--device-profile",
    "gpu",
    "--latency-runs",
    "30",
    "--snr",
    "30",
    "20",
    "10",
    "5",
    "--train-aug-snr",
    "30",
    "20",
    "10",
    "--train-noise-copies",
    "1",
    "--band-correction-hz",
    "7500",
    "--checkpoint-monitor",
    "val_eer",
)
RUN_PREFIX = "tcc_reproducao_"
ENABLE_FLAG = "XFAKE_ENABLE_BENCHMARK_RUN"
LOCK_NAME = ".benchmark_run.lock"
UI_MARKER = "launched_by_ui.json"
ORCHESTRATOR_LOG = "orchestrator.log"
MIN_FREE_GB = 20.0
_TRAIN_LINE = re.compile(r"\[TRAIN\].*?epoch=(\d+)/(\d+)(?:.*?eta_min=([\d.]+))?")


@dataclass(frozen=True)
class ArticleRunSpec:
    """Onde o run do artigo lê e grava; tudo derivado dos caminhos operacionais."""

    project_root: Path
    data_dir: Path
    results_root: Path
    models_root: Path
    run_name: str
    python: str = sys.executable

    @property
    def dataset(self) -> Path:
        return self.data_dir / "datasets" / ARTICLE_DATASET

    @property
    def test_lock(self) -> Path:
        return self.dataset.with_name(self.dataset.name + ".test-lock.json")

    @property
    def out_dir(self) -> Path:
        return self.results_root / self.run_name

    @property
    def models_dir(self) -> Path:
        return self.models_root / self.run_name

    @property
    def orchestrator(self) -> Path:
        return self.project_root / "benchmarks" / "run_models_sequential.py"

    @classmethod
    def from_paths(
        cls, paths: Any, run_name: Optional[str] = None, now: Optional[datetime] = None
    ) -> "ArticleRunSpec":
        """Spec a partir de ``OperationalPaths``; sem nome, um run novo."""
        nome = run_name or RUN_PREFIX + (now or datetime.now()).strftime("%Y%m%d-%H%M")
        if not nome.startswith(RUN_PREFIX) or "/" in nome or "\\" in nome:
            raise ValueError(f"nome de run inválido: {nome!r}")
        return cls(
            project_root=Path(paths.root),
            data_dir=Path(paths.data),
            results_root=Path(paths.results),
            models_root=Path(paths.models),
            run_name=nome,
        )


def build_command(spec: ArticleRunSpec, resume: bool = False) -> List[str]:
    """O comando do artigo, com a pasta de saída do run."""
    comando = [
        spec.python,
        str(spec.orchestrator),
        "--dataset",
        str(spec.dataset),
        "--test-lock",
        str(spec.test_lock),
        *ARTICLE_FLAGS,
        "--out",
        str(spec.out_dir),
    ]
    if resume:
        comando.append("--resume")
    return comando


def child_env(
    spec: ArticleRunSpec, base: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Ambiente do processo filho, com a pasta de modelos isolada."""
    ambiente = dict(os.environ if base is None else base)
    for chave in ("MODELS_DIR", "DEEPFAKE_MODELS_DIR", "XFAKE_MODELS_DIR"):
        ambiente[chave] = str(spec.models_dir)
    ambiente.setdefault("PYTHONIOENCODING", "utf-8")
    ambiente.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return ambiente


def docker_command(spec: ArticleRunSpec, shell: str = "powershell") -> str:
    """O mesmo run pelo compose de benchmark, com a pasta de modelos isolada.

    ``docker compose run`` aplica o comando do serviço, que já traz as flags do
    artigo e ``--resume``; ``-e MODELS_DIR`` vence o ``DEEPFAKE_MODELS_DIR`` fixo
    do compose. As duas variáveis de pasta são lidas na interpolação do YAML,
    então vão no ambiente do shell.
    """
    dataset = f"data/datasets/{ARTICLE_DATASET}"
    saida = f"data/results/{spec.run_name}"
    modelos = f"/app/data/models/{spec.run_name}"
    compose = (
        "docker compose --env-file .env -f "
        "infrastructure/docker/compose/benchmark.nvidia.yml "
        f"run --rm -e MODELS_DIR={modelos} benchmark"
    )
    if shell == "bash":
        return (
            f"XFAKE_BENCHMARK_DATASET={dataset} XFAKE_BENCHMARK_OUT={saida} \\\n  "
            + compose
        )
    return (
        f'$env:XFAKE_BENCHMARK_DATASET="{dataset}"; '
        f'$env:XFAKE_BENCHMARK_OUT="{saida}"\n' + compose
    )


# ──────────────────────────── pré-condições ────────────────────────────


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _lock_check(spec: ArticleRunSpec) -> Dict[str, Any]:
    try:
        dados = json.loads(spec.test_lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dados if isinstance(dados, dict) else {}


def preflight(
    spec: ArticleRunSpec,
    *,
    gpu_available: Callable[[], bool],
    auth_enabled: bool,
    env: Optional[Mapping[str, str]] = None,
    resume: bool = False,
) -> List[Check]:
    """Checagens para iniciar (ou retomar) o run; todas precisam passar."""
    ambiente = os.environ if env is None else env
    checks: List[Check] = []
    habilitado = str(ambiente.get(ENABLE_FLAG, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    checks.append(
        Check(
            "Execução habilitada",
            habilitado,
            (
                f"{ENABLE_FLAG}=1 no ambiente do app"
                if habilitado
                else f"defina {ENABLE_FLAG}=1 no .env para permitir runs pela aba"
            ),
        )
    )
    checks.append(
        Check(
            "Login ativo",
            bool(auth_enabled),
            (
                "a interface pede usuário e senha"
                if auth_enabled
                else "defina XFAKE_GRADIO_USERNAME e XFAKE_GRADIO_PASSWORD: sem login, "
                "qualquer pessoa com acesso à página iniciaria um run de dias"
            ),
        )
    )
    try:
        gpu = bool(gpu_available())
    except Exception:  # noqa: BLE001 - sem diagnóstico, sem GPU
        gpu = False
    checks.append(
        Check(
            "GPU visível ao TensorFlow",
            gpu,
            (
                "o plano roda com os lotes de GPU do artigo"
                if gpu
                else "sem GPU o run cairia para CPU, com lotes diferentes dos do artigo; "
                "use o comando Docker abaixo numa máquina com GPU"
            ),
        )
    )
    selo = _lock_check(spec)
    dataset_ok = spec.dataset.is_file()
    tamanho_ok = dataset_ok and (
        not selo.get("dataset_size_bytes")
        or spec.dataset.stat().st_size == int(selo["dataset_size_bytes"])
    )
    checks.append(
        Check(
            "Dataset do artigo",
            bool(tamanho_ok),
            (
                f"{spec.dataset.name} com o tamanho declarado no selo"
                if tamanho_ok
                else (
                    f"tamanho de {spec.dataset.name} difere do selo"
                    if dataset_ok
                    else f"{spec.dataset} não encontrado"
                )
            ),
        )
    )
    contagens_ok = bool(selo) and all(
        int((selo.get("split_counts") or {}).get(k, -1)) == v
        for k, v in ARTICLE_SPLIT_COUNTS.items()
    )
    checks.append(
        Check(
            "Selo do teste",
            contagens_ok,
            (
                "partições 12.162/1.456/1.382, conferidas de novo pelo orquestrador"
                if contagens_ok
                else f"{spec.test_lock.name} ausente ou com outras partições"
            ),
        )
    )
    try:
        livre = shutil.disk_usage(spec.results_root.anchor or spec.results_root).free
        livre_gb = livre / 1e9
    except OSError:
        livre_gb = 0.0
    checks.append(
        Check(
            "Espaço em disco",
            livre_gb >= MIN_FREE_GB,
            f"{livre_gb:.0f} GB livres (mínimo {MIN_FREE_GB:.0f} GB)",
        )
    )
    ativo = active_run(spec.results_root)
    checks.append(
        Check(
            "Nenhum run em andamento",
            ativo is None,
            (
                "livre"
                if ativo is None
                else f"{ativo.get('run_name')} em andamento (pid {ativo.get('pid')})"
            ),
        )
    )
    if resume:
        marcado = (spec.out_dir / UI_MARKER).is_file()
        checks.append(
            Check(
                "Run criado pela aba",
                marcado,
                (
                    f"retoma {spec.run_name}"
                    if marcado
                    else "só runs iniciados pela aba podem ser retomados por ela"
                ),
            )
        )
    else:
        nova = not spec.out_dir.exists()
        checks.append(
            Check(
                "Pasta de saída nova",
                nova,
                str(spec.out_dir) if nova else f"{spec.out_dir} já existe",
            )
        )
    return checks


def can_start(checks: List[Check]) -> bool:
    return bool(checks) and all(c.ok for c in checks)


# ──────────────────────────── trava e processo ────────────────────────────


def pid_alive(pid: int) -> bool:
    """Processo ainda vivo, sem depender de psutil."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        codigo = ctypes.c_ulong()
        try:
            kernel32.GetExitCodeProcess(handle, ctypes.byref(codigo))
        finally:
            kernel32.CloseHandle(handle)
        return codigo.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_path(results_root: Path) -> Path:
    return Path(results_root) / LOCK_NAME


def active_run(results_root: Path) -> Optional[Dict[str, Any]]:
    """A trava, se o processo dela estiver vivo; trava de processo morto é removida."""
    caminho = _lock_path(results_root)
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(dados, dict) and pid_alive(int(dados.get("pid") or 0)):
        return dados
    try:
        caminho.unlink()
    except OSError:
        pass
    return None


def _write_json(path: Path, dados: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(dados), indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(path)


def _spawn(comando: List[str], cwd: Path, env: Mapping[str, str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    saida = log.open("ab")
    opcoes: Dict[str, Any] = {
        "cwd": str(cwd),
        "env": dict(env),
        "stdout": saida,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        opcoes["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        opcoes["start_new_session"] = True
    try:
        # Lista de argumentos sem shell, montada por build_command/consolidate.
        return subprocess.Popen(comando, **opcoes)  # noqa: S603  # nosec B603
    finally:
        saida.close()


def start(
    spec: ArticleRunSpec,
    checks: List[Check],
    resume: bool = False,
    command: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Inicia (ou retoma) o run; recusa se alguma checagem falhou.

    ``command`` existe para os testes trocarem o orquestrador por um processo
    curto; a aba e a API nunca o passam.
    """
    if not can_start(checks):
        falhas = ", ".join(c.name for c in checks if not c.ok)
        raise RuntimeError(f"Pré-condições não atendidas: {falhas}")
    comando = command or build_command(spec, resume=resume)
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    spec.models_dir.mkdir(parents=True, exist_ok=True)
    proc = _spawn(
        comando, spec.project_root, child_env(spec), spec.out_dir / ORCHESTRATOR_LOG
    )
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trava = {
        "pid": proc.pid,
        "run_name": spec.run_name,
        "out_dir": str(spec.out_dir),
        "models_dir": str(spec.models_dir),
        "started_at": agora,
        "resume": bool(resume),
        "command": comando,
    }
    _write_json(_lock_path(spec.results_root), trava)
    marcador = spec.out_dir / UI_MARKER
    historico: Dict[str, Any] = {}
    if marcador.is_file():
        try:
            historico = json.loads(marcador.read_text(encoding="utf-8"))
        except ValueError:
            historico = {}
    eventos = list(historico.get("events") or [])
    eventos.append({"event": "resume" if resume else "start", "at": agora})
    _write_json(
        marcador,
        {
            "run_name": spec.run_name,
            "protocol": "artigo do TCC (escopo oficial, recorte de 15 mil)",
            "command": comando,
            "models_dir": str(spec.models_dir),
            "events": eventos,
        },
    )
    logger.info("Benchmark do artigo iniciado: %s (pid %s)", spec.run_name, proc.pid)
    return trava


def stop(results_root: Path, grace_s: float = 10.0) -> Optional[Dict[str, Any]]:
    """Para o run ativo e toda a árvore de processos; devolve a trava parada."""
    trava = active_run(results_root)
    if trava is None:
        return None
    pid = int(trava["pid"])
    if os.name == "nt":
        # taskkill do sistema, sem shell; o pid vem da trava gravada pela aba.
        subprocess.run(  # noqa: S603,S607  # nosec B603 B607
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            limite = time.time() + grace_s
            while time.time() < limite and pid_alive(pid):
                time.sleep(0.5)
            if pid_alive(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    marcador = Path(trava["out_dir"]) / UI_MARKER
    try:
        dados = json.loads(marcador.read_text(encoding="utf-8"))
        dados.setdefault("events", []).append(
            {"event": "stop", "at": datetime.now(timezone.utc).isoformat()}
        )
        _write_json(marcador, dados)
    except (OSError, ValueError):
        pass
    try:
        _lock_path(results_root).unlink()
    except OSError:
        pass
    return trava


def consolidate(spec: ArticleRunSpec) -> int:
    """Consolida o run DENTRO dele (nunca em ``data/results/paper``); devolve o pid."""
    if not (spec.out_dir / UI_MARKER).is_file():
        raise RuntimeError("só runs iniciados pela aba são consolidados por ela")
    destino = spec.out_dir / "consolidated"
    comando = [
        spec.python,
        str(spec.project_root / "tools" / "reporting" / "consolidate_results.py"),
        str(spec.out_dir),
        "--prefer-last",
        "--out",
        str(destino),
        "--copy-to",
        str(destino / "figures"),
    ]
    proc = _spawn(
        comando, spec.project_root, child_env(spec), destino / "consolidate.log"
    )
    return proc.pid


# ──────────────────────────── progresso ────────────────────────────


@dataclass
class ModelProgress:
    model: str
    status: str
    elapsed_s: Optional[float] = None
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    eta_min: Optional[float] = None


@dataclass
class RunStatus:
    active: bool
    run_name: Optional[str]
    out_dir: Optional[str]
    summary_status: Optional[str]
    models: List[ModelProgress] = field(default_factory=list)
    current: Optional[ModelProgress] = None
    log_tail: List[str] = field(default_factory=list)
    lock: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tail(path: Path, lines: int = 40, max_bytes: int = 200_000) -> List[str]:
    """Últimas linhas de um log, lendo só o fim do arquivo."""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            tamanho = handle.tell()
            handle.seek(max(0, tamanho - max_bytes))
            dados = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return dados.splitlines()[-lines:]


def _current_model(out_dir: Path, concluidos: set) -> Optional[ModelProgress]:
    """Modelo em curso: tem ``run_fingerprint.json`` e ainda não tem resultado."""
    candidatos = [
        p
        for p in out_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith(("_", "."))
        and (p / "run_fingerprint.json").is_file()
        and not (p / "results.json").is_file()
        and p.name not in concluidos
    ]
    if not candidatos:
        return None
    pasta = max(candidatos, key=lambda p: p.stat().st_mtime)
    progresso = ModelProgress(model=pasta.name, status="running")
    for linha in reversed(tail(pasta / "run.log", lines=200)):
        achado = _TRAIN_LINE.search(linha)
        if achado:
            progresso.epoch = int(achado.group(1))
            progresso.epochs = int(achado.group(2))
            if achado.group(3):
                progresso.eta_min = float(achado.group(3))
            break
    return progresso


def run_status(results_root: Path, run_dir: Optional[Path] = None) -> RunStatus:
    """Estado do run ativo ou, sem run ativo, do run indicado."""
    trava = active_run(results_root)
    pasta = Path(trava["out_dir"]) if trava else (Path(run_dir) if run_dir else None)
    if pasta is None or not pasta.is_dir():
        return RunStatus(
            active=False, run_name=None, out_dir=None, summary_status=None, lock=trava
        )
    resumo: Dict[str, Any] = {}
    try:
        resumo = json.loads((pasta / "run_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        resumo = {}
    modelos = [
        ModelProgress(
            model=str(item.get("model")),
            status=str(item.get("status")),
            elapsed_s=item.get("elapsed_s"),
        )
        for item in (resumo.get("models") or [])
        if isinstance(item, dict)
    ]
    concluidos = {
        Path(str(item.get("output_dir") or "")).name
        for item in (resumo.get("models") or [])
        if isinstance(item, dict)
    }
    return RunStatus(
        active=trava is not None,
        run_name=pasta.name,
        out_dir=str(pasta),
        summary_status=resumo.get("status"),
        models=modelos,
        current=_current_model(pasta, concluidos) if trava else None,
        log_tail=tail(pasta / ORCHESTRATOR_LOG, lines=30),
        lock=trava,
    )
