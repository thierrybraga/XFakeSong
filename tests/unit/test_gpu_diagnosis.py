"""Testes da revisão Linux/GPU: matriz de diagnóstico + probe read-only.

Cobre `app.core.gpu`:
- `_diagnose_gpu_situation` em todas as combinações relevantes (incl. o novo
  `driver_missing` p/ GPU no PCI sem driver em Linux bare-metal).
- Helpers de detecção PCI (`lspci`/`sysfs`) — tipos e robustez.
- `probe_gpu_status()` é READ-ONLY (não altera a política global de precisão).
"""

from __future__ import annotations

import pytest

import app.core.gpu as gpu


@pytest.fixture
def patch_env(monkeypatch):
    """Força plataforma/WSL/CUDA/DML no módulo gpu (revertido pelo monkeypatch)."""

    def _apply(system="Linux", wsl=False, tf_cuda=False, dml=False):
        monkeypatch.setattr(gpu.platform, "system", lambda: system)
        monkeypatch.setattr(gpu, "_is_wsl", lambda: wsl)
        monkeypatch.setattr(gpu, "_tf_built_with_cuda", lambda: tf_cuda)
        monkeypatch.setattr(gpu, "_detect_directml_plugin", lambda: dml)

    return _apply


# ───────────────────────── matriz de diagnóstico ─────────────────────────


def test_diag_no_gpu(patch_env):
    patch_env(system="Linux")
    d = gpu._diagnose_gpu_situation([], 0, pci_present=False)
    assert d["diagnosis"] == "no_gpu"
    assert d["severity"] == "info"
    assert d["hints"] == []


def test_diag_ok(patch_env):
    patch_env(system="Linux", tf_cuda=True)
    d = gpu._diagnose_gpu_situation([{"name": "RTX 4090"}], 1, pci_present=True)
    assert d["diagnosis"] == "ok"


def test_diag_driver_missing_on_bare_linux(patch_env):
    """GPU no PCI, sem driver, Linux nativo → orienta instalar o driver."""
    patch_env(system="Linux", wsl=False)
    d = gpu._diagnose_gpu_situation([], 0, pci_present=True)
    assert d["diagnosis"] == "driver_missing"
    assert d["severity"] == "warning"
    assert any(
        ("nvidia-driver" in h) or ("ubuntu-drivers" in h) for h in d["hints"]
    )


def test_diag_pci_present_on_wsl_is_not_driver_missing(patch_env):
    """Em WSL o driver vem do host Windows — pci-only não vira driver_missing."""
    patch_env(system="Linux", wsl=True)
    d = gpu._diagnose_gpu_situation([], 0, pci_present=True)
    assert d["diagnosis"] == "no_gpu"


def test_diag_wsl_no_cuda(patch_env):
    patch_env(system="Linux", wsl=True, tf_cuda=False)
    d = gpu._diagnose_gpu_situation([{"name": "RTX 3060"}], 0, pci_present=True)
    assert d["diagnosis"] == "wsl_no_cuda"
    assert any("and-cuda" in h for h in d["hints"])


def test_diag_windows_native_no_cuda(patch_env):
    patch_env(system="Windows", wsl=False, dml=False)
    d = gpu._diagnose_gpu_situation([{"name": "RTX 3060"}], 0, pci_present=True)
    assert d["diagnosis"] == "windows_native_no_gpu_support"
    assert d["severity"] == "warning"


def test_diag_windows_dml_installed(patch_env):
    patch_env(system="Windows", wsl=False, dml=True)
    d = gpu._diagnose_gpu_situation([{"name": "RTX 3060"}], 0, pci_present=True)
    assert d["diagnosis"] == "windows_dml_installed_but_invisible"


# ───────────────────────── helpers de detecção PCI ─────────────────────────


def test_pci_helpers_return_types():
    # Não importa o ambiente — apenas não podem lançar e retornam o tipo certo.
    assert isinstance(gpu._detect_nvidia_via_lspci(), list)
    assert isinstance(gpu._detect_nvidia_via_sysfs(), int)
    assert isinstance(gpu._detect_nvidia_pci_only(), bool)


def test_sysfs_is_zero_off_linux(monkeypatch):
    monkeypatch.setattr(gpu.platform, "system", lambda: "Windows")
    assert gpu._detect_nvidia_via_sysfs() == 0


# ───────────────────────── probe read-only ─────────────────────────


def test_probe_does_not_mutate_global_policy():
    tf = pytest.importorskip("tensorflow")
    before = tf.keras.mixed_precision.global_policy().name
    st = gpu.probe_gpu_status()
    after = tf.keras.mixed_precision.global_policy().name
    assert before == after  # probe NÃO pode ligar mixed_float16 globalmente
    for key in (
        "system",
        "is_wsl",
        "tf_available",
        "tf_built_with_cuda",
        "nvidia_hardware",
        "nvidia_pci_present",
        "tf_gpus",
        "diagnosis",
        "summary",
    ):
        assert key in st
    assert isinstance(st["summary"], str) and st["summary"]


def test_probe_uses_cache_when_setup_done(monkeypatch):
    fake = {
        "platform": {
            "system": "Linux",
            "release": "6.x",
            "python": "3.13",
            "is_wsl": False,
        },
        "tf_available": True,
        "tf_built_with_cuda": True,
        "directml_installed": False,
        "nvidia_hardware": [{"name": "RTX 4090"}],
        "nvidia_pci_present": True,
        "gpus_detected": [
            {"name": "RTX 4090", "compute_capability": "8.9", "tensor_core": True}
        ],
        "mixed_precision_enabled": True,
        "diagnosis": {
            "diagnosis": "ok",
            "message": "",
            "hints": [],
            "severity": "info",
        },
    }
    monkeypatch.setattr(gpu, "_setup_done", True)
    monkeypatch.setattr(gpu, "_setup_result", fake)
    st = gpu.probe_gpu_status()
    assert st["tf_gpus"] == fake["gpus_detected"]
    assert st["nvidia_pci_present"] is True
    assert st["tf_built_with_cuda"] is True
