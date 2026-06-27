"""Runtime performance utilities for CPU/GPU training and inference."""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePerformanceConfig:
    cpu_count: int
    intra_op_threads: int
    inter_op_threads: int
    max_parallel_workers: int
    enable_onednn: bool
    enable_xla: bool
    gpu_memory_growth: bool
    gpu_memory_limit_mb: Optional[int]
    cuda_malloc_async: bool
    enable_tf32: bool


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def get_runtime_performance_config() -> RuntimePerformanceConfig:
    cpu_count = max(1, os.cpu_count() or 1)
    default_intra = max(
        1, min(cpu_count, cpu_count - 1 if cpu_count > 2 else cpu_count)
    )
    default_inter = max(1, min(4, cpu_count // 2 or 1))
    return RuntimePerformanceConfig(
        cpu_count=cpu_count,
        intra_op_threads=_env_int("XFAKE_TF_INTRA_OP_THREADS", default_intra),
        inter_op_threads=_env_int("XFAKE_TF_INTER_OP_THREADS", default_inter),
        max_parallel_workers=_env_int("XFAKE_NUM_WORKERS", default_intra),
        enable_onednn=_env_bool("XFAKE_ENABLE_ONEDNN", True),
        enable_xla=_env_bool("XFAKE_ENABLE_XLA", True),
        gpu_memory_growth=_env_bool("XFAKE_GPU_MEMORY_GROWTH", True),
        gpu_memory_limit_mb=_env_optional_int("XFAKE_GPU_MEMORY_LIMIT_MB"),
        cuda_malloc_async=_env_bool("XFAKE_CUDA_MALLOC_ASYNC", True),
        # TF32 (Ampere+, CC>=8.0): acelera matmul/conv em float32 com leve perda
        # de precisão. Default OFF p/ reprodutibilidade do benchmark; ligue via
        # XFAKE_ENABLE_TF32=true para ganho nos modelos que rodam em float32
        # (WavLM/HuBERT/RawNet2/Ensemble, que optam por não usar mixed_float16).
        enable_tf32=_env_bool("XFAKE_ENABLE_TF32", False),
    )


def configure_runtime_environment() -> RuntimePerformanceConfig:
    """Set env vars that must exist before TensorFlow/BLAS initialize."""

    cfg = get_runtime_performance_config()
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.intra_op_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.intra_op_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg.intra_op_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cfg.intra_op_threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(cfg.intra_op_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(cfg.inter_op_threads))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1" if cfg.enable_onednn else "0")
    if cfg.cuda_malloc_async:
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    if cfg.enable_xla:
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=1")
    return cfg


def configure_tensorflow_runtime(tf_module: Any = None) -> Dict[str, Any]:
    """Apply TensorFlow runtime threading and JIT settings when possible."""

    cfg = configure_runtime_environment()
    try:
        tf = tf_module
        if tf is None:
            import tensorflow as tf  # noqa: PLC0415

        try:
            tf.config.threading.set_intra_op_parallelism_threads(cfg.intra_op_threads)
            tf.config.threading.set_inter_op_parallelism_threads(cfg.inter_op_threads)
        except RuntimeError as exc:
            logger.debug("TensorFlow threading already initialized: %s", exc)

        try:
            tf.config.optimizer.set_jit(bool(cfg.enable_xla))
        except Exception as exc:
            logger.debug("TensorFlow XLA JIT config skipped: %s", exc)

        # TF32 em GPUs Ampere+ (opt-in): acelera ops float32 sem o risco de
        # NaN do fp16. Off por default p/ não alterar a numérica do benchmark.
        try:
            tf.config.experimental.enable_tensor_float_32_execution(
                bool(cfg.enable_tf32)
            )
            if cfg.enable_tf32:
                logger.info("TF32 habilitado (Ampere+): ops float32 aceleradas.")
        except Exception as exc:
            logger.debug("TF32 config skipped: %s", exc)
    except ImportError:
        pass
    return asdict(cfg)


def configure_gpu_memory(tf_module: Any, gpus: list[Any]) -> Dict[str, Any]:
    """Configure memory growth or virtual memory limit for visible GPUs."""

    cfg = get_runtime_performance_config()
    result = {
        "memory_growth_applied": False,
        "memory_limit_mb": cfg.gpu_memory_limit_mb,
        "errors": [],
    }
    for gpu in gpus:
        try:
            if cfg.gpu_memory_limit_mb:
                tf_module.config.set_logical_device_configuration(
                    gpu,
                    [
                        tf_module.config.LogicalDeviceConfiguration(
                            memory_limit=cfg.gpu_memory_limit_mb
                        )
                    ],
                )
            elif cfg.gpu_memory_growth:
                tf_module.config.experimental.set_memory_growth(gpu, True)
                result["memory_growth_applied"] = True
        except RuntimeError as exc:
            msg = f"GPU memory config skipped; runtime already initialized: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)
        except Exception as exc:
            msg = f"GPU memory config failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)
    return result


def optimize_tf_dataset(
    dataset: Any,
    *,
    cache: bool = False,
    prefetch: bool = True,
    deterministic: bool = False,
) -> Any:
    """Apply safe tf.data options for throughput without duplicating RAM by default."""

    try:
        import tensorflow as tf  # noqa: PLC0415

        options = tf.data.Options()
        options.experimental_deterministic = deterministic
        try:
            options.threading.private_threadpool_size = (
                get_runtime_performance_config().max_parallel_workers
            )
        except Exception:
            pass
        dataset = dataset.with_options(options)
        if cache:
            dataset = dataset.cache()
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
    except Exception as exc:
        logger.debug("tf.data optimization skipped: %s", exc)
    return dataset


def release_training_memory() -> None:
    """Best-effort RAM/VRAM cleanup after training/evaluation phases."""

    try:
        import tensorflow as tf  # noqa: PLC0415

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def get_resource_snapshot() -> Dict[str, Any]:
    cfg = get_runtime_performance_config()
    snapshot: Dict[str, Any] = {"runtime": asdict(cfg)}
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        snapshot["ram"] = {
            "total_mb": round(vm.total / (1024 * 1024), 1),
            "available_mb": round(vm.available / (1024 * 1024), 1),
            "percent": vm.percent,
        }
    except Exception:
        snapshot["ram"] = None
    return snapshot


__all__ = [
    "RuntimePerformanceConfig",
    "configure_gpu_memory",
    "configure_runtime_environment",
    "configure_tensorflow_runtime",
    "get_resource_snapshot",
    "get_runtime_performance_config",
    "optimize_tf_dataset",
    "release_training_memory",
]
