#!/usr/bin/env python3
"""
benchmark_latency.py — Benchmark de Latência de Inferência por Arquitetura.

Mede o tempo de inferência para áudio sintético de duração configurável em cada
arquitetura registrada na factory, reportando:
  - Parâmetros totais
  - Memória estimada (MB)
  - Latência média ± desvio (ms)
  - P50 / P95 (ms)
  - Throughput (áudios/s)

Uso:
  python scripts/benchmark_latency.py                     # todas as arquiteturas
  python scripts/benchmark_latency.py --model Conformer   # apenas uma
  python scripts/benchmark_latency.py --duration 10       # duração do áudio (s)
  python scripts/benchmark_latency.py --n-runs 50         # repetições
  python scripts/benchmark_latency.py --output results/latency.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BenchmarkLatency")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 128
N_MELS = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(input_type: str, duration_s: float) -> np.ndarray:
    """Gera input sintético (batch=1) para o tipo de entrada da arquitetura."""
    n_samples = int(duration_s * SAMPLE_RATE)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1

    if input_type == "raw_audio":
        return audio.reshape(1, n_samples, 1)

    # spectrogram — log-mel
    import tensorflow as tf
    audio_t = tf.constant(audio[np.newaxis])
    n_freq = N_FFT // 2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        N_MELS, n_freq, SAMPLE_RATE, 0.0, SAMPLE_RATE / 2
    )
    stft = tf.signal.stft(audio_t, N_FFT, HOP, fft_length=N_FFT,
                          window_fn=tf.signal.hann_window, pad_end=True)
    mel = tf.math.log(tf.abs(tf.tensordot(tf.abs(stft), mel_w, axes=1)) + 1e-6)
    return tf.expand_dims(mel, axis=-1).numpy()  # (1, T, N_MELS, 1)


def _count_params(model) -> int:
    try:
        return int(model.count_params())
    except Exception:
        return 0


def _benchmark(model, dummy_input: np.ndarray, n_runs: int = 30, warmup: int = 5) -> dict:
    """Executa benchmark de latência, retorna estatísticas em ms."""
    for _ in range(warmup):
        try:
            model.predict(dummy_input, verbose=0)
        except Exception:
            break

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            model.predict(dummy_input, verbose=0)
        except Exception as e:
            logger.warning(f"Inferência falhou: {e}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        return {"error": "Todas as inferências falharam"}

    arr = np.array(latencies)
    return {
        "mean_ms":   round(float(np.mean(arr)), 1),
        "std_ms":    round(float(np.std(arr)), 1),
        "min_ms":    round(float(np.min(arr)), 1),
        "p50_ms":    round(float(np.percentile(arr, 50)), 1),
        "p95_ms":    round(float(np.percentile(arr, 95)), 1),
        "throughput": round(1000.0 / float(np.mean(arr)), 2),
        "n_runs":    len(latencies),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None,
                        help="Arquitetura específica (default: todas)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duração do áudio de teste em segundos (default: 10.0)")
    parser.add_argument("--n-runs", type=int, default=30,
                        help="Número de medições por modelo (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Salvar resultados em JSON (ex: results/latency.json)")
    args = parser.parse_args()

    from app.domain.models.architectures.factory import (
        architecture_factory_registry,
        create_model_by_name,
    )

    available = architecture_factory_registry.list_architectures()
    targets = [args.model] if args.model else available

    unknown = [m for m in targets if m not in available]
    if unknown:
        logger.error(f"Arquiteturas desconhecidas: {unknown}. Disponíveis: {available}")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("BENCHMARK DE LATÊNCIA DE INFERÊNCIA")
    logger.info("=" * 65)
    logger.info(f"  Arquiteturas : {len(targets)}")
    logger.info(f"  Áudio        : {args.duration}s @ {SAMPLE_RATE}Hz")
    logger.info(f"  Repetições   : {args.n_runs}")
    logger.info("")

    all_results: dict = {}

    for arch_name in targets:
        logger.info(f"--- {arch_name} ---")
        try:
            spec = architecture_factory_registry.get_architecture_info(arch_name)
            input_type = spec.input_requirements.get("input_type", "spectrogram") if spec else "spectrogram"

            dummy = _make_input(input_type, args.duration)
            input_shape = tuple(dummy.shape[1:])

            model = create_model_by_name(arch_name, input_shape=input_shape, num_classes=2)
            params = _count_params(model)
            mem_mb = round(params * 4 / (1024 ** 2), 1)

            logger.info(f"  input_type : {input_type}")
            logger.info(f"  input_shape: {input_shape}")
            logger.info(f"  parâmetros : {params:,}")
            logger.info(f"  memória    : {mem_mb} MB")

            bench = _benchmark(model, dummy, n_runs=args.n_runs)
            if "error" not in bench:
                logger.info(
                    f"  latência   : {bench['mean_ms']} ± {bench['std_ms']} ms  "
                    f"(p95={bench['p95_ms']} ms, {bench['throughput']} áudios/s)"
                )
            else:
                logger.error(f"  {bench['error']}")

            all_results[arch_name] = {
                "input_type": input_type,
                "input_shape": list(input_shape),
                "params": params,
                "memory_mb": mem_mb,
                "benchmark": bench,
            }
        except Exception as e:
            logger.error(f"  FALHA: {type(e).__name__}: {e}")
            all_results[arch_name] = {"error": str(e)}
        logger.info("")

    # ── Tabela resumo ────────────────────────────────────────────────
    logger.info("=" * 75)
    logger.info(f"  {'Arquitetura':<28} {'Parâmetros':>12} {'Mem(MB)':>9} {'Lat(ms)':>10} {'Throughput':>12}")
    logger.info("  " + "-" * 72)
    for arch, info in all_results.items():
        if "error" in info and "benchmark" not in info:
            logger.info(f"  {arch:<28} {'ERRO':>44}")
            continue
        bench = info.get("benchmark", {})
        if "error" in bench:
            logger.info(f"  {arch:<28} {info.get('params', 0):>12,} {info.get('memory_mb', 0):>8.1f}  {'FALHOU':>10}")
        else:
            logger.info(
                f"  {arch:<28} {info.get('params', 0):>12,} {info.get('memory_mb', 0):>8.1f}"
                f" {bench.get('mean_ms', 0):>9.1f}  {bench.get('throughput', 0):>9.1f} /s"
            )
    logger.info("=" * 75)
    logger.info(f"  * Latência medida para áudio de {args.duration}s")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResultados salvos em: {out}")


if __name__ == "__main__":
    main()
