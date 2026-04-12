#!/usr/bin/env python3
"""
benchmark_latency.py — Benchmark de Latencia de Inferencia por Arquitetura.

Mede o tempo de inferencia para audio de 10 segundos por arquitetura,
gerando a Tabela 7 do TCC (Analise de Complexidade Computacional).

Metricas medidas:
  - Latencia media (ms) para audio de 10s (N=50 repeticoes)
  - Parametros totais do modelo
  - Memoria estimada (MB)
  - Throughput (audios/s)

Uso:
  python scripts/benchmark_latency.py                    # todas as arquiteturas
  python scripts/benchmark_latency.py --model conformer  # apenas uma
  python scripts/benchmark_latency.py --duration 10      # duracao do audio (s)
  python scripts/benchmark_latency.py --n-runs 50        # repeticoes
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

# Mapeamento nome -> (modulo, funcao_create_model, input_shape_fn)
ARCHITECTURE_MAP = {
    "conformer": {
        "module": "app.domain.models.architectures.conformer",
        "input_type": "raw_audio",
    },
    "efficientnet_lstm": {
        "module": "app.domain.models.architectures.efficientnet_lstm",
        "input_type": "raw_audio",
    },
    "multiscale_cnn": {
        "module": "app.domain.models.architectures.multiscale_cnn",
        "input_type": "raw_audio",
    },
    "aasist": {
        "module": "app.domain.models.architectures.aasist",
        "input_type": "raw_audio",
    },
    "rawnet2": {
        "module": "app.domain.models.architectures.rawnet2",
        "input_type": "raw_audio",
    },
    "ensemble_adaptive": {
        "module": "app.domain.models.architectures.ensemble",
        "input_type": "raw_audio",
        "variant": "ensemble_adaptive",
    },
}


def generate_dummy_audio(duration_s: float = 10.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gera audio sintetico (ruido branco) para benchmark."""
    n_samples = int(duration_s * sample_rate)
    audio = np.random.randn(n_samples).astype(np.float32)
    # Normalizar
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    return audio


def load_model(arch_name: str, duration_s: float):
    """Carrega o modelo para a arquitetura especificada."""
    import importlib

    cfg = ARCHITECTURE_MAP[arch_name]
    n_samples = int(duration_s * SAMPLE_RATE)
    input_shape = (n_samples,)

    try:
        module = importlib.import_module(cfg["module"])
        variant = cfg.get("variant", arch_name)
        model = module.create_model(
            input_shape=input_shape,
            num_classes=1,
            architecture=variant,
        )
        return model, input_shape
    except Exception as e:
        logger.error(f"Falha ao carregar {arch_name}: {e}")
        return None, None


def count_params(model) -> int:
    """Conta parametros totais do modelo."""
    try:
        return model.count_params()
    except Exception:
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return 0


def estimate_memory_mb(model) -> float:
    """Estima memoria do modelo em MB (parametros * 4 bytes float32)."""
    params = count_params(model)
    return params * 4 / (1024 * 1024)


def benchmark_model(
    model,
    input_shape: tuple,
    duration_s: float = 10.0,
    n_runs: int = 50,
    warmup_runs: int = 5,
) -> dict:
    """
    Executa benchmark de latencia de inferencia.

    Args:
        model: modelo carregado
        input_shape: shape do input (n_samples,)
        duration_s: duracao do audio de teste
        n_runs: numero de medicoes
        warmup_runs: runs de aquecimento (nao contam)

    Returns:
        dict com estatisticas de latencia
    """
    audio = generate_dummy_audio(duration_s)
    batch = audio.reshape(1, -1)  # (1, n_samples)

    latencies_ms = []

    # Aquecimento
    for _ in range(warmup_runs):
        try:
            model.predict(batch, verbose=0)
        except Exception:
            try:
                import torch
                with torch.no_grad():
                    t = torch.FloatTensor(batch)
                    model(t)
            except Exception:
                pass

    # Medicao
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            model.predict(batch, verbose=0)
        except Exception:
            try:
                import torch
                with torch.no_grad():
                    t = torch.FloatTensor(batch)
                    model(t)
            except Exception as e:
                logger.warning(f"Inferencia falhou: {e}")
                continue
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    if not latencies_ms:
        return {"error": "Todas as inferencias falharam"}

    latencies = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(latencies)), 1),
        "std_ms": round(float(np.std(latencies)), 1),
        "min_ms": round(float(np.min(latencies)), 1),
        "max_ms": round(float(np.max(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "throughput_audios_per_s": round(1000.0 / float(np.mean(latencies)), 2),
        "n_runs": len(latencies_ms),
        "audio_duration_s": duration_s,
    }


def print_table7(results: dict):
    """Imprime a Tabela 7 do TCC (Analise de Complexidade Computacional)."""
    logger.info("\n" + "=" * 75)
    logger.info("TABELA 7 — Analise de Complexidade Computacional")
    logger.info("=" * 75)
    logger.info(
        f"  {'Arquitetura':<22} {'Parametros':>12} {'Memoria(MB)':>12} "
        f"{'Latencia(ms)':>14} {'Throughput':>12}"
    )
    logger.info("  " + "-" * 72)

    for arch, info in results.items():
        if "error" in info:
            logger.info(f"  {arch:<22} {'ERRO':>12}")
            continue
        bench = info.get("benchmark", {})
        logger.info(
            f"  {arch:<22} "
            f"{info.get('params', 0):>12,d} "
            f"{info.get('memory_mb', 0):>11.1f} "
            f"{bench.get('mean_ms', 0):>13.1f} "
            f"{bench.get('throughput_audios_per_s', 0):>10.1f} aud/s"
        )
    logger.info("=" * 75)
    logger.info("  * Latencia medida para audio de 10s, CPU (sem GPU)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de latencia de inferencia — Tabela 7 do TCC"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(ARCHITECTURE_MAP.keys()),
        help="Arquitetura especifica (default: todas)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duracao do audio de teste em segundos (default: 10.0)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=50,
        help="Numero de medicoes por modelo (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Salvar resultados em JSON (ex: results/latency.json)",
    )
    args = parser.parse_args()

    archs_to_test = [args.model] if args.model else list(ARCHITECTURE_MAP.keys())

    logger.info("=" * 60)
    logger.info("BENCHMARK DE LATENCIA — TCC UFSJ 2026")
    logger.info("=" * 60)
    logger.info(f"  Arquiteturas : {archs_to_test}")
    logger.info(f"  Audio        : {args.duration}s @ {SAMPLE_RATE}Hz")
    logger.info(f"  Repeticoes   : {args.n_runs}")

    all_results = {}

    for arch_name in archs_to_test:
        logger.info(f"\n--- {arch_name.upper()} ---")

        model, input_shape = load_model(arch_name, args.duration)
        if model is None:
            all_results[arch_name] = {"error": "Falha ao carregar modelo"}
            continue

        params = count_params(model)
        memory_mb = estimate_memory_mb(model)
        logger.info(f"  Parametros  : {params:,}")
        logger.info(f"  Memoria     : {memory_mb:.1f} MB")

        logger.info(f"  Iniciando {args.n_runs} medicoes...")
        bench = benchmark_model(
            model, input_shape,
            duration_s=args.duration,
            n_runs=args.n_runs,
        )

        logger.info(
            f"  Latencia    : {bench.get('mean_ms', '?')} ms "
            f"(+/- {bench.get('std_ms', '?')} ms)"
        )
        logger.info(
            f"  Throughput  : {bench.get('throughput_audios_per_s', '?')} audios/s"
        )

        all_results[arch_name] = {
            "params": params,
            "memory_mb": round(memory_mb, 1),
            "benchmark": bench,
        }

    # Tabela final
    print_table7(all_results)

    # Salvar JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResultados salvos em: {output_path}")

    # Aviso se latencia do Ensemble > 100ms (meta do TCC)
    if "ensemble_adaptive" in all_results:
        ens = all_results["ensemble_adaptive"].get("benchmark", {})
        lat = ens.get("mean_ms", 999)
        if lat <= 100:
            logger.info(
                f"\nMETA TCC ATINGIDA: Ensemble latencia = {lat}ms <= 100ms"
            )
        else:
            logger.warning(
                f"\nATENCAO: Ensemble latencia = {lat}ms > 100ms (meta TCC). "
                "Considere otimizacao ou GPU."
            )


if __name__ == "__main__":
    main()
