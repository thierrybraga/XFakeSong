#!/usr/bin/env python3
"""Gerador dos notebooks de estudo/reprodução do XFakeSong (TCC).

Reconstrói, de forma reprodutível, os notebooks ATIVOS em `notebooks/`
(`00_index`, `features/`, `models/`, `pipeline/`) com células limpas:

- markdown sem indentação espúria (que vira bloco de código no Jupyter) e em
  UTF-8 correto;
- código que de fato COMPILA e usa a API real do projeto
  (`benchmarks.data.BenchmarkData`, `create_model_by_name`, `BenchmarkConfig`/
  `run_benchmark`, `TrainingService`, `ModelLoader`+`Predictor`);
- `source` gravado como lista de linhas (formato nbformat correto).

Cada célula de código é validada com `compile()` ANTES de ser gravada — se o
gerador roda sem erro, todo o código dos notebooks é sintaticamente válido.

NÃO altera `notebooks/legacy/` (preservado).

Uso:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bootstrap comum: localiza a raiz do projeto e a coloca no sys.path.
BOOTSTRAP = """\
from pathlib import Path
import sys

ROOT = Path.cwd()
while not (ROOT / "app").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Projeto:", ROOT)
"""


# ───────────────────────── helpers de célula ─────────────────────────

def _src_lines(text: str) -> list:
    """Texto → lista de linhas (com \\n, exceto a última) p/ nbformat."""
    text = textwrap.dedent(text).strip("\n")
    return text.splitlines(keepends=True) or [""]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src_lines(text)}


def code(text: str) -> dict:
    body = textwrap.dedent(text).strip("\n")
    compile(body, "<notebook-cell>", "exec")  # valida sintaxe
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src_lines(text),
    }


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(rel_path: str, cells: list) -> None:
    path = NB / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ok  {rel_path} ({len(cells)} células)")


# ───────────────────────── catálogo de modelos ─────────────────────────
# (num, slug_arquivo, nome_no_registry, família, porquê estudar)
MODELS = [
    (1, "wavlm", "WavLM", "Foundation/SSL (raw-audio)",
     "Backbone auto-supervisionado; cabeça de classificação sobre features SSL."),
    (2, "hubert", "HuBERT", "Foundation/SSL (raw-audio)",
     "Alternativa SSL ao WavLM; mesmo padrão de fine-tuning."),
    (3, "rawnet2", "RawNet2", "Forma de onda (SincNet)",
     "Processa PCM 1D via Sinc-Convolutions; baseline end-to-end."),
    (4, "sonic_sleuth", "Sonic Sleuth", "Espectrograma (features in-model)",
     "Extrai LFCC/MFCC/CQT dentro do grafo (tf.signal)."),
    (5, "aasist", "AASIST", "Forma de onda + grafo",
     "Graph attention espectro-temporal; SOTA em datasets controlados."),
    (6, "rawgat_st", "RawGAT-ST", "Forma de onda + grafo duplo",
     "SincConv + grafos espectral/temporal com fusão."),
    (7, "conformer", "Conformer", "Espectrograma (conv+atenção)",
     "Convolução + self-attention para contexto local e global."),
    (8, "hybrid_cnn_transformer", "Hybrid CNN-Transformer",
     "Espectrograma (híbrido)",
     "CNN para features locais + Transformer para dependências longas."),
    (9, "spectrogram_transformer", "SpectrogramTransformer",
     "Espectrograma (AST)",
     "Audio Spectrogram Transformer: patches do espectrograma."),
    (10, "efficientnet_lstm", "EfficientNet-LSTM", "Espectrograma + temporal",
     "EfficientNet (mel→224×224) + LSTM com atenção temporal."),
    (11, "multiscale_cnn", "MultiscaleCNN", "Espectrograma (Res2Net)",
     "Convoluções multi-escala (Res2Net); baseline neural principal."),
    (12, "ensemble", "Ensemble", "Espectrograma (multi-ramo)",
     "5 ramos convolucionais com fusão adaptativa; leve e rápido."),
    (13, "svm", "SVM", "Clássico (tabular)",
     "Baseline leve sobre features acústicas agregadas; ideal em CPU."),
    (14, "random_forest", "RandomForest", "Clássico (tabular)",
     "Ensemble de árvores sobre features agregadas; rápido e robusto."),
]
CLASSICAL = {"SVM", "RandomForest"}


def _input_type(display_name: str) -> str:
    if display_name in CLASSICAL:
        return "tabular"
    try:
        from app.domain.models.architectures.factory import get_architecture_info

        info = get_architecture_info(display_name)
        if info:
            return info.input_requirements.get("input_type", "?")
    except Exception as exc:
        print(f"  aviso: contrato não lido para {display_name}: {exc}")
    return "?"


# ───────────────────────── 00_index ─────────────────────────

def build_index():
    write_nb("00_index.ipynb", [
        md("""
        # XFakeSong — Índice dos Notebooks

        Estudos organizados em quatro blocos:

        - `pipeline/`: benchmark completo, treino e inferência.
        - `features/`: extração e análise de features.
        - `models/`: um notebook por arquitetura (14 modelos).
        - `legacy/`: notebooks antigos preservados.

        **Ordem sugerida:** features → pipeline → models.
        """),
        code(BOOTSTRAP),
        code("""
        # Lista todos os notebooks do projeto.
        for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
            print(path.relative_to(ROOT))
        """),
    ])


# ───────────────────────── features ─────────────────────────

def build_features():
    write_nb("features/01_feature_extraction_study.ipynb", [
        md("""
        # Estudo de Extração de Features

        Demonstra o **front-end real** usado pelo sistema na inferência
        (`app.domain.services.detection.audio_preprocessing.prepare_audio_for_model`,
        baseado em `tf.signal`, in-graph):

        - **LFCC** (default desde a melhoria P0 — supera o mel em anti-spoofing);
        - **log-mel**;
        - **raw-audio** (forma de onda PCM 1D);
        - métricas clássicas de estudo: **MFCC**, **centroide espectral**,
          **bandwidth**, **ZCR** e energia RMS.

        Configuração: `sample_rate=16 kHz`, `n_fft=512`, `hop_length=128`,
        `n_mels = n_lfcc = 80`.
        """),
        code(BOOTSTRAP),
        md("## 1. Gerar um sinal sintético de 1 segundo"),
        code("""
        import numpy as np

        rng = np.random.default_rng(0)
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        # Soma de senóides + ruído leve (proxy de voz para demonstração).
        wav = (0.6 * np.sin(2 * np.pi * 220 * t)
               + 0.3 * np.sin(2 * np.pi * 440 * t)
               + 0.05 * rng.standard_normal(16000)).astype("float32")
        print("waveform:", wav.shape, wav.dtype)
        """),
        md("## 2. Extrair os três front-ends reais"),
        code("""
        from app.domain.services.detection.audio_preprocessing import (
            prepare_audio_for_model,
        )

        lfcc = prepare_audio_for_model(
            wav, input_type="spectrogram", input_shape=(100, 80),
            sample_rate=16000, feature_frontend="lfcc",
        )
        logmel = prepare_audio_for_model(
            wav, input_type="spectrogram", input_shape=(100, 80),
            sample_rate=16000, feature_frontend="logmel",
        )
        raw = prepare_audio_for_model(
            wav, input_type="raw_audio", input_shape=(16000, 1),
            sample_rate=16000,
        )
        import numpy as np
        print("LFCC   :", np.asarray(lfcc).shape)
        print("log-mel:", np.asarray(logmel).shape)
        print("raw    :", np.asarray(raw).shape)
        """),
        md("## 3. Features clássicas para estudo"),
        code("""
        import librosa
        import pandas as pd

        mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(y=wav, sr=16000)
        bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=16000)
        zcr = librosa.feature.zero_crossing_rate(wav)
        rms = librosa.feature.rms(y=wav)

        feature_summary = pd.DataFrame([{
            "mfcc_mean": float(mfcc.mean()),
            "mfcc_std": float(mfcc.std()),
            "centroid_mean_hz": float(centroid.mean()),
            "bandwidth_mean_hz": float(bandwidth.mean()),
            "zcr_mean": float(zcr.mean()),
            "rms_mean": float(rms.mean()),
        }])
        feature_summary
        """),
        md("## 4. Visualizar"),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(13, 6.2))
        ax = ax.ravel()
        ax[0].plot(wav[:1000], color="#3b82f6", lw=0.8)
        ax[0].set_title("Forma de onda (1000 amostras)")
        ax[1].imshow(np.asarray(lfcc).T, aspect="auto", origin="lower", cmap="magma")
        ax[1].set_title("LFCC (80 × T)")
        ax[2].imshow(np.asarray(logmel).T, aspect="auto", origin="lower", cmap="magma")
        ax[2].set_title("log-mel (80 × T)")
        ax[3].imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
        ax[3].set_title("MFCC")
        ax[4].plot(centroid.ravel(), label="centroid")
        ax[4].plot(bandwidth.ravel(), label="bandwidth")
        ax[4].set_title("Centroide e bandwidth")
        ax[4].legend()
        ax[5].plot(zcr.ravel(), label="ZCR")
        ax[5].plot(rms.ravel(), label="RMS")
        ax[5].set_title("ZCR e RMS")
        ax[5].legend()
        fig.tight_layout()
        plt.show()
        """),
        md("""
        ## Leituras

        - `docs/04_FEATURES.md` — features e extratores.
        - `docs/09_INFERENCIA.md` — como o `input_contract` garante paridade
          treino↔inferência (o mesmo front-end usado aqui é reproduzido na
          detecção).
        """),
    ])


# ───────────────────────── models/* ─────────────────────────

def build_models():
    for num, slug, name, family, why in MODELS:
        itype = _input_type(name)
        is_classical = name in CLASSICAL
        # célula de inspeção difere para clássico × neural
        if is_classical:
            inspect = f"""
            ARCHITECTURE = {name!r}

            # Modelos clássicos (sklearn) são treinados/avaliados pelo harness de
            # benchmark (caminho clássico), não pelo factory Keras. Rodamos um
            # benchmark rápido em dados sintéticos só para validar o fluxo.
            from benchmarks import BenchmarkConfig, run_benchmark

            cfg = BenchmarkConfig.quick(
                architectures=[ARCHITECTURE],
                synthetic_n=120,
                snr_levels_db=[20],
                output_dir=str(ROOT / "results" / "notebook_smoke" / ARCHITECTURE.lower()),
            )
            results = run_benchmark(cfg)
            print("clean:", results["architectures"][ARCHITECTURE]["clean"])
            """
        else:
            inspect = f"""
            ARCHITECTURE = {name!r}

            # Modelos neurais são instanciados pelo factory (mesmo caminho do
            # benchmark/treino). summary() valida shapes sem treino pesado.
            from app.domain.models.architectures.factory import create_model_by_name

            input_shape = tuple(prepared.X.shape[1:])
            model = create_model_by_name(ARCHITECTURE, input_shape=input_shape, num_classes=1)
            model.summary()
            print("Parâmetros:", model.count_params())
            """
        write_nb(f"models/{num:02d}_{slug}.ipynb", [
            md(f"""
            # Estudo do modelo: {name}

            - **Família:** {family}
            - **Tipo de entrada (registry):** `{itype}`
            - **Por quê:** {why}
            """),
            md("""
            ## Objetivos

            1. Mostrar o **contrato de entrada** que o benchmark prepara para esta
               arquitetura.
            2. Gerar dados sintéticos mínimos para validar shape e fluxo.
            3. Instanciar/rodar o modelo pelo **mesmo caminho** do benchmark.

            Para treino real e métricas, use `notebooks/pipeline/`.
            """),
            code(BOOTSTRAP),
            code(f"""
            ARCHITECTURE = {name!r}

            from benchmarks.data import BenchmarkData

            data = BenchmarkData.synthetic(n=48, shape=(64, 40), seed=7)
            prepared = data.prepare_for_architecture(ARCHITECTURE)
            print("Arquitetura     :", ARCHITECTURE)
            print("Shape original  :", data.X.shape)
            print("Shape preparado :", prepared.X.shape)
            print("Metadados       :", prepared.metadata)
            """),
            md("## Inspeção do modelo"),
            code(inspect),
            md("""
            ## Como ler este notebook

            - `docs/08_ARQUITETURAS.md` — descrição de todas as arquiteturas.
            - `docs/10_TREINAMENTO.md` — hiperparâmetros e estratégia de treino.
            - `docs/15_BENCHMARK.md` — execução completa, métricas e gráficos.

            No relatório, compare sempre: acurácia, EER, AUC-ROC, min-tDCF,
            latência, tamanho do modelo, matriz de confusão e distribuição de scores.
            """),
            md(f"""
            ## Hiperparâmetros e análise esperada

            - **Treino rápido:** use `epochs=2` apenas para validar fluxo.
            - **Treino para TCC:** use o preset do benchmark completo em
              `notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb`.
            - **Entrada preparada:** confira a célula de `BenchmarkData` acima;
              ela mostra o shape real usado no harness.
            - **Arquivos esperados no relatório:** `architectures/{slug}/metrics.json`,
              `confusion_matrix.png`, `roc.png`, `score_distribution.png` e,
              quando houver histórico, `convergence.png`.
            """),
            code(f"""
            import json

            metrics_path = ROOT / "results" / "tcc_full_20k" / "architectures" / {slug!r} / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                print(json.dumps(metrics.get("clean", metrics), indent=2, ensure_ascii=False)[:2000])
            else:
                print("Sem métricas salvas ainda:", metrics_path)
                print("Rode o benchmark completo ou ajuste o caminho de results.")
            """),
        ])


# ───────────────────────── pipeline/* ─────────────────────────

def build_pipeline():
    # 01 — benchmark
    write_nb("pipeline/01_benchmark_tcc_full_pipeline.ipynb", [
        md("""
        # Pipeline de Benchmark do TCC

        Roda o sistema de benchmark e documenta o automador completo do TCC:
        download/preparação do dataset, split estratificado, treino, inferência,
        tabelas LaTeX, figuras PNG, `dataset.md` e `tcc_report.md`.
        """),
        code(BOOTSTRAP),
        md("## 1. Benchmark rápido (sintético) — valida o harness em segundos"),
        code("""
        from benchmarks import BenchmarkConfig, run_benchmark

        cfg = BenchmarkConfig.quick(
            architectures=["MultiscaleCNN", "SVM"],
            synthetic_n=160,
            snr_levels_db=[20],
            output_dir=str(ROOT / "results" / "notebook_benchmark"),
        )
        results = run_benchmark(cfg)

        for name, r in results["architectures"].items():
            if r.get("status") == "ok":
                c = r["clean"]
                print(f"{name:<16} acc={c['accuracy']*100:5.1f}%  "
                      f"AUC={c.get('auc_roc', float('nan')):.3f}  "
                      f"conv={r.get('converged')}")
            else:
                print(f"{name:<16} ERRO: {r.get('error')}")
        """),
        code("""
        # Artefatos gerados (tabelas .tex, figuras .png, CSV/JSON).
        out = ROOT / "results" / "notebook_benchmark"
        for f in sorted(out.rglob("*")):
            if f.is_file():
                print(f.relative_to(out))
        """),
        md("""
        ## 2. Execução completa do TCC

        A célula abaixo é executável, mas fica desligada por padrão para evitar
        downloads e treinos longos sem confirmação explícita.
        """),
        code("""
        import subprocess

        RUN_FULL_PIPELINE = False
        OUTPUT_DIR = ROOT / "results" / "tcc_full_20k"
        DATASET_NPZ = ROOT / "app" / "datasets" / "benchmark_audio_raw_20k.npz"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_tcc_pipeline.py"),
            "--tcc-full-dataset",
            "--out", str(OUTPUT_DIR),
            "--npz", str(DATASET_NPZ),
        ]
        print(" ".join(cmd))

        if RUN_FULL_PIPELINE:
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print("Execução completa desativada. Defina RUN_FULL_PIPELINE = True para rodar.")
        """),
        md("""
        Comando equivalente no terminal:

        ```bash
        python scripts/run_tcc_pipeline.py --tcc-full-dataset \\
            --out results/tcc_full_20k \\
            --npz app/datasets/benchmark_audio_raw_20k.npz
        ```

        Ou o benchmark direto sobre um `.npz` já exportado:

        ```bash
        python scripts/run_benchmark.py --full --dataset app/datasets/brspeech_df.npz
        ```

        Veja `docs/15_BENCHMARK.md` para o mapeamento saída → tabela/figura do TCC.
        """),
        md("## 3. Validar e ler artefatos do relatório"),
        code("""
        import json
        import pandas as pd

        report_dir = ROOT / "results" / "tcc_full_20k"
        required = [
            "dataset.md",
            "dataset_manifest.json",
            "results.json",
            "results.csv",
            "tcc_report.md",
            "figures/roc.png",
            "figures/confusion_matrices.png",
            "figures/score_distributions.png",
        ]
        missing = [p for p in required if not (report_dir / p).exists()]
        print("Diretório:", report_dir)
        print("Ausentes:", missing or "nenhum")

        if (report_dir / "results.csv").exists():
            display(pd.read_csv(report_dir / "results.csv"))
        if (report_dir / "results.json").exists():
            data = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
            print("Arquiteturas:", list(data.get("architectures", {}).keys()))
        """),
    ])

    # 02 — treino
    write_nb("pipeline/02_training_model.ipynb", [
        md("""
        # Treino de um Modelo (pipeline real)

        Treina pelo `TrainingService` — o MESMO caminho usado pela API e pelo
        benchmark — a partir de um `.npz` (`X_train`/`y_train`). Salva o modelo
        como artefato de inferência (sem estado do otimizador) com o
        `input_contract` que garante paridade treino↔inferência.
        """),
        code(BOOTSTRAP),
        md("## 1. Dataset sintético `.npz` (espectrograma pequeno)"),
        code("""
        import tempfile
        import numpy as np

        rng = np.random.default_rng(0)
        n, T, F = 300, 32, 16
        X = rng.standard_normal((n, T, F)).astype("float32")
        y = rng.integers(0, 2, n).astype("int64")
        X[y == 1] += 0.6  # separa as classes → treinável em poucas épocas

        workdir = Path(tempfile.mkdtemp(prefix="nb_train_"))
        npz = workdir / "dataset.npz"
        np.savez(npz, X_train=X, y_train=y)
        print("dataset:", npz)
        """),
        md("## 2. Treinar via TrainingService"),
        code("""
        from app.core.interfaces.base import ProcessingStatus
        from app.domain.services.training_service import TrainingService

        models_dir = workdir / "models"
        svc = TrainingService(models_dir=str(models_dir))
        res = svc.train_model(
            architecture="MultiscaleCNN",
            dataset_path=str(npz),
            config={"epochs": 2, "batch_size": 32, "model_name": "nb_demo"},
        )
        assert res.status == ProcessingStatus.SUCCESS, res.errors
        md_ = res.data
        print("acurácia:", round(md_.accuracy, 4))
        print("artefatos:", sorted(p.name for p in models_dir.glob("nb_demo*")))
        """),
        md("""
        ## Notas

        - O `.keras` é salvo **sem o estado do otimizador** (~3× menor, load mais
          rápido, saída idêntica) — ver `trainer.save_inference_keras`.
        - O `nb_demo_config.json` carrega o `input_contract` (temperatura/EER/OOD
          calibrados) lido na inferência. Ver `notebooks/pipeline/03_inference.ipynb`.
        """),
    ])

    # 03 — inferência
    write_nb("pipeline/03_inference.ipynb", [
        md("""
        # Inferência (ModelLoader + Predictor)

        Treina um modelo minúsculo, salva, recarrega pelo `ModelLoader` e prevê
        com o `Predictor` — o caminho real de detecção, incluindo a leitura do
        `input_contract` (temperatura/EER) e o fallback ONNX→TF.
        """),
        code(BOOTSTRAP),
        md("## 1. Treinar + salvar um modelo de demonstração"),
        code("""
        import tempfile
        import numpy as np
        from app.core.interfaces.base import ProcessingStatus
        from app.domain.services.training_service import TrainingService

        rng = np.random.default_rng(1)
        n, T, F = 240, 32, 16
        X = rng.standard_normal((n, T, F)).astype("float32")
        y = rng.integers(0, 2, n).astype("int64")
        X[y == 1] += 0.6

        workdir = Path(tempfile.mkdtemp(prefix="nb_infer_"))
        npz = workdir / "ds.npz"
        np.savez(npz, X_train=X, y_train=y)
        models_dir = workdir / "models"
        res = TrainingService(models_dir=str(models_dir)).train_model(
            architecture="MultiscaleCNN", dataset_path=str(npz),
            config={"epochs": 2, "batch_size": 32, "model_name": "nb_infer"},
        )
        assert res.status == ProcessingStatus.SUCCESS, res.errors
        print("treinado e salvo em", models_dir)
        """),
        md("## 2. Carregar e prever"),
        code("""
        from app.domain.services.detection.model_loader import ModelLoader
        from app.domain.services.detection.predictor import Predictor

        loader = ModelLoader(models_dir=str(models_dir))
        loader.load_available_models()
        mi = loader.get_model("nb_infer")
        print("modelo:", mi.architecture, "| input_shape:", mi.input_shape)
        print("temperatura calibrada:", mi.temperature,
              "| eer_threshold:", mi.eer_threshold)

        sample = X[:1][0]  # uma amostra (T, F)
        out = Predictor().predict(mi, sample)
        print("resultado:", out.data)
        """),
        md("""
        ## Notas

        - `ModelLoader` lê o `input_contract` do `_config.json` (temperatura/EER/OOD)
          e o `Predictor` os aplica automaticamente.
        - Se houver um `nb_infer.onnx` ao lado e `onnxruntime` instalado, a
          inferência usa ONNX (mais rápida em CPU) com fallback para o TF.
        """),
    ])


def main() -> int:
    print("Gerando notebooks (legacy/ é preservado)...")
    build_index()
    build_features()
    build_models()
    build_pipeline()
    # Validação final: recarrega tudo e recompila cada célula de código.
    total_code = 0
    for p in sorted(NB.rglob("*.ipynb")):
        if "legacy" in p.parts:
            continue
        nbj = json.loads(p.read_text(encoding="utf-8"))
        for c in nbj["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(p), "exec")
                total_code += 1
    print(f"OK — todas as células de código compilam ({total_code} células).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
