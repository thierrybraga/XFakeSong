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

O gerador é determinístico e NÃO depende de TensorFlow no build: o `input_type`
de cada modelo vem do catálogo `MODELS` (cross-check opcional com o registry
quando o factory está importável).

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

# Diretório canônico do relatório completo do TCC (fonte única do caminho).
TCC_RESULTS = "tcc_full_20k"

# Bootstrap comum + compatível com Google Colab / Kaggle:
# - localiza a raiz do projeto (pasta com `app/`) e a coloca no sys.path;
# - em ambiente limpo (sem o projeto no disco) clona o repositório;
# - instala apenas as dependências de áudio que faltarem (TF/NumPy/sklearn/pandas
#   já vêm no Colab — reinstalar tudo causaria conflitos).
# Usa `subprocess` (não `!pip`) para continuar sendo Python válido/compilável.
BOOTSTRAP = """\
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_REPO_URL = "https://github.com/thierrybraga/XFakeSong.git"


def _project_root() -> Path:
    p = Path.cwd()
    while not (p / "app").exists() and p.parent != p:
        p = p.parent
    return p


# Colab/Kaggle limpos: clona o repositório se o projeto não estiver no disco.
if not (_project_root() / "app").exists():
    if not Path("XFakeSong").exists():
        print("Clonando XFakeSong...")
        subprocess.run(["git", "clone", "--depth", "1", _REPO_URL], check=True)
    os.chdir("XFakeSong")

_DEPS = {"librosa": "librosa>=0.10", "soundfile": "soundfile>=0.12",
         "pywt": "PyWavelets>=1.4"}
_missing = [s for m, s in _DEPS.items() if importlib.util.find_spec(m) is None]
if _missing:
    print("Instalando dependências de áudio:", *_missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing],
                   check=True)

ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Projeto:", ROOT, "| Colab:", "google.colab" in sys.modules)
"""

# Captura de ambiente/versões — reprodutibilidade do estudo (TCC).
SESSION_INFO = """\
import platform

print("Python     :", platform.python_version())
print("Plataforma :", platform.platform())
for _mod in ("numpy", "pandas", "librosa", "sklearn", "tensorflow"):
    try:
        _m = __import__(_mod)
        print(f"{_mod:11s}:", getattr(_m, "__version__", "?"))
    except Exception:  # pragma: no cover
        print(f"{_mod:11s}: (ausente)")
try:
    import tensorflow as _tf
    print("GPUs TF    :", len(_tf.config.list_physical_devices("GPU")))
except Exception:  # pragma: no cover
    pass
"""

# Determinismo: semente única (numpy + TF) — use no início dos notebooks que
# treinam, para métricas reproduzíveis.
SEED_ALL = """\
import numpy as np

try:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
except Exception:  # pragma: no cover
    np.random.seed(0)
"""

# Avaliação rápida por modelo (gated): treina+avalia em dados sintéticos via o
# MESMO harness de benchmark do TCC (accuracy/AUC/EER reais). Desligado por
# padrão para manter a execução leve. `ARCHITECTURE` vem da célula anterior.
EVAL_CELL = """\
import os

# Treina + avalia este modelo em dados sintéticos (rápido) pelo MESMO harness do
# benchmark do TCC. LIGADO por padrão para o estudo ser funcional de ponta a
# ponta; defina XFAKE_RUN_EVAL=0 para pular (usado no CI de build).
# Em Colab, selecione um runtime com GPU (Ambiente de execução → Alterar o tipo
# de hardware → GPU) para acelerar.
RUN_EVAL = os.environ.get("XFAKE_RUN_EVAL", "1") == "1"

if RUN_EVAL:
    from benchmarks import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig.quick(
        architectures=[ARCHITECTURE],
        synthetic_n=160,
        snr_levels_db=[20],
        output_dir=str(ROOT / "results" / "notebook_eval" / ARCHITECTURE.lower()),
    )
    r = run_benchmark(cfg)["architectures"][ARCHITECTURE]
    if r.get("status") == "ok":
        c = r["clean"]
        print(f"accuracy = {c['accuracy'] * 100:5.1f}%")
        print(f"AUC-ROC  = {c.get('auc_roc', float('nan')):.3f}")
        print(f"EER      = {c.get('eer', float('nan')):.3f}")
        print(f"convergiu: {r.get('converged')}")
    else:
        print("ERRO:", r.get("error"))
else:
    print("Treino desligado (XFAKE_RUN_EVAL=0).")
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
# (num, slug_arquivo, nome_no_registry, input_type, família, porquê estudar)
#
# `input_type` é HARDCODED aqui (raw_audio | spectrogram | tabular) — fonte
# autoritativa: `input_requirements["input_type"]` do registry. Manter no
# catálogo deixa o gerador determinístico e SEM dependência de TensorFlow no
# build (`_input_type` faz cross-check com o factory só quando ele está
# importável, avisando se divergir). Se mudar um contrato no registry, rode o
# gerador num ambiente com TF: o aviso aponta a linha a atualizar.
MODELS = [
    (1, "wavlm", "WavLM", "raw_audio", "Foundation/SSL (raw-audio)",
     "Backbone auto-supervisionado; cabeça de classificação sobre features SSL."),
    (2, "hubert", "HuBERT", "raw_audio", "Foundation/SSL (raw-audio)",
     "Alternativa SSL ao WavLM; mesmo padrão de fine-tuning."),
    (3, "rawnet2", "RawNet2", "raw_audio", "Forma de onda (SincNet)",
     "Processa PCM 1D via Sinc-Convolutions; baseline end-to-end."),
    (4, "sonic_sleuth", "Sonic Sleuth", "spectrogram",
     "Espectrograma (features in-model)",
     "Extrai LFCC/MFCC/CQT dentro do grafo (tf.signal)."),
    (5, "aasist", "AASIST", "raw_audio", "Forma de onda + grafo",
     "Graph attention espectro-temporal; SOTA em datasets controlados."),
    (6, "rawgat_st", "RawGAT-ST", "raw_audio", "Forma de onda + grafo duplo",
     "SincConv + grafos espectral/temporal com fusão."),
    (7, "conformer", "Conformer", "spectrogram", "Espectrograma (conv+atenção)",
     "Convolução + self-attention para contexto local e global."),
    (8, "hybrid_cnn_transformer", "Hybrid CNN-Transformer", "spectrogram",
     "Espectrograma (híbrido)",
     "CNN para features locais + Transformer para dependências longas."),
    (9, "spectrogram_transformer", "SpectrogramTransformer", "spectrogram",
     "Espectrograma (AST)",
     "Audio Spectrogram Transformer: patches do espectrograma."),
    (10, "efficientnet_lstm", "EfficientNet-LSTM", "spectrogram",
     "Espectrograma + temporal",
     "EfficientNet (mel→224×224) + LSTM com atenção temporal."),
    (11, "multiscale_cnn", "MultiscaleCNN", "spectrogram",
     "Espectrograma (Res2Net)",
     "Convoluções multi-escala (Res2Net); baseline neural principal."),
    (12, "ensemble", "Ensemble", "spectrogram", "Espectrograma (multi-ramo)",
     "5 ramos convolucionais com fusão adaptativa; leve e rápido."),
    (13, "svm", "SVM", "tabular", "Clássico (tabular)",
     "Baseline leve sobre features acústicas agregadas; ideal em CPU."),
    (14, "random_forest", "RandomForest", "tabular", "Clássico (tabular)",
     "Ensemble de árvores sobre features agregadas; rápido e robusto."),
]
CLASSICAL = {"SVM", "RandomForest"}
# Modelos SSL: célula OPCIONAL que instala `transformers` para usar o backbone
# real (WavLM/HuBERT do HuggingFace). Sem ele, o projeto cai numa implementação
# simplificada que treina/infere normalmente — por isso o install é opcional.
SSL_SLUGS = {"wavlm", "hubert"}

SSL_DEPS_CELL = """\
# OPCIONAL: instala `transformers` para usar o backbone SSL pré-treinado real
# (baixa o checkpoint na 1ª execução; requer internet e, idealmente, GPU).
# Se você pular esta célula, o projeto usa uma implementação simplificada do
# WavLM/HuBERT que treina e infere normalmente — só não carrega os pesos SSL.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("transformers") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.30"],
        check=True,
    )
print("transformers disponível — backbone SSL real será usado.")
"""


def _input_type(display_name: str, declared: str) -> str:
    """Retorna o input_type do catálogo. Se o factory estiver importável, faz
    cross-check com o registry e avisa (sem falhar) caso tenha divergido —
    mantém o build determinístico e TF-free."""
    try:
        from app.domain.models.architectures.factory import get_architecture_info

        info = get_architecture_info(display_name)
        if info:
            actual = info.input_requirements.get("input_type")
            if actual and actual != declared:
                print(f"  AVISO: input_type de {display_name} divergiu: "
                      f"catálogo={declared!r} ≠ registry={actual!r} — atualize MODELS.")
            return actual or declared
    except Exception:
        pass  # TF ausente → usa o valor (determinístico) do catálogo.
    return declared


# ───────────────────────── 00_index ─────────────────────────

def build_index():
    # TOC com links relativos (clicáveis no Jupyter e no HTML renderizado).
    model_links = "\n".join(
        f"   - [{num:02d} · {name}](models/{num:02d}_{slug}.ipynb)"
        for num, slug, name, _itype, _family, _why in MODELS
    )
    write_nb("00_index.ipynb", [
        md("""
        # XFakeSong — Índice dos Notebooks

        Estudos organizados em três blocos:

        - `features/`: extração e análise de features.
        - `pipeline/`: benchmark completo, treino e inferência.
        - `models/`: um notebook por arquitetura (14 modelos).

        **Ordem sugerida:** features → pipeline → models.
        """),
        md(f"""
        ## Sumário (links clicáveis)

        1. [Estudo de features](features/01_feature_extraction_study.ipynb)
        2. Pipeline
           - [01 · Benchmark do TCC](pipeline/01_benchmark_tcc_full_pipeline.ipynb)
           - [02 · Treino de um modelo](pipeline/02_training_model.ipynb)
           - [03 · Inferência](pipeline/03_inference.ipynb)
        3. Modelos
        {model_links}
        """),
        md("""
        ## Como rodar

        ### Google Colab (recomendado para treinar com GPU)

        Abra qualquer notebook por:
        `https://colab.research.google.com/github/thierrybraga/XFakeSong/blob/main/notebooks/<caminho>`
        (ex.: `notebooks/pipeline/02_training_model.ipynb`). Depois:

        1. **Ambiente de execução → Alterar o tipo de hardware → GPU** (T4 grátis).
        2. Rode a **primeira célula**: ela clona o repositório e instala as
           dependências de áudio automaticamente (a primeira execução demora um
           pouco).

        ### Localmente

        ```bash
        pip install -r requirements.txt -r requirements-dev.txt   # inclui jupyter/nbclient
        jupyter lab          # ou: jupyter notebook
        ```

        ### Treino

        Os notebooks de **modelo** treinam + avaliam em dados sintéticos ao rodar
        (rápido); defina o ambiente `XFAKE_RUN_EVAL=0` para pular. Todos são
        **self-contained** (sem dataset externo). **WavLM/HuBERT** baixam um
        checkpoint SSL na 1ª execução (requer internet). Treino real com dataset
        próprio: `pipeline/02_training_model.ipynb`.
        """),
        code(BOOTSTRAP),
        md("## Ambiente (versões — reprodutibilidade)"),
        code(SESSION_INFO),
        md("## Notebooks disponíveis"),
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
    for num, slug, name, declared_itype, family, why in MODELS:
        itype = _input_type(name, declared_itype)
        is_classical = name in CLASSICAL
        # célula de inspeção difere para clássico × neural
        if is_classical:
            inspect = f"""
            ARCHITECTURE = {name!r}

            # Modelos clássicos (sklearn) NÃO passam pelo factory Keras: têm um
            # construtor próprio que monta um Pipeline (StandardScaler + classificador).
            # Aqui instanciamos pelo MESMO caminho do benchmark clássico e
            # inspecionamos a ESTRUTURA — simétrico ao summary() dos neurais. O
            # treino+avaliação real acontece na célula seguinte (mesmo harness).
            #
            # Features (T×F) são achatadas para um vetor (T*F,): é exatamente o que
            # o runner clássico faz internamente antes de chamar fit().
            import numpy as np

            if ARCHITECTURE == "SVM":
                from app.domain.models.architectures.svm import (
                    create_svm_model as make_classical,
                )
            else:
                from app.domain.models.architectures.random_forest import (
                    create_random_forest_model as make_classical,
                )

            n_features = int(np.prod(prepared.X.shape[1:]))
            clf = make_classical(input_shape=(n_features,), num_classes=2)
            print("Entrada achatada :", (n_features,))
            print("Pipeline         :", " -> ".join(s for s, _ in clf.pipeline.steps))
            print("Hiperparâmetros  :", clf.get_params())
            """
        else:
            inspect = f"""
            ARCHITECTURE = {name!r}

            # Modelos neurais são instanciados pelo factory (mesmo caminho do
            # benchmark/treino). summary() valida shapes sem treino pesado.
            # num_classes=1 → saída sigmoid de 1 unidade = p(fake) (convenção do
            # projeto). O treino real pode usar 2-unit softmax conforme o preset;
            # o input_contract registra a convenção usada (ver pipeline/).
            from app.domain.models.architectures.factory import create_model_by_name

            input_shape = tuple(prepared.X.shape[1:])
            model = create_model_by_name(ARCHITECTURE, input_shape=input_shape, num_classes=1)
            model.summary()
            print("Parâmetros:", model.count_params())
            """
        # WavLM/HuBERT precisam do `transformers` (download SSL) — célula extra.
        ssl_setup = (
            [md("## Backbone SSL real (`transformers`, opcional)"), code(SSL_DEPS_CELL)]
            if slug in SSL_SLUGS else []
        )
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
            2. Instanciar o modelo pelo **mesmo caminho** do benchmark (`summary()`).
            3. **Treinar + avaliar** em dados sintéticos (accuracy/AUC/EER) — ligado
               por padrão (defina `XFAKE_RUN_EVAL=0` para pular).

            Para treino com o seu próprio dataset, use
            `notebooks/pipeline/02_training_model.ipynb`.
            """),
            code(BOOTSTRAP),
            *ssl_setup,
            md("## Reprodutibilidade (semente numpy + TensorFlow)"),
            code(SEED_ALL),
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
            md("## Treino + avaliação em dados sintéticos"),
            code(EVAL_CELL),
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

            metrics_path = ROOT / "results" / {TCC_RESULTS!r} / "architectures" / {slug!r} / "metrics.json"
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
        md("""
        ## 1. Benchmark rápido (sintético) — valida o harness em segundos

        Treina algumas arquiteturas em **dados sintéticos** e gera as MESMAS
        métricas, tabela e figuras do relatório do TCC. Como os dados são
        sintéticos, **os números não são resultados reais** (acurácia ~50% é
        esperada) — servem só para validar o harness de ponta a ponta. Seed fixo
        (`42`, default do benchmark). Resultados reais: Seção 2.
        """),
        code("""
        import pandas as pd

        try:
            from IPython.display import Image, display
        except Exception:           # fora do kernel IPython
            display, Image = print, None

        from benchmarks import BenchmarkConfig, run_benchmark

        try:
            import tensorflow as tf
            print("GPUs disponíveis:", len(tf.config.list_physical_devices("GPU")))
        except Exception:
            pass

        OUT = ROOT / "results" / "notebook_benchmark"
        cfg = BenchmarkConfig.quick(
            architectures=["MultiscaleCNN", "Ensemble", "SVM", "RandomForest"],
            synthetic_n=160,
            snr_levels_db=[20],
            output_dir=str(OUT),
        )
        run_benchmark(cfg)

        # Tabela canônica do relatório, ordenada por EER (menor = melhor).
        cols = ["arquitetura", "convergiu", "accuracy", "eer", "auc_roc",
                "min_tdcf", "params", "size_mb", "latency_ms"]
        df = pd.read_csv(OUT / "results.csv")
        df = df[[c for c in cols if c in df.columns]].sort_values("eer")
        display(df.round(4))
        """),
        md("### Figuras agregadas geradas pelo benchmark"),
        code("""
        # Exibe inline as figuras que o benchmark gravou em disco.
        figs = OUT / "figures"
        for name in ["roc.png", "score_distributions.png", "confusion_matrices.png",
                     "eficiencia.png", "robustez.png"]:
            fp = figs / name
            if fp.exists() and Image is not None:
                print(name)
                display(Image(filename=str(fp)))
            elif fp.exists():
                print("figura:", fp)
        """),
        md("""
        ## 2. Execução completa do TCC

        > ⚠️ **Pesado — requer GPU e tempo.** Baixa/prepara ~20k amostras (10k
        > real + 10k fake) e treina o preset completo de arquiteturas. Em GPU leva
        > de dezenas de minutos a horas; em CPU é inviável. Por isso fica
        > **desligado por padrão** (`RUN_FULL_PIPELINE = False`). O
        > `--tcc-full-dataset` já ativa download + benchmark completo + probe da API.

        A execução completa grava o relatório em `results/tcc_full_20k/`:
        `dataset.md`, `dataset_manifest.json`, `results.json`/`results.csv`,
        `tcc_report.md` e as figuras `figures/roc.png`,
        `figures/confusion_matrices.png` e `figures/score_distributions.png`
        (estes dois primeiros, `dataset.md`/`dataset_manifest.json`, só saem aqui —
        a Seção 1 não baixa dataset).
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

        Ou o benchmark direto sobre um `.npz` já exportado (veja como gerar o
        dataset em `docs/12_DATASETS.md`):

        ```bash
        python scripts/run_benchmark.py --full --dataset app/datasets/SEU_DATASET.npz
        ```

        Veja `docs/15_BENCHMARK.md` para o mapeamento saída → tabela/figura do TCC.
        """),
        md("## 3. Validar e ler artefatos do relatório"),
        code("""
        import pandas as pd

        try:
            from IPython.display import display
        except Exception:
            display = print

        # Lê o relatório que existir: o completo (Seção 2) OU o rápido (Seção 1) —
        # assim a tabela aparece mesmo sem rodar o pipeline de horas.
        candidates = [ROOT / "results" / "tcc_full_20k",
                      ROOT / "results" / "notebook_benchmark"]
        report_dir = next((d for d in candidates if (d / "results.csv").exists()),
                          candidates[0])
        print("Lendo relatório de:", report_dir)

        # dataset.md/manifest só existem na execução completa (Seção 2).
        required = ["results.csv", "results.json", "tcc_report.md",
                    "figures/roc.png", "figures/confusion_matrices.png",
                    "figures/score_distributions.png"]
        missing = [p for p in required if not (report_dir / p).exists()]
        print("Ausentes:", missing or "nenhum")

        if (report_dir / "results.csv").exists():
            display(pd.read_csv(report_dir / "results.csv"))
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
        md("## Reprodutibilidade (semente numpy + TensorFlow)"),
        code(SEED_ALL),
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
        md("""
        ## 3. Treinar com um dataset REAL (download) — opcional

        Baixa um dataset real do HuggingFace, monta o `.npz` e treina pelo MESMO
        `TrainingService`. Desligado por padrão (download + treino são pesados).

        > ⚠️ **Requer `datasets<4.0`** — a 4.x exige `torchcodec` e quebra o
        > download de áudio (a célula instala a versão certa). Algumas fontes
        > pedem um **token do HuggingFace** (erro 401) — exporte `HF_TOKEN` antes
        > ou veja `docs/12_DATASETS.md`. **FLEURS** (real, PT-BR) é público e leve;
        > fontes *fake* como WaveFake são grandes (dezenas de GB).
        """),
        code("""
        import subprocess

        DOWNLOAD_REAL = False          # → True para baixar + treinar com dados reais
        N_PER_CLASS = 150              # amostras por classe (pequeno p/ demonstração)
        REAL_NPZ = ROOT / "app" / "datasets" / "nb_real.npz"

        if DOWNLOAD_REAL:
            # 1. datasets compatível (a 4.x quebra o download de áudio).
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "datasets>=2.14,<4.0"], check=True)
            # 2. Pipeline canônico: download → preparo → split → export .npz → treino.
            cmd = [
                sys.executable, str(ROOT / "scripts" / "run_tcc_pipeline.py"),
                "--download", "--skip-real-cv",
                "--archs", "SVM",
                "--target-per-class", str(N_PER_CLASS),
                "--epochs", "5",
                "--npz", str(REAL_NPZ),
                "--out", str(ROOT / "results" / "nb_real"),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
            print("NPZ real:", REAL_NPZ, "| existe:", REAL_NPZ.exists())
        else:
            print("DOWNLOAD_REAL=False — usando o dataset sintético da Seção 1.")
            print("Defina DOWNLOAD_REAL=True para baixar dados reais (ver docs/12).")
        """),
        md("""
        Com o `.npz` real em mãos, treine **qualquer arquitetura** repetindo a
        Seção 2 com `dataset_path=str(REAL_NPZ)`. Para baixar uma fonte específica
        direto (sem o pipeline completo):

        ```bash
        pip install 'datasets>=2.14,<4.0'                                  # obrigatório
        python scripts/download_datasets.py --fleurs --max-samples 300     # real (público)
        python scripts/download_datasets.py --brspeech --max-samples 600   # real+fake (pode pedir token)
        ```
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
        md("## Reprodutibilidade (semente numpy + TensorFlow)"),
        code(SEED_ALL),
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
    print("Gerando notebooks ativos (00_index, features/, models/, pipeline/)...")
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
