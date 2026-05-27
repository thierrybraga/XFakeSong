#!/usr/bin/env python3
"""Gera amostras de áudio sintético para testes rápidos de pipeline.

Cria N arquivos WAV em app/datasets/fake/ contendo senóides + ruído
branco — útil para validar que o pipeline de treino/inferência funciona
sem precisar de um dataset real.

Uso:
    python scripts/setup_fake_dataset.py              # 100 amostras (padrão)
    python scripts/setup_fake_dataset.py --n 50       # 50 amostras
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
FAKE_DIR = BASE_DIR / "app" / "datasets" / "fake"
SAMPLE_RATE = 16000
DURATION = 3  # segundos

def generate_synthetic_audio(filename):
    """Gera um arquivo de áudio sintético (ruído + senóides)."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    
    # Gerar sinal "fake" (combinação de frequências artificiais)
    f1 = 440 + np.random.uniform(-50, 50)
    f2 = 880 + np.random.uniform(-50, 50)
    signal = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
    
    # Adicionar ruído
    noise = 0.1 * np.random.normal(0, 1, len(t))
    audio = signal + noise
    
    # Normalizar
    audio = audio / np.max(np.abs(audio))
    
    sf.write(filename, audio, SAMPLE_RATE)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, metavar="N", help="Número de amostras a gerar (padrão: 100)")
    parser.add_argument("--out", type=Path, default=FAKE_DIR, metavar="DIR", help="Diretório de saída")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Gerando {args.n} amostras de áudio fake em {out_dir}...")
    for i in tqdm(range(args.n)):
        filename = out_dir / f"synthetic_fake_{i:04d}.wav"
        if not filename.exists():
            generate_synthetic_audio(filename)

    print(f"Concluído — {args.n} arquivos em {out_dir}")


if __name__ == "__main__":
    main()
