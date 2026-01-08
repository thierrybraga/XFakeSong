
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Configurações
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
FAKE_DIR = APP_DIR / "datasets" / "fake"
SAMPLE_RATE = 16000
DURATION = 3  # segundos
NUM_SAMPLES = 100

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

def main():
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Gerando {NUM_SAMPLES} amostras de áudio fake em {FAKE_DIR}...")
    
    for i in tqdm(range(NUM_SAMPLES)):
        filename = FAKE_DIR / f"synthetic_fake_{i:04d}.wav"
        if not filename.exists():
            generate_synthetic_audio(filename)
            
    print("Concluído.")

if __name__ == "__main__":
    main()
