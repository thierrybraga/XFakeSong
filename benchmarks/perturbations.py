"""Perturbações de robustez além do AWGN (rigor acadêmico, 2026-07-14).

O AWGN é a perturbação mais artificial do repertório: deepfakes reais
circulam por telefonia/mensageiros, onde a degradação dominante é a de
CODEC com perdas. Este módulo aplica um round-trip de codec (MP3/Opus)
à forma de onda canônica — ANTES dos frontends, no mesmo ponto do
protocolo em que o AWGN é aplicado — via ffmpeg.

Uso no benchmark: `BenchmarkConfig.codec_eval = ["mp3", "opus"]`
(desligado por padrão; requer ffmpeg no PATH — presente na imagem Docker
e no ambiente Windows do projeto).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger("benchmark")

# Parâmetros típicos de mensageria/telefonia (VoIP ~24 kbps Opus;
# MP3 64 kbps é o piso comum de reencode agressivo de áudio de voz).
CODEC_SPECS: Dict[str, Dict[str, str]] = {
    "mp3": {"ext": ".mp3", "codec": "libmp3lame", "bitrate": "64k"},
    "opus": {"ext": ".opus", "codec": "libopus", "bitrate": "24k"},
}


def ffmpeg_available() -> bool:
    """True quando o executável ffmpeg está no PATH."""
    return shutil.which("ffmpeg") is not None


def _write_wav(path: Path, x: np.ndarray, sample_rate: int) -> None:
    # PCM 16-bit — entrada canônica dos codecs de voz.
    x = np.asarray(x, dtype="float32").ravel()
    peak = float(np.abs(x).max())
    if peak > 1.0:  # protege o quantizador; preserva a forma
        x = x / peak
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        n = fh.getnframes()
        pcm = np.frombuffer(fh.readframes(n), dtype="<i2")
    return (pcm.astype("float32") / 32767.0).copy()


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou: {proc.stderr[-500:]}")


def codec_roundtrip(
    X: np.ndarray,
    codec: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Aplica encode→decode com perdas a um lote de formas de onda.

    Retorna array com o MESMO shape de entrada (o decode é reamostrado de
    volta a ``sample_rate`` e ajustado ao comprimento original — codecs
    inserem padding/delay de frame). Determinístico (codecs são funções
    determinísticas do sinal), então não há semente.
    """
    spec = CODEC_SPECS.get(codec)
    if spec is None:
        raise ValueError(
            f"Codec desconhecido: {codec!r}. Disponíveis: {sorted(CODEC_SPECS)}"
        )
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg não encontrado no PATH — necessário para codec_eval."
        )

    X = np.asarray(X, dtype="float32")
    flat = X.reshape(len(X), -1)
    out = np.empty_like(flat)
    with tempfile.TemporaryDirectory(prefix="codec_") as td:
        tmp = Path(td)
        src = tmp / "in.wav"
        enc = tmp / f"enc{spec['ext']}"
        dec = tmp / "out.wav"
        for i, row in enumerate(flat):
            _write_wav(src, row, sample_rate)
            _run_ffmpeg(
                ["-i", str(src), "-c:a", spec["codec"], "-b:a",
                 spec["bitrate"], "-ar", str(sample_rate), str(enc)]
            )
            _run_ffmpeg(
                ["-i", str(enc), "-ar", str(sample_rate), "-ac", "1", str(dec)]
            )
            y = _read_wav(dec)
            # Alinha comprimento: codecs adicionam delay/padding de frame.
            if len(y) >= flat.shape[1]:
                y = y[: flat.shape[1]]
            else:
                y = np.pad(y, (0, flat.shape[1] - len(y)))
            out[i] = y
    return out.reshape(X.shape).astype("float32")
