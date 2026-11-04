#!/usr/bin/env python3
"""
download_datasets.py — Download de datasets para detecção de deepfake de áudio.

Substitui download_portuguese_datasets.py e download_pt_datasets_v2.py.

Datasets PT-BR (foco local):
  --brspeech      BRSpeech-DF (AKCIT-Deepfake/BRSpeech-DF): bonafide→real, spoof→fake
  --cetuc         CETUC/CommonVoice v17 PT-BR / FLEURS / OpenSLR (real)
  --fake-voices   Fake Voices XTTS (unfake/fake_voices): ZIPs por falante (fake)
  --fleurs        FLEURS PT-BR Google (real)
  --mls-portuguese  MLS Portuguese/LibriVox via download direto (real, CC BY 4.0)
  --tts-portuguese  TTS-Portuguese Corpus via download direto (real, CC BY 4.0)
  --mlaad-pt      MLAAD v9 subconjunto PT (fake, CC-BY-NC 4.0)
  --common-voice-pt  Mozilla Common Voice v17 PT (real, CC0)

Datasets internacionais (referência anti-spoofing):
  --asvspoof2019  ASVspoof 2019 LA — benchmark padrão (real + fake, ODC-BY)
  --wavefake      WaveFake — 6 vocoders TTS (fake only, MIT)
  --in-the-wild   In-the-Wild — 58 celebridades (real + deepfake, CC 4.0)
  --asvspoof5     ASVspoof 5 (2024) — 20+ tipos de ataque (real + fake)

Atalhos:
  --all           brspeech + cetuc + fake-voices (recomendado para build_dataset.py)
  --all-intl      asvspoof2019 + wavefake + in-the-wild (datasets internacionais)

Uso:
  python tools/dataset/download_datasets.py --all --max-samples 2000
  python tools/dataset/download_datasets.py --brspeech --max-samples 1000
  python tools/dataset/download_datasets.py --cetuc --max-samples 1000
  python tools/dataset/download_datasets.py --fake-voices --max-speakers 20
  python tools/dataset/download_datasets.py --fleurs --max-samples 500
  python tools/dataset/download_datasets.py --mlaad-pt --max-samples 500
  python tools/dataset/download_datasets.py --asvspoof2019 --max-samples 2000
  python tools/dataset/download_datasets.py --wavefake --max-samples 2000
  python tools/dataset/download_datasets.py --in-the-wild --max-samples 1000
  python tools/dataset/download_datasets.py --asvspoof5 --max-samples 2000
  python tools/dataset/download_datasets.py --all-intl --max-samples 2000

CLI fina: a lógica vive em ``app/domain/services/dataset/download.py``,
a mesma que a aba de datasets do Gradio usa.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.services.dataset.download import (  # noqa: E402
    DEFAULT_MAX_SAMPLES,
    check_dependencies,
    download_asvspoof5,
    download_asvspoof2019,
    download_brspeech,
    download_cetuc,
    download_common_voice_pt,
    download_fake_voices,
    download_fleurs,
    download_in_the_wild,
    download_mlaad_pt,
    download_mls_portuguese,
    download_tts_portuguese,
    download_wavefake,
    print_report,
    setup_dirs,
    warn_if_datasets_incompatible,
)

logger = logging.getLogger("DownloadDatasets")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── PT-BR ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--all",
        action="store_true",
        help="brspeech + cetuc + fake-voices (recomendado PT-BR)",
    )
    parser.add_argument(
        "--brspeech", action="store_true", help="BRSpeech-DF (real + fake PT-BR)"
    )
    parser.add_argument(
        "--cetuc", action="store_true", help="CETUC/CommonVoice PT-BR (real)"
    )
    parser.add_argument(
        "--fake-voices", action="store_true", help="Fake Voices XTTS via ZIPs (fake)"
    )
    parser.add_argument(
        "--fleurs", action="store_true", help="FLEURS PT-BR do Google (real)"
    )
    parser.add_argument(
        "--mls-portuguese",
        action="store_true",
        help="MLS Portuguese/LibriVox via download direto (real, CC BY 4.0)",
    )
    parser.add_argument(
        "--tts-portuguese",
        action="store_true",
        help="TTS-Portuguese Corpus via download direto (real, CC BY 4.0)",
    )
    parser.add_argument(
        "--mlaad-pt",
        action="store_true",
        help="MLAAD v9 subconjunto PT (fake, CC-BY-NC 4.0)",
    )
    parser.add_argument(
        "--common-voice-pt",
        action="store_true",
        help="Mozilla Common Voice v17 PT (real, CC0)",
    )
    # ── Internacional ──────────────────────────────────────────────────────
    parser.add_argument(
        "--all-intl",
        action="store_true",
        help="asvspoof2019 + wavefake + in-the-wild (referência anti-spoofing)",
    )
    parser.add_argument(
        "--asvspoof2019",
        action="store_true",
        help="ASVspoof 2019 LA — benchmark padrão (real + fake, ODC-BY)",
    )
    parser.add_argument(
        "--wavefake",
        action="store_true",
        help="WaveFake — 6 vocoders TTS (fake only, MIT)",
    )
    parser.add_argument(
        "--in-the-wild",
        action="store_true",
        help="In-the-Wild — 58 celebridades (real + deepfake, CC 4.0)",
    )
    parser.add_argument(
        "--asvspoof5",
        action="store_true",
        help="ASVspoof 5 (2024) — 20+ tipos de ataque (real + fake)",
    )
    # ── Parâmetros ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Máximo de amostras por classe (padrão: {DEFAULT_MAX_SAMPLES})",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=20,
        help="Máximo de falantes do Fake Voices (padrão: 20)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    warn_if_datasets_incompatible()

    _any = any(
        [
            args.all,
            args.all_intl,
            args.brspeech,
            args.cetuc,
            args.fake_voices,
            args.fleurs,
            args.mls_portuguese,
            args.tts_portuguese,
            args.mlaad_pt,
            args.common_voice_pt,
            args.asvspoof2019,
            args.wavefake,
            args.in_the_wild,
            args.asvspoof5,
        ]
    )
    if not _any:
        parser.print_help()
        print("\nExemplos recomendados:")
        print("  python tools/dataset/download_datasets.py --all --max-samples 2000")
        print(
            "  python tools/dataset/download_datasets.py --all-intl --max-samples 2000"
        )
        print(
            "  python tools/dataset/download_datasets.py --asvspoof2019 --wavefake --max-samples 1000"
        )
        return

    # Pré-check de dependências: avisa sobre TUDO que falta de uma vez,
    # antes de iniciar qualquer download (BUG.Datasets.2).
    needs_hub = args.all or args.fake_voices  # Fake Voices usa huggingface_hub
    core_deps = ["datasets"]
    if needs_hub:
        core_deps += ["huggingface_hub", "pandas"]
    missing = check_dependencies(core_deps)
    if missing:
        logger.warning(
            "Algumas dependências de download podem estar faltando:\n"
            f"    pip install {' '.join(missing)}\n"
            "  Continuando — fontes individuais avisarão se não puderem rodar."
        )

    setup_dirs()
    logger.info(f"Max amostras por classe: {args.max_samples}")

    # PT-BR
    if args.all or args.brspeech:
        download_brspeech(args.max_samples)

    if args.all or args.cetuc:
        download_cetuc(args.max_samples)

    if args.all or args.fake_voices:
        download_fake_voices(max_speakers=args.max_speakers)

    if args.fleurs:
        download_fleurs(args.max_samples)

    if args.mls_portuguese:
        download_mls_portuguese(args.max_samples)

    if args.tts_portuguese:
        download_tts_portuguese(args.max_samples)

    if args.mlaad_pt:
        download_mlaad_pt(args.max_samples)

    if args.common_voice_pt:
        download_common_voice_pt(args.max_samples)

    # Internacional
    if args.all_intl or args.asvspoof2019:
        download_asvspoof2019(args.max_samples)

    if args.all_intl or args.wavefake:
        download_wavefake(args.max_samples)

    if args.all_intl or args.in_the_wild:
        download_in_the_wild(args.max_samples)

    if args.asvspoof5:
        download_asvspoof5(args.max_samples)

    print_report()
    logger.info("Download concluído!")


if __name__ == "__main__":
    main()
