#!/usr/bin/env python3
"""
build_dataset.py — Orquestrador da Fase 1 (Dataset Real e Balanceamento).

Composicao definida:
  REAL  (10.000 amostras):
    - 5.000 x BRSpeech-DF bonafide/   (fala real PT-BR, padrão ASVspoof)
    - 5.000 x CommonVoice v17 PT + FLEURS PT-BR (diversidade de falantes)

  FAKE  (10.000 amostras):
    - 5.000 x BRSpeech-DF spoof/      (múltiplos geradores TTS/VC)
    - 5.000 x Fake Voices XTTS        (gerador independente — cross-generator)

  TOTAL : 20.000 amostras balanceadas 1:1
  SPLIT : 70% treino / 15% validacao / 15% teste (estratificado)

Uso:
  python scripts/build_dataset.py                        # executa tudo
  python scripts/build_dataset.py --target 1000          # 1000 por classe (debug)
  python scripts/build_dataset.py --skip-download        # apenas preprocessa
  python scripts/build_dataset.py --only-splits          # apenas recria os splits
  python scripts/build_dataset.py --delete-excess        # descarte destrutivo dos excedentes
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BuildDataset")

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
OVERFLOW_DIR = DATASETS_DIR / "overflow"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.dataset_catalog import (  # noqa: E402
    DATASET_CATALOG,
    get_tier,
    summarize_dataset_paths,
    tier_choices,
)

# ---------------------------------------------------------------------------
# Composicao do dataset (valores padrao — ajustavel via --target)
# ---------------------------------------------------------------------------
COMPOSITION = {
    "target_per_class": 10000,
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    "sources": {
        "real": [
            {
                "name": "BRSpeech-DF bonafide",
                "hf_repo": "AKCIT-Deepfake/BRSpeech-DF",
                "type": "real",
                "target": 5000,
                "file_prefix": "brspeech",
                "script": "download_datasets.py",
                "args": ["--brspeech"],
            },
            {
                "name": "CommonVoice v17 PT + FLEURS PT-BR",
                "hf_repo": "mozilla-foundation/common_voice_17_0 + google/fleurs",
                "type": "real",
                "target": 5000,
                "file_prefix": "cvpt|fleurs|cetuc",
                "script": "download_datasets.py",
                "args": ["--common-voice-pt", "--fleurs"],
            },
        ],
        "fake": [
            {
                "name": "BRSpeech-DF spoof",
                "hf_repo": "AKCIT-Deepfake/BRSpeech-DF",
                "type": "fake",
                "target": 5000,
                "file_prefix": "brspeech",
                "script": "download_datasets.py",
                "args": ["--brspeech"],
            },
            {
                "name": "Fake Voices XTTS (unfake/fake_voices)",
                "hf_repo": "unfake/fake_voices",
                "type": "fake",
                "target": 5000,
                "file_prefix": "fkvoice",
                "script": "download_datasets.py",
                "args": ["--fake-voices"],
            },
        ],
    },
}


def run(cmd: list, description: str) -> int:
    """Executa um subprocesso e loga o resultado."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXECUTANDO: {description}")
    logger.info(f"  Comando: {' '.join(str(c) for c in cmd)}")
    logger.info(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        logger.error(f"FALHOU (codigo {result.returncode}): {description}")
        raise RuntimeError(
            f"Etapa obrigatoria falhou: {description}. "
            "Interrompendo para evitar balancear/preprocessar dataset incompleto."
        )
    else:
        logger.info(f"OK: {description}")
    return result.returncode


def count_wavs(directory: Path, prefixes: list = None) -> int:
    """Conta WAVs no diretorio, opcionalmente filtrando por prefixo."""
    if not directory.exists():
        return 0
    if not prefixes:
        return len(list(directory.glob("*.wav")))
    total = 0
    for prefix in prefixes:
        total += len(list(directory.glob(f"{prefix}_*.wav")))
    return total


def print_status():
    """Imprime situacao atual do dataset."""
    target_per_class = int(COMPOSITION.get("target_per_class", 10000))
    half_target = target_per_class // 2
    real_total = count_wavs(REAL_DIR)
    fake_total = count_wavs(FAKE_DIR)

    brspeech_real = count_wavs(REAL_DIR, ["brspeech"])
    cv_real = count_wavs(REAL_DIR, ["cetuc", "cv", "cvpt", "fleurs"])
    brspeech_fake = count_wavs(FAKE_DIR, ["brspeech"])
    xtts_fake = count_wavs(FAKE_DIR, ["fkvoice", "fakevoice"])

    logger.info("\n" + "=" * 60)
    logger.info("STATUS ATUAL DO DATASET")
    logger.info("=" * 60)
    logger.info(f"  REAL  total : {real_total:>5d}")
    logger.info(f"    BRSpeech bonafide : {brspeech_real:>5d}  (meta: {half_target})")
    logger.info(f"    CommonVoice/FLEURS: {cv_real:>5d}  (meta: {half_target})")
    logger.info(f"  FAKE  total : {fake_total:>5d}")
    logger.info(f"    BRSpeech spoof    : {brspeech_fake:>5d}  (meta: {half_target})")
    logger.info(f"    Fake Voices XTTS  : {xtts_fake:>5d}  (meta: {half_target})")
    logger.info(f"  TOTAL       : {real_total + fake_total:>5d}  (meta: {target_per_class * 2})")
    logger.info("=" * 60)

    return real_total, fake_total


def step_download(target_per_class: int, skip_real_cv: bool = False):
    """Fase de download das fontes.

    Quando `skip_real_cv` (tiers test/small): a classe real vem INTEIRAMENTE do
    BRSpeech bonafide (alvo = target_per_class). Caso contrário (medium/large):
    metade BRSpeech bonafide + metade CommonVoice/FLEURS.
    """
    half = target_per_class // 2
    cvpt_target = half // 2
    fleurs_target = half - cvpt_target

    # Alvo de real vindo do BRSpeech: tudo (test/small) ou metade (medium/large).
    brspeech_real_goal = target_per_class if skip_real_cv else half

    # --- BRSpeech-DF (real bonafide + fake spoof juntos; o downloader divide
    #     max_samples em metade real + metade fake).
    brspeech_real = count_wavs(REAL_DIR, ["brspeech"])
    brspeech_fake = count_wavs(FAKE_DIR, ["brspeech"])
    brspeech_needed = (
        max(0, brspeech_real_goal - brspeech_real)
        + max(0, brspeech_real_goal - brspeech_fake)
    )

    if brspeech_needed > 0:
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_datasets.py"),
             "--brspeech", "--max-samples", str(brspeech_real_goal * 2)],
            f"BRSpeech-DF bonafide + spoof ({brspeech_real_goal} por classe)",
        )
    else:
        logger.info(f"BRSpeech-DF ja tem {brspeech_real} real + {brspeech_fake} fake. Pulando.")

    # --- CommonVoice PT + FLEURS (apenas real)
    if not skip_real_cv:
        cvpt_real = count_wavs(REAL_DIR, ["cvpt"])
        fleurs_real = count_wavs(REAL_DIR, ["fleurs"])
        public_real = count_wavs(REAL_DIR, ["cetuc", "cv", "cvpt", "fleurs"])

        if cvpt_real < cvpt_target:
            run(
                [sys.executable, str(SCRIPTS_DIR / "download_datasets.py"),
                 "--common-voice-pt", "--max-samples", str(cvpt_target)],
                f"Common Voice PT-BR (meta: {cvpt_target} reais)",
            )
        else:
            logger.info(f"Common Voice PT-BR ja tem {cvpt_real} amostras. Pulando.")

        if fleurs_real < fleurs_target:
            run(
                [sys.executable, str(SCRIPTS_DIR / "download_datasets.py"),
                 "--fleurs", "--max-samples", str(fleurs_target)],
                f"FLEURS PT-BR (meta: {fleurs_target} reais)",
            )
        else:
            logger.info(f"FLEURS PT-BR ja tem {fleurs_real} amostras. Pulando.")

        public_real = count_wavs(REAL_DIR, ["cetuc", "cv", "cvpt", "fleurs"])
        if public_real < half:
            remaining = half - public_real
            run(
                [sys.executable, str(SCRIPTS_DIR / "download_datasets.py"),
                 "--cetuc", "--max-samples", str(remaining)],
                f"CETUC/OpenSLR fallback (faltam {remaining} reais publicos)",
            )

    # --- Fake Voices XTTS
    xtts_fake = count_wavs(FAKE_DIR, ["fkvoice", "fakevoice"])
    if xtts_fake < half:
        # max_speakers * max_per_speaker ~= half
        max_speakers = max(10, half // 50)
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_datasets.py"),
             "--fake-voices", "--max-speakers", str(max_speakers)],
            f"Fake Voices XTTS (meta: {half} fakes, {max_speakers} falantes)",
        )
    else:
        logger.info(f"Fake Voices XTTS ja tem {xtts_fake} amostras. Pulando.")


def _excess_round_robin(files: list[Path], keep_n: int) -> list[Path]:
    """Escolhe quais arquivos REMOVER mantendo um mix proporcional por fonte.

    Em vez de cortar a cauda alfabética (que zeraria um gerador inteiro, p.ex.
    todos os `fkvoice_*`), agrupa por prefixo de fonte e mantém os primeiros
    `keep_n` em ordem round-robin entre as fontes — preservando a diversidade
    de geradores (importante para o protocolo cross-generator).
    """
    if keep_n >= len(files):
        return []
    by_prefix: dict[str, list[Path]] = {}
    for f in files:
        prefix = f.stem.split("_", 1)[0]
        by_prefix.setdefault(prefix, []).append(f)
    order = sorted(by_prefix)  # determinístico
    kept: list[Path] = []
    i = 0
    while len(kept) < keep_n and any(by_prefix[p] for p in order):
        bucket = by_prefix[order[i % len(order)]]
        if bucket:
            kept.append(bucket.pop(0))
        i += 1
    kept_set = set(kept)
    return [f for f in files if f not in kept_set]


def _archive_or_delete(files: list[Path], label: str, delete_excess: bool) -> int:
    """Arquiva excedentes fora do conjunto ativo ou remove se solicitado."""
    moved = 0
    archive_dir = OVERFLOW_DIR / label
    if not delete_excess:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        if delete_excess:
            src.unlink()
            moved += 1
            continue

        dst = archive_dir / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while dst.exists():
                dst = archive_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.move(str(src), str(dst))
        moved += 1
    return moved


def step_balance(target_per_class: int, delete_excess: bool = False):
    """
    Garante balanceamento 1:1 entre classes.
    Se uma classe tiver mais que target_per_class, arquiva o excesso em
    app/datasets/overflow/ por padrão, mantendo os WAVs brutos recuperáveis.
    Use --delete-excess apenas quando quiser descarte destrutivo.
    """
    logger.info("\n" + "=" * 60)
    logger.info("BALANCEAMENTO 1:1")
    logger.info("=" * 60)

    real_files = sorted(REAL_DIR.glob("*.wav"))
    fake_files = sorted(FAKE_DIR.glob("*.wav"))

    real_count = len(real_files)
    fake_count = len(fake_files)

    logger.info(f"  Antes: {real_count} real + {fake_count} fake")

    # Calcular alvo: minimo entre target e o que existe, garantindo igualdade
    effective_target = min(target_per_class, real_count, fake_count)

    if effective_target < 100:
        logger.warning(
            f"  ATENCAO: Apenas {effective_target} amostras por classe disponíveis. "
            "Execute os downloads primeiro."
        )
        return real_count, fake_count

    # Remover excesso de REAL
    if real_count > effective_target:
        excess = real_count - effective_target
        # Round-robin por fonte preserva a diversidade (não zera um gerador).
        to_remove = _excess_round_robin(real_files, effective_target)
        moved = _archive_or_delete(to_remove, "real", delete_excess)
        action = "removidos" if delete_excess else "arquivados"
        logger.info(f"  Real: {action} {moved}/{excess} arquivos excedentes")

    # Remover excesso de FAKE
    if fake_count > effective_target:
        excess = fake_count - effective_target
        to_remove = _excess_round_robin(fake_files, effective_target)
        moved = _archive_or_delete(to_remove, "fake", delete_excess)
        action = "removidos" if delete_excess else "arquivados"
        logger.info(f"  Fake: {action} {moved}/{excess} arquivos excedentes")

    real_final = len(list(REAL_DIR.glob("*.wav")))
    fake_final = len(list(FAKE_DIR.glob("*.wav")))
    ratio = real_final / max(fake_final, 1)

    logger.info(f"  Depois: {real_final} real + {fake_final} fake (ratio {ratio:.3f})")

    if abs(ratio - 1.0) > 0.05:
        logger.warning("  AVISO: Ratio fora de 5% de 1.0. Verifique os downloads.")
    else:
        logger.info("  Balanceamento OK (ratio dentro de 5% de 1:1)")

    return real_final, fake_final


def step_preprocess(train_ratio: float, val_ratio: float, test_ratio: float,
                    speaker_disjoint: bool = False):
    """Roda o pipeline de pre-processamento com os ratios corretos."""
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "preprocess_dataset.py"),
        "--full",
        "--train-ratio", str(train_ratio),
        "--val-ratio",   str(val_ratio),
        "--test-ratio",  str(test_ratio),
    ]
    if speaker_disjoint:
        cmd.append("--speaker-disjoint")
    run(
        cmd,
        f"Pre-processamento + splits {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
        + (" (disjunto por falante)" if speaker_disjoint else ""),
    )


def save_dataset_config(target_per_class: int, train_r: float, val_r: float, test_r: float,
                        tier: str | None = None):
    """Salva dataset_config.json com a estrategia documentada."""
    real_total = count_wavs(REAL_DIR)
    fake_total = count_wavs(FAKE_DIR)

    brspeech_real = count_wavs(REAL_DIR, ["brspeech"])
    cv_real = count_wavs(REAL_DIR, ["cetuc", "cv", "cvpt", "fleurs"])
    brspeech_fake = count_wavs(FAKE_DIR, ["brspeech"])
    xtts_fake = count_wavs(FAKE_DIR, ["fkvoice", "fakevoice"])
    active_paths = [
        str(path.relative_to(BASE_DIR))
        for path in list(REAL_DIR.glob("*.wav")) + list(FAKE_DIR.glob("*.wav"))
    ]

    tier_info = get_tier(tier) if tier else None
    split_strategy = (
        tier_info.split_strategy if tier_info else "stratified"
    )
    speaker_aware = bool(tier_info.speaker_aware) if tier_info else False

    speakers_summary = {}
    try:
        from app.core.speaker_manifest import summarize_speakers

        speakers_summary = summarize_speakers(active_paths)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Resumo de falantes indisponivel: {exc}")

    config = {
        "version": "1.1",
        "description": "Dataset PT-BR para deteccao de deepfake de audio — TCC UFSJ 2026",
        "tier": tier or "custom",
        "tier_purpose": tier_info.purpose if tier_info else "alvo manual via --target",
        "split_strategy": split_strategy,
        "speaker_aware": speaker_aware,
        "speakers": speakers_summary,
        "balancing_strategy": (
            "1:1 (real:fake), conjunto ativo limitado por classe; "
            "excedentes arquivados em app/datasets/overflow por padrão"
        ),
        "target_per_class": target_per_class,
        "total_samples": real_total + fake_total,
        "real_samples": real_total,
        "fake_samples": fake_total,
        "ratio_real_fake": round(real_total / max(fake_total, 1), 4),
        "source_summary": summarize_dataset_paths(active_paths),
        "dataset_catalog": {
            name: {
                "type": info.source_type,
                "cli_flag": info.cli_flag,
                "prefixes": list(info.prefixes),
                "repository": info.repository,
                "url": info.url,
                "license": info.license,
                "language": info.language,
                "audio_count": info.audio_count,
                "duration": info.duration,
                "speakers": info.speakers,
                "benchmark_use": info.benchmark_use,
            }
            for name, info in DATASET_CATALOG.items()
        },
        "split": {
            "train": train_r,
            "val": val_r,
            "test": test_r,
            "method": (
                "StratifiedGroupKFold por falante (random_state=42)"
                if split_strategy == "speaker_disjoint"
                else "StratifiedShuffleSplit (random_state=42)"
            ),
        },
        "sources": {
            "real": {
                "BRSpeech-DF bonafide": {
                    "repo": "AKCIT-Deepfake/BRSpeech-DF",
                    "license": DATASET_CATALOG["BRSpeech-DF"].license,
                    "count": brspeech_real,
                    "target": target_per_class // 2,
                },
                "CommonVoice_v17_PT_FLEURS": {
                    "repo": "mozilla-foundation/common_voice_17_0 + google/fleurs",
                    "license": "CC0 / CC-BY-4.0",
                    "count": cv_real,
                    "target": target_per_class // 2,
                },
            },
            "fake": {
                "BRSpeech-DF spoof": {
                    "repo": "AKCIT-Deepfake/BRSpeech-DF",
                    "license": DATASET_CATALOG["BRSpeech-DF"].license,
                    "generators": "multiplos TTS/VC PT-BR",
                    "count": brspeech_fake,
                    "target": target_per_class // 2,
                },
                "Fake_Voices_XTTS": {
                    "repo": "unfake/fake_voices",
                    "license": "MIT",
                    "generators": "XTTS v2 (Coqui TTS)",
                    "count": xtts_fake,
                    "target": target_per_class // 2,
                },
            },
        },
        "audio_format": {
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "encoding": "PCM_16",
            "duration_range_sec": [1.0, 30.0],
        },
        "preprocessing": {
            "vad": "Silero VAD (torch.hub)",
            "agc": "Peak normalization 0.95 headroom",
            "resampling": "librosa (kaiser_best)",
            "duplicate_removal": "MD5 hash",
        },
    }

    config_path = DATASETS_DIR / "dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"\nConfig salvo em: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador de dataset PT-BR para deepfake detection (Fase 1 TCC)"
    )
    parser.add_argument(
        "--tier", choices=tier_choices(), default=None,
        help="Tier de dataset (test/small/medium/large) — define tamanho, fontes, "
             "split e protocolo de falante. Veja docs/12_DATASETS.md.",
    )
    parser.add_argument(
        "--target", type=int, default=None,
        help="Numero de amostras por classe (override; default do tier, ou 10000 sem tier)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Proporcao de treino (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Proporcao de validacao (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Proporcao de teste (default: 0.15)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Pular downloads (usar apenas o que ja existe)",
    )
    parser.add_argument(
        "--skip-real-cv", action="store_true",
        help="Pular download de CommonVoice/FLEURS (apenas BRSpeech + FakeVoices)",
    )
    parser.add_argument(
        "--only-splits", action="store_true",
        help="Apenas recriar os splits (skip download + balance)",
    )
    parser.add_argument(
        "--delete-excess", action="store_true",
        help="Remove excedentes em vez de arquivar em app/datasets/overflow",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Apenas mostrar status atual do dataset",
    )

    args = parser.parse_args()

    # Resolver tier → tamanho/fontes/split/falante (override manual via --target).
    tier = get_tier(args.tier) if args.tier else None
    if tier is not None:
        target_per_class = int(args.target) if args.target is not None else tier.per_class
        skip_real_cv = args.skip_real_cv or tier.skip_real_cv
        speaker_disjoint = tier.split_strategy == "speaker_disjoint"
        train_r, val_r, test_r = (
            tier.split["train"], tier.split["val"], tier.split["test"],
        )
    else:
        target_per_class = int(args.target) if args.target is not None else 10000
        skip_real_cv = args.skip_real_cv
        speaker_disjoint = False
        train_r, val_r, test_r = args.train_ratio, args.val_ratio, args.test_ratio

    COMPOSITION["target_per_class"] = target_per_class

    # Validar ratios
    total_ratio = train_r + val_r + test_r
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Soma dos ratios deve ser 1.0 (atual: {total_ratio:.3f})")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("BUILD DATASET — FASE 1 TCC (UFSJ 2026)")
    logger.info("=" * 60)
    if tier is not None:
        logger.info(f"  Tier              : {tier.name} — {tier.purpose}")
        logger.info(f"  Estrategia split  : {tier.split_strategy}"
                    + (" (disjunto por falante)" if speaker_disjoint else ""))
    logger.info(f"  Target por classe : {target_per_class}")
    logger.info(f"  Total esperado    : {target_per_class * 2}")
    logger.info(f"  Split             : {int(train_r*100)}/{int(val_r*100)}/{int(test_r*100)}")
    real_comp = (
        f"{target_per_class} BRSpeech-bonafide"
        if skip_real_cv
        else f"{target_per_class//2} BRSpeech-bonafide + {target_per_class//2} CommonVoice/FLEURS"
    )
    logger.info(f"  Composicao real   : {real_comp}")
    logger.info(f"  Composicao fake   : {target_per_class//2} BRSpeech-spoof + {target_per_class//2} Fake Voices XTTS")

    if args.status:
        print_status()
        return

    # --- Etapa 1: Downloads
    if not args.skip_download and not args.only_splits:
        logger.info("\n>>> ETAPA 1: DOWNLOADS")
        step_download(target_per_class, skip_real_cv=skip_real_cv)
    else:
        logger.info("\n>>> ETAPA 1: DOWNLOADS (pulado)")

    # --- Etapa 1.5: Dedup ANTES do balance
    # O preprocess (--full) tambem deduplica, mas DEPOIS do balance — o que
    # remove duplicatas ja contadas no 1:1 e reintroduz desbalanceamento
    # (ex.: real 10000 -> 7500 apos dedup, com fake intacto). Deduplicar aqui,
    # antes de balancear, garante que o balanceamento opere sobre dados unicos.
    if not args.only_splits:
        logger.info("\n>>> ETAPA 1.5: DEDUP (antes do balance)")
        run(
            [sys.executable, str(SCRIPTS_DIR / "preprocess_dataset.py"),
             "--remove-duplicates"],
            "Remocao de duplicatas por hash (pre-balance)",
        )

    # --- Etapa 2: Balanceamento
    if not args.only_splits:
        logger.info("\n>>> ETAPA 2: BALANCEAMENTO")
        step_balance(target_per_class, delete_excess=args.delete_excess)

    # --- Etapa 3: Pre-processamento + Splits
    logger.info("\n>>> ETAPA 3: PRE-PROCESSAMENTO + SPLITS")
    step_preprocess(train_r, val_r, test_r, speaker_disjoint=speaker_disjoint)

    # --- Etapa 4: Salvar config
    logger.info("\n>>> ETAPA 4: SALVAR CONFIG")
    save_dataset_config(target_per_class, train_r, val_r, test_r, tier=args.tier)

    # --- Status final
    real_total, fake_total = print_status()

    logger.info("\n" + "=" * 60)
    logger.info("BUILD CONCLUIDO")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {real_total + fake_total} amostras ({real_total} real + {fake_total} fake)")
    logger.info(f"  Splits : app/datasets/splits/train/ | val/ | test/")
    logger.info(f"  Config : app/datasets/dataset_config.json")
    logger.info("\nProximo passo (Fase 2):")
    logger.info("  python scripts/train_advanced.py --model conformer --epochs 100")
    logger.info("  (ou use a interface Gradio: python main.py --gradio)")


if __name__ == "__main__":
    main()
