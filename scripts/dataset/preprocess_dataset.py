#!/usr/bin/env python3
"""
Pré-processamento e validação de datasets de áudio para o XFakeSong.

Funcionalidades:
  - Verifica integridade dos WAVs (corrompidos, silenciosos)
  - Reamostra para 16kHz mono
  - Normaliza amplitude
  - Remove amostras fora do range de duração (1-30s)
  - Remove duplicatas por hash
  - Gera relatório detalhado
  - Cria splits train/val/test estratificados

Uso:
  python scripts/dataset/preprocess_dataset.py --validate          # Apenas validar
  python scripts/dataset/preprocess_dataset.py --normalize         # Normalizar todos
  python scripts/dataset/preprocess_dataset.py --create-splits     # Criar train/val/test
  python scripts/dataset/preprocess_dataset.py --full              # Tudo acima
  python scripts/dataset/preprocess_dataset.py --create-zip        # Gerar ZIP para upload na UI
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DatasetPreprocessor")

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# Consolidado 2026-07-14: raiz canônica é data/datasets (settings.paths.datasets_dir);
# o antigo app/datasets causou fragmentação (stub de 64 amostras homônimo do dataset real).
DATASETS_DIR = BASE_DIR / "data" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
PROCESSED_DIR = DATASETS_DIR / "processed"
PROCESSED_REAL_DIR = PROCESSED_DIR / "real"
PROCESSED_FAKE_DIR = PROCESSED_DIR / "fake"
SPLITS_DIR = DATASETS_DIR / "splits"

TARGET_SR = 16_000
MIN_DURATION = 1.0
MAX_DURATION = 30.0
SILENCE_THRESHOLD_DB = -50  # abaixo disso = silêncio
RESAMPLE_TYPE = "soxr_hq"
PREPROCESSING_VERSION = "xfakesong-audio-canonical-v2"


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------
def validate_dataset():
    """Valida integridade de todos os arquivos WAV."""
    logger.info("=" * 60)
    logger.info("VALIDACAO DO DATASET")
    logger.info("=" * 60)

    issues = {"corrupted": [], "too_short": [], "too_long": [], "silent": [],
              "wrong_sr": [], "stereo": [], "nan_inf": []}
    stats = {"real": {"count": 0, "total_duration": 0.0, "durations": []},
             "fake": {"count": 0, "total_duration": 0.0, "durations": []}}

    for label, directory in [("real", REAL_DIR), ("fake", FAKE_DIR)]:
        wav_files = sorted(directory.glob("*.wav"))
        logger.info(f"\nValidando {len(wav_files)} arquivos em {directory.name}/...")

        for wav_path in wav_files:
            try:
                info = sf.info(str(wav_path))
                duration = info.duration
                sr = info.samplerate
                channels = info.channels

                if sr != TARGET_SR:
                    issues["wrong_sr"].append((str(wav_path), sr))

                if channels > 1:
                    issues["stereo"].append(str(wav_path))

                if duration < MIN_DURATION:
                    issues["too_short"].append((str(wav_path), duration))
                    continue

                if duration > MAX_DURATION:
                    issues["too_long"].append((str(wav_path), duration))
                    continue

                # Carregar amostra para checagens de conteúdo
                y, _ = librosa.load(str(wav_path), sr=TARGET_SR, duration=5.0)

                # BUG FIX: detectar NaN/Inf — arquivos corrompidos que passariam
                # silenciosamente (NaN < threshold é sempre False → contado válido!)
                # e depois causariam loss:nan no treino.
                if not np.all(np.isfinite(y)):
                    issues["nan_inf"].append(str(wav_path))
                    continue

                # Checar silêncio
                rms = np.sqrt(np.mean(y**2))
                if rms < 10 ** (SILENCE_THRESHOLD_DB / 20):
                    issues["silent"].append(str(wav_path))
                    continue

                stats[label]["count"] += 1
                stats[label]["total_duration"] += duration
                stats[label]["durations"].append(duration)

            except Exception as e:
                issues["corrupted"].append((str(wav_path), str(e)))

    # Relatório
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADO DA VALIDACAO")
    logger.info("=" * 60)

    for label in ["real", "fake"]:
        s = stats[label]
        if s["durations"]:
            logger.info(f"\n  [{label.upper()}]")
            logger.info(f"    Arquivos validos : {s['count']}")
            logger.info(f"    Duracao total    : {s['total_duration']/3600:.1f}h")
            logger.info(f"    Duracao media    : {np.mean(s['durations']):.1f}s")
            logger.info(f"    Duracao min/max  : {np.min(s['durations']):.1f}s / {np.max(s['durations']):.1f}s")
        else:
            logger.warning(f"\n  [{label.upper()}] Nenhum arquivo valido encontrado!")

    total_issues = sum(len(v) for v in issues.values())
    if total_issues > 0:
        logger.warning(f"\n  Problemas encontrados: {total_issues}")
        if issues["corrupted"]:
            logger.warning(f"    Corrompidos    : {len(issues['corrupted'])}")
        if issues["nan_inf"]:
            logger.warning(f"    NaN/Inf (graves): {len(issues['nan_inf'])} — REMOVA antes de treinar!")
        if issues["too_short"]:
            logger.warning(f"    Muito curtos   : {len(issues['too_short'])}")
        if issues["too_long"]:
            logger.warning(f"    Muito longos   : {len(issues['too_long'])}")
        if issues["silent"]:
            logger.warning(f"    Silenciosos    : {len(issues['silent'])}")
        if issues["wrong_sr"]:
            logger.warning(f"    Sample rate != 16kHz: {len(issues['wrong_sr'])}")
        if issues["stereo"]:
            logger.warning(f"    Stereo (nao mono)   : {len(issues['stereo'])}")
    else:
        logger.info("\n  Nenhum problema encontrado!")

    return stats, issues


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------
_NORM_MANIFEST = PROCESSED_DIR / "preprocessing_manifest.json"


def _file_sig(path: Path) -> dict:
    """Content-addressed signature used by the preprocessing cache."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "sha256": digest.hexdigest(),
        "size": int(path.stat().st_size),
        "pipeline_version": PREPROCESSING_VERSION,
    }


def _load_norm_manifest() -> dict:
    try:
        return json.loads(_NORM_MANIFEST.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_norm_manifest(manifest: dict) -> None:
    try:
        _NORM_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        _NORM_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"  nao foi possivel salvar manifesto de normalizacao: {e}")


def normalize_all(force: bool = False):
    """Create canonical WAVs without overwriting or deleting source audio.

    Decode mono with explicit soxr_hq resampling and write to ``processed/``.
    Preserve loudness except when attenuation is required to avoid clipping.
    Invalid inputs are rejected from the processed layer while acquisition
    files remain immutable.
    """
    logger.info("=" * 60)
    logger.info("NORMALIZACAO DO DATASET")
    logger.info("=" * 60)

    fixed = 0
    removed = 0
    skipped = 0
    manifest = {} if force else _load_norm_manifest()

    for label, directory, output_dir in [
        ("real", REAL_DIR, PROCESSED_REAL_DIR),
        ("fake", FAKE_DIR, PROCESSED_FAKE_DIR),
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_files = sorted(directory.glob("*.wav"))
        logger.info(f"\nNormalizando {len(wav_files)} arquivos em {directory.name}/...")

        for wav_path in wav_files:
            rel = str(wav_path.relative_to(DATASETS_DIR)).replace("\\", "/")
            output_path = output_dir / wav_path.name
            if not force:
                try:
                    if output_path.exists() and manifest.get(rel) == _file_sig(wav_path):
                        skipped += 1
                        continue  # ja normalizado e inalterado
                except OSError:
                    pass
            try:
                y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True, res_type=RESAMPLE_TYPE)

                # BUG FIX: sanitizar NaN/Inf ANTES de qualquer cálculo numérico.
                # Sem isso: np.max(np.abs(NaN))=NaN → y/NaN=NaN → arquivo salvo
                # como NaN → wizard de treino lê e gera loss:nan na 1ª época.
                if not np.all(np.isfinite(y)):
                    finite = np.isfinite(y)
                    if finite.sum() < y.size * 0.5:
                        logger.warning(f"  Rejeitando (NaN/Inf >50%): {wav_path.name}")
                        removed += 1
                        continue
                    # Substitui NaN/Inf locais por silêncio, preserva o resto
                    y = np.where(finite, y, 0.0).astype(np.float32)

                duration = len(y) / TARGET_SR
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    removed += 1
                    continue

                # Checar silêncio
                rms = np.sqrt(np.mean(y**2))
                if not np.isfinite(rms) or rms < 10 ** (SILENCE_THRESHOLD_DB / 20):
                    removed += 1
                    continue

                peak = float(np.max(np.abs(y)))
                if peak > 1.0:
                    y = (y / peak * 0.999).astype(np.float32)

                # Validação final antes de gravar (defesa em profundidade)
                if not np.all(np.isfinite(y)):
                    logger.warning(f"  Rejeitando (NaN pós-processamento): {wav_path.name}")
                    removed += 1
                    continue

                sf.write(str(output_path), y, TARGET_SR, subtype="PCM_16")
                fixed += 1
                manifest[rel] = _file_sig(wav_path)

            except Exception as e:
                logger.warning(f"  Rejeitando arquivo corrompido: {wav_path.name} ({e})")
                removed += 1

    _save_norm_manifest(manifest)
    logger.info(
        f"\nCanonicalizacao completa: {fixed} escritos em {PROCESSED_DIR}, "
        f"{skipped} em cache, {removed} rejeitados; fontes preservadas"
    )


# ---------------------------------------------------------------------------
# Remover duplicatas
# ---------------------------------------------------------------------------
def _canonical_pcm_sha256(path: Path) -> str:
    """Hash decoded mono PCM, independent of WAV container metadata."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1)
    pcm16 = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype("<i2")
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{len(pcm16)}".encode("ascii"))
    digest.update(pcm16.tobytes())
    return digest.hexdigest()


def remove_duplicates():
    """Remove exact decoded-audio duplicates; abort cross-label conflicts."""
    logger.info("Verificando duplicatas por SHA-256 de PCM canonico...")
    hashes: dict[str, tuple[str, str]] = {}
    removed = 0

    for label, directory in [
        ("real", PROCESSED_REAL_DIR),
        ("fake", PROCESSED_FAKE_DIR),
    ]:
        for wav_path in sorted(directory.glob("*.wav")):
            h = _canonical_pcm_sha256(wav_path)
            if h in hashes:
                previous_label, previous_name = hashes[h]
                if previous_label != label:
                    raise RuntimeError(
                        "Conflito de rotulo: audio PCM identico em "
                        f"{previous_label}/{previous_name} e {label}/{wav_path.name}"
                    )
                logger.info(
                    "  Duplicata PCM: %s == %s", wav_path.name, previous_name
                )
                wav_path.unlink()
                removed += 1
            else:
                hashes[h] = (label, wav_path.name)

    logger.info("Duplicatas PCM removidas da camada processada: %d", removed)
    return removed


# ---------------------------------------------------------------------------
def _near_duplicate_fingerprint(path: Path) -> np.ndarray:
    """Compact spectral fingerprint robust to container and gain changes."""
    audio, sample_rate = librosa.load(
        str(path), sr=8_000, mono=True, duration=10.0, res_type=RESAMPLE_TYPE
    )
    if len(audio) < 512:
        raise ValueError(f"Audio curto demais para fingerprint: {path}")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=512, hop_length=160, n_mels=32
    )
    log_mel = librosa.power_to_db(mel + 1e-12, ref=np.max)
    sections = np.array_split(log_mel, 8, axis=1)
    pooled = np.concatenate([section.mean(axis=1) for section in sections])
    pooled = pooled - pooled.mean()
    norm = float(np.linalg.norm(pooled))
    return (pooled / max(norm, 1e-12)).astype("float32")


def audit_near_duplicates(
    cosine_distance_threshold: float = 0.002,
    max_neighbors: int = 5,
) -> dict:
    """Find perceptual near-duplicates without deleting ambiguous samples.

    Cross-label candidates abort the pipeline. Same-label candidates are kept
    and reported for curator review because automatic deletion could remove
    legitimate repeated phrases or speakers.
    """
    from sklearn.neighbors import NearestNeighbors

    records = [
        (label, path)
        for label, directory in (
            ("real", PROCESSED_REAL_DIR), ("fake", PROCESSED_FAKE_DIR)
        )
        for path in sorted(directory.glob("*.wav"))
    ]
    if len(records) < 2:
        return {"samples": len(records), "candidates": []}

    features = np.vstack(
        [_near_duplicate_fingerprint(path) for _label, path in records]
    )
    neighbors = min(max(2, int(max_neighbors)), len(records))
    model = NearestNeighbors(
        n_neighbors=neighbors, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(features)
    distances, indices = model.kneighbors(features)
    candidates = []
    seen_pairs = set()
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, right in zip(row_distances[1:], row_indices[1:]):
            pair = tuple(sorted((left, int(right))))
            if pair in seen_pairs or float(distance) > cosine_distance_threshold:
                continue
            seen_pairs.add(pair)
            left_label, left_path = records[pair[0]]
            right_label, right_path = records[pair[1]]
            candidates.append(
                {
                    "left": f"{left_label}/{left_path.name}",
                    "right": f"{right_label}/{right_path.name}",
                    "cosine_distance": float(distance),
                    "cross_label": left_label != right_label,
                }
            )

    report = {
        "method": "pooled_logmel_cosine_v1",
        "threshold": cosine_distance_threshold,
        "samples": len(records),
        "candidates": candidates,
    }
    report_path = PROCESSED_DIR / "near_duplicate_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    conflicts = [item for item in candidates if item["cross_label"]]
    if conflicts:
        raise RuntimeError(
            f"Quase-duplicatas entre rotulos detectadas: {len(conflicts)}; "
            f"consulte {report_path}"
        )
    logger.info("Auditoria near-duplicate: %d candidatos", len(candidates))
    return report


# Criar splits train/val/test
# ---------------------------------------------------------------------------
def _stratified_indices(files, labels, val_ratio, test_ratio):
    """Split estratificado por classe (default). Retorna (train, val, test) idx."""
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_ratio + test_ratio), random_state=42
    )
    train_idx, temp_idx = next(sss1.split(files, labels))
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=42)
    val_idx_local, test_idx_local = next(sss2.split(files[temp_idx], labels[temp_idx]))
    return train_idx, temp_idx[val_idx_local], temp_idx[test_idx_local]


def _speaker_disjoint_indices(files, labels, groups, val_ratio, test_ratio):
    """Split DISJUNTO POR FALANTE: nenhum falante em train e test ao mesmo tempo.

    Espelha benchmarks/data.py:_grouped_split — usa StratifiedGroupKFold para
    segurar um fold como teste e, dentro do restante, um fold como validacao.
    Cai para o split estratificado quando ha poucos grupos.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise ValueError(
            f"speaker_disjoint requires at least 3 explicit speakers; found {n_groups}"
        )

    idx = np.arange(len(labels))
    n_splits = max(2, min(round(1.0 / max(test_ratio, 1e-6)), n_groups))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    trainval_idx, test_idx = next(sgkf.split(idx, labels, groups))

    g_tv = groups[trainval_idx]
    if len(np.unique(g_tv)) >= 2:
        rel_val = val_ratio / (1.0 - test_ratio)
        inner = max(2, min(round(1.0 / max(rel_val, 1e-6)), len(np.unique(g_tv))))
        sgkf2 = StratifiedGroupKFold(n_splits=inner, shuffle=True, random_state=42)
        tr_rel, val_rel = next(sgkf2.split(trainval_idx, labels[trainval_idx], g_tv))
        return trainval_idx[tr_rel], trainval_idx[val_rel], test_idx

    raise ValueError(
        "speaker_disjoint cannot create validation without speaker overlap"
    )


def create_splits(
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    speaker_disjoint=False,
    expected_per_class: int | None = None,
    source_quotas: dict[str, dict[str, int]] | None = None,
):
    """Cria splits train/val/test.

    speaker_disjoint=True (tier `large`): mantem cada falante inteiramente em um
    unico conjunto, medindo generalizacao a USUARIOS NAO VISTOS. Default: split
    estratificado por classe.
    """
    logger.info("=" * 60)
    logger.info("CRIANDO SPLITS TRAIN/VAL/TEST"
                + (" (disjunto por falante)" if speaker_disjoint else ""))
    logger.info("=" * 60)

    # Coletar todos os arquivos
    files = []
    labels = []

    for wav_path in sorted(PROCESSED_REAL_DIR.glob("*.wav")):
        files.append(wav_path)
        labels.append(0)  # real

    for wav_path in sorted(PROCESSED_FAKE_DIR.glob("*.wav")):
        files.append(wav_path)
        labels.append(1)  # fake

    files = np.array(files)
    labels = np.array(labels)

    # Balance only after quality checks. The former flow reduced raw inputs to
    # exactly N first, so silence/duplicate rejection left no replacements.
    if source_quotas:
        selected: list[int] = []
        label_values = {"real": 0, "fake": 1}
        rng = np.random.default_rng(42)
        for class_name, quotas in source_quotas.items():
            label = label_values[class_name]
            for prefix, quota in quotas.items():
                candidates = np.array(
                    [
                        idx
                        for idx, path in enumerate(files)
                        if labels[idx] == label
                        and path.name.lower().startswith(f"{prefix.lower()}_")
                    ],
                    dtype=int,
                )
                if len(candidates) < quota:
                    raise RuntimeError(
                        "Reserva valida insuficiente apos qualidade/dedup: "
                        f"{class_name}:{prefix}={len(candidates)}, quota={quota}"
                    )
                selected.extend(
                    rng.choice(candidates, size=quota, replace=False).tolist()
                )
        selected_idx = np.asarray(selected, dtype=int)
        files = files[selected_idx]
        labels = labels[selected_idx]
        logger.info(
            "Selecao pos-validacao por quotas: %s",
            json.dumps(source_quotas, sort_keys=True),
        )

    logger.info(f"Total: {len(files)} ({sum(labels == 0)} real + {sum(labels == 1)} fake)")

    real_count = int(sum(labels == 0))
    fake_count = int(sum(labels == 1))
    if expected_per_class is not None and (
        real_count != expected_per_class or fake_count != expected_per_class
    ):
        raise RuntimeError(
            "Camada processada nao corresponde ao alvo apos validacao/dedup: "
            f"real={real_count}, fake={fake_count}, alvo={expected_per_class}"
        )
    if real_count != fake_count:
        raise RuntimeError(
            f"Classes desbalanceadas na camada processada: {real_count} != {fake_count}"
        )
    if len(files) < 10:
        raise RuntimeError("Muito poucos arquivos processados para criar splits")

    # Hierarquia anti-leakage: texto > enunciado > amostra unica. IDs ausentes
    # nunca sao inventados como um mesmo grupo coletivo.
    groups = None
    content_groups = None
    source_groups = None
    try:
        from app.domain.dataset_metadata.speaker_manifest import (
            MissingSampleMetadataError,
            sample_metadata_for_path,
            speaker_for_path,
        )

        groups = np.array(
            [speaker_for_path(f, strict=speaker_disjoint) for f in files],
            dtype=object,
        )
        content_values = []
        source_values = []
        for path in files:
            item = sample_metadata_for_path(path)
            source = str(item.get("source") or "unknown").lower()
            source_values.append(source)
            if item.get("text_id"):
                content_values.append(f"{source}:text:{item['text_id']}")
            elif item.get("utterance_id"):
                content_values.append(f"{source}:utt:{item['utterance_id']}")
            else:
                content_values.append(f"sample:{path.name}")
        content_groups = np.asarray(content_values, dtype=object)
        source_groups = np.asarray(source_values, dtype=object)
    except MissingSampleMetadataError:
        raise
    except Exception as exc:
        if speaker_disjoint:
            raise RuntimeError("Identificacao de falante indisponivel") from exc
        logger.warning("Metadados hierarquicos indisponiveis: %s", exc)

    strategy = "stratified"
    if speaker_disjoint and groups is not None:
        train_idx, val_idx, test_idx = _speaker_disjoint_indices(
            files, labels, groups, val_ratio, test_ratio
        )
        strategy = "speaker_disjoint"
    elif content_groups is not None and len(np.unique(content_groups)) < len(files):
        train_idx, val_idx, test_idx = _speaker_disjoint_indices(
            files, labels, content_groups, val_ratio, test_ratio
        )
        strategy = "content_disjoint"
    else:
        train_idx, val_idx, test_idx = _stratified_indices(
            files, labels, val_ratio, test_ratio
        )

    for split_name, indices in (
        ("train", train_idx), ("val", val_idx), ("test", test_idx)
    ):
        if set(np.unique(labels[indices]).tolist()) != {0, 1}:
            raise ValueError(
                f"{strategy} produced a single-class {split_name} split"
            )

    # Criar diretórios e copiar
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_real = SPLITS_DIR / split_name / "real"
        split_fake = SPLITS_DIR / split_name / "fake"
        split_real.mkdir(parents=True, exist_ok=True)
        split_fake.mkdir(parents=True, exist_ok=True)

        # Limpar existentes
        for f in split_real.glob("*.wav"):
            f.unlink()
        for f in split_fake.glob("*.wav"):
            f.unlink()

        for idx in indices:
            src = files[idx]
            if labels[idx] == 0:
                dst = split_real / src.name
            else:
                dst = split_fake / src.name
            shutil.copy2(str(src), str(dst))

        real_count = len(list(split_real.glob("*.wav")))
        fake_count = len(list(split_fake.glob("*.wav")))
        logger.info(f"  {split_name}: {real_count} real + {fake_count} fake = {real_count + fake_count}")

    # Salvar metadata
    metadata = {
        "total_files": len(files),
        "total_real": int(sum(labels == 0)),
        "total_fake": int(sum(labels == 1)),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "preprocessing_version": PREPROCESSING_VERSION,
        "resampling": RESAMPLE_TYPE,
        "loudness_policy": "preserve; attenuate only to prevent clipping",
        "raw_audio_immutable": True,
        "exact_duplicate_hash": "sha256(decoded_mono_pcm16)",
        "target_sr": TARGET_SR,
        "split_strategy": strategy,
    }
    if content_groups is not None:
        c_train = set(content_groups[train_idx].tolist())
        c_val = set(content_groups[val_idx].tolist())
        c_test = set(content_groups[test_idx].tolist())
        metadata["content_leakage"] = {
            "train_val": len(c_train & c_val),
            "train_test": len(c_train & c_test),
            "val_test": len(c_val & c_test),
        }
        if any(metadata["content_leakage"].values()):
            raise RuntimeError(
                f"Conteudo repetido entre splits: {metadata['content_leakage']}"
            )

    if source_groups is not None:
        source_counts = {}
        source_oracle_correct = 0
        for source in sorted(set(source_groups.tolist())):
            source_labels = labels[source_groups == source]
            real_n = int(np.sum(source_labels == 0))
            fake_n = int(np.sum(source_labels == 1))
            source_counts[str(source)] = {"real": real_n, "fake": fake_n}
            source_oracle_correct += max(real_n, fake_n)
        source_oracle_accuracy = source_oracle_correct / len(labels)
        metadata["source_class_audit"] = {
            "counts": source_counts,
            "majority_oracle_accuracy": source_oracle_accuracy,
            "confounded": bool(source_oracle_accuracy > 0.55),
        }
        if source_oracle_accuracy > 0.55:
            logger.warning(
                "Fonte prediz rotulo com acuracia %.2f%%; escopo valido: in-domain",
                source_oracle_accuracy * 100.0,
            )


    # Estatistica de falantes / usuarios nao vistos (quando ha grupos)
    if groups is not None:
        g_train = set(groups[train_idx].tolist())
        g_val = set(groups[val_idx].tolist())
        g_test = set(groups[test_idx].tolist())
        unseen = g_test - g_train - g_val
        metadata["speakers"] = {
            "total": int(len(set(groups.tolist()))),
            "identified": int(sum(1 for g in set(groups.tolist()) if ":" in str(g))),
            "train": len(g_train),
            "val": len(g_val),
            "test": len(g_test),
            "unseen_in_test": len(unseen),
            "leakage_overlap_train_test": len(g_train & g_test),
        }
        logger.info(
            "  Falantes: %d total · teste %d (não vistos: %d · vazamento train∩test: %d)",
            metadata["speakers"]["total"], len(g_test),
            len(unseen), len(g_train & g_test),
        )

    with open(SPLITS_DIR / "splits_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Splits salvos em {SPLITS_DIR}")
    logger.info(f"Metadata salvo em {SPLITS_DIR / 'splits_metadata.json'}")


# ---------------------------------------------------------------------------
# Criar ZIP para upload via UI
# ---------------------------------------------------------------------------
def create_training_zip(output_name: str = "dataset_pt_deepfake.zip"):
    """Cria ZIP com estrutura real/ + fake/ para upload na UI do XFakeSong."""
    logger.info("=" * 60)
    logger.info("CRIANDO ZIP PARA UPLOAD")
    logger.info("=" * 60)

    output_path = DATASETS_DIR / output_name

    real_files = sorted(PROCESSED_REAL_DIR.glob("*.wav"))
    fake_files = sorted(PROCESSED_FAKE_DIR.glob("*.wav"))

    logger.info(f"Empacotando {len(real_files)} real + {len(fake_files)} fake...")

    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for wav_path in real_files:
            zf.write(str(wav_path), f"real/{wav_path.name}")
        for wav_path in fake_files:
            zf.write(str(wav_path), f"fake/{wav_path.name}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ZIP criado: {output_path} ({size_mb:.1f} MB)")
    logger.info(f"Estrutura: real/ ({len(real_files)} arquivos) + fake/ ({len(fake_files)} arquivos)")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-processamento de dataset de audio")
    parser.add_argument("--validate", action="store_true", help="Validar integridade")
    parser.add_argument("--normalize", action="store_true", help="Normalizar todos os WAVs")
    parser.add_argument("--force-normalize", action="store_true",
                        help="Re-normaliza tudo, ignorando o cache (.normalized_manifest.json)")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remover duplicatas")
    parser.add_argument(
        "--audit-near-duplicates",
        action="store_true",
        help="Auditar quase-duplicatas por fingerprint log-Mel",
    )
    parser.add_argument("--create-splits", action="store_true", help="Criar splits train/val/test")
    parser.add_argument("--create-zip", action="store_true", help="Criar ZIP para upload na UI")
    parser.add_argument("--full", action="store_true", help="Executar pipeline completo")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Proporção de treino (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Proporção de validação (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Proporção de teste (default: 0.15)",
    )
    parser.add_argument(
        "--speaker-disjoint", action="store_true",
        help="Split disjunto por falante (tier large / usuários não vistos)",
    )
    parser.add_argument(
        "--expected-per-class",
        type=int,
        help="Falha se a camada processada nao tiver exatamente este total por classe",
    )
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        metavar="CLASSE:PREFIXO=TOTAL",
        help="Quota aplicada apos validacao/dedup; ex.: real:brspeech=3750",
    )

    args = parser.parse_args()

    if not any([args.validate, args.normalize, args.remove_duplicates,
                args.audit_near_duplicates, args.create_splits, args.create_zip, args.full]):
        parser.print_help()
        return

    # Validar ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.001:
        logger.error(f"Soma dos ratios deve ser 1.0 (atual: {total:.3f})")
        sys.exit(1)

    if args.full or args.validate:
        validate_dataset()

    if args.full or args.normalize:
        normalize_all(force=args.force_normalize)

    if args.full or args.remove_duplicates:
        remove_duplicates()

    if args.full or args.audit_near_duplicates:
        audit_near_duplicates()

    source_quotas: dict[str, dict[str, int]] = {}
    for spec in args.source_quota:
        try:
            class_prefix, raw_total = spec.split("=", 1)
            class_name, prefix = class_prefix.split(":", 1)
            class_name = class_name.strip().lower()
            prefix = prefix.strip().lower()
            total = int(raw_total)
        except (TypeError, ValueError):
            parser.error(f"quota invalida: {spec!r}")
        if class_name not in {"real", "fake"} or not prefix or total <= 0:
            parser.error(f"quota invalida: {spec!r}")
        source_quotas.setdefault(class_name, {})[prefix] = total

    if source_quotas and args.expected_per_class:
        for class_name in ("real", "fake"):
            if sum(source_quotas.get(class_name, {}).values()) != args.expected_per_class:
                parser.error(
                    f"quotas de {class_name} devem somar "
                    f"--expected-per-class={args.expected_per_class}"
                )

    if args.full or args.create_splits:
        create_splits(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            speaker_disjoint=args.speaker_disjoint,
            expected_per_class=args.expected_per_class,
            source_quotas=source_quotas or None,
        )

    if args.create_zip:
        create_training_zip()

    logger.info("\nPipeline concluido!")


if __name__ == "__main__":
    main()
