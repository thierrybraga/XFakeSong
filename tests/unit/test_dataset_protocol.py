"""Regressoes do Protocolo de Dataset (CETUC pareado com clones XTTS-v2).

As garantias do protocolo sao estruturais: valem por causa da forma da particao e
da regra de selecao, nao por causa dos numeros de um artefato especifico. Estes
testes exercitam essa estrutura sem precisar do corpus em disco.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from scripts.dataset.audit_paired_corpus import (
    auc,
    audit_balance,
    audit_disjointness,
    majority_oracle,
)
from scripts.dataset.build_paired_pt_corpus import (
    TARGET_RMS_DB,
    _normalize_loudness,
    extract_window,
    normalize_text,
    text_id_for,
)
from scripts.dataset.build_paired_splits import (
    assign,
    content_groups,
    duplicate_audio_utterances,
    sentence_partition,
    short_utterances,
    speaker_partition,
)
from scripts.dataset.export_paired_npz import (
    _enforce,
    derive_max_pairs,
    select_pairs,
)

SPLIT_NAMES = ("train", "val", "test")


def _record(
    speaker: str,
    sentence: str,
    label: int,
    official: str = "train",
    duration: float = 4.0,
) -> dict:
    """Uma amostra do corpus pareado, com a proveniencia que o protocolo exige."""
    generator = "bonafide" if label == 0 else "xtts_v2"
    side = "real" if label == 0 else "fake"
    return {
        "path": f"{side}/{speaker}/ptpair_{speaker}_{sentence}_{generator}.wav",
        "class": side,
        "label": label,
        "source": "ptpair",
        "speaker_id": speaker,
        "cetuc_official_split": official,
        "utterance_id": f"{speaker}-{sentence}",
        "sentence_index": sentence,
        "text_id": f"t{sentence}",
        "generator_id": generator,
        "content_sha256": f"{speaker}{sentence}{label}".ljust(64, "0"),
        "duration_sec": duration,
    }


def _corpus(speakers: dict[str, str], sentences: list[str]) -> list[dict]:
    """Grid completo locutor x frase x classe."""
    return [
        _record(speaker, sentence, label, official)
        for speaker, official in speakers.items()
        for sentence in sentences
        for label in (0, 1)
    ]


def _speaker_corpus(n_female: int, n_male: int) -> list[dict]:
    speakers = {f"F{i:03d}": "train" for i in range(n_female)}
    speakers |= {f"M{i:03d}": "train" for i in range(n_male)}
    return _corpus(speakers, ["0000"])


# ---------------------------------------------------------------------------
# Identidade de texto
# ---------------------------------------------------------------------------


def test_text_id_ignores_case_and_punctuation_but_not_words() -> None:
    assert text_id_for("Pesquisa e uma coisa!") == text_id_for(
        "  pesquisa   e uma coisa  "
    )
    assert text_id_for("pesquisa e uma coisa") != text_id_for("pesquisa e outra coisa")


def test_text_normalization_preserves_accents() -> None:
    """Acentos sao fonemicamente relevantes: nao sao a mesma frase."""
    assert normalize_text("nao") != normalize_text("não")


# ---------------------------------------------------------------------------
# Particao de locutores
# ---------------------------------------------------------------------------


def test_speaker_partition_never_repeats_a_speaker_across_splits() -> None:
    records = _speaker_corpus(28, 28)
    assignment = speaker_partition(records, "stratified", (0.6, 0.2, 0.2), seed=42)

    assert len(assignment) == 56
    by_split: dict[str, set[str]] = defaultdict(set)
    for speaker, split in assignment.items():
        by_split[split].add(speaker)
    assert by_split["train"] & by_split["val"] == set()
    assert by_split["train"] & by_split["test"] == set()
    assert by_split["val"] & by_split["test"] == set()
    assert sum(len(group) for group in by_split.values()) == 56


def test_speaker_partition_balances_sex_across_splits() -> None:
    """Com 11 locutores no teste, um sorteio simples pode dar 11 de um sexo so."""
    records = _speaker_corpus(28, 28)
    assignment = speaker_partition(records, "stratified", (0.6, 0.2, 0.2), seed=42)

    for split in SPLIT_NAMES:
        group = [s for s, v in assignment.items() if v == split]
        female = sum(1 for s in group if s.startswith("F"))
        assert female == len(group) - female, (split, group)


def test_speaker_partition_is_deterministic_and_seed_sensitive() -> None:
    records = _speaker_corpus(28, 28)
    first = speaker_partition(records, "stratified", (0.6, 0.2, 0.2), seed=42)
    assert first == speaker_partition(records, "stratified", (0.6, 0.2, 0.2), seed=42)
    assert first != speaker_partition(records, "stratified", (0.6, 0.2, 0.2), seed=7)


def test_speaker_partition_keeps_every_split_populated_when_tiny() -> None:
    """Fracoes agressivas num grupo pequeno nao podem esvaziar o treino."""
    records = _speaker_corpus(3, 3)
    assignment = speaker_partition(records, "stratified", (0.2, 0.4, 0.4), seed=42)
    counts = Counter(assignment.values())
    assert counts["train"] >= 1 and counts["val"] >= 1 and counts["test"] >= 1


def test_official_strategy_uses_the_cetuc_partition() -> None:
    records = _corpus({"M001": "train", "M002": "dev", "M003": "test"}, ["0000"])
    assignment = speaker_partition(records, "official", (0.6, 0.2, 0.2), seed=42)
    assert assignment == {"M001": "train", "M002": "val", "M003": "test"}


def test_unknown_strategy_fails_loudly() -> None:
    records = _speaker_corpus(2, 2)
    with pytest.raises(SystemExit, match="estrategia de locutor desconhecida"):
        speaker_partition(records, "aleatoria", (0.6, 0.2, 0.2), seed=42)


def test_assign_uses_the_given_speaker_partition_not_the_official_one() -> None:
    """Regressao: `assign` precisa honrar a particao recebida.

    Uma versao anterior comparava o dicionario inteiro em vez do split do
    locutor, o que gravava o dicionario como valor de `split`.
    """
    records = _corpus({"M001": "train", "M002": "dev"}, ["0000"])
    assign(records, {"0000": "test"}, {"M001": "test", "M002": "test"})
    assert {r["split"] for r in records} == {"test"}


# ---------------------------------------------------------------------------
# Particao de frases e grupos de conteudo
# ---------------------------------------------------------------------------


def test_sentence_partition_is_deterministic_and_disjoint() -> None:
    records = _corpus({"M001": "train"}, [f"{i:04d}" for i in range(100)])

    first = sentence_partition(records, (60, 20, 20), seed=42)
    second = sentence_partition(records, (60, 20, 20), seed=42)
    other = sentence_partition(records, (60, 20, 20), seed=7)

    assert first == second
    assert first != other
    for split, size in (("train", 60), ("val", 20), ("test", 20)):
        assert sum(1 for v in first.values() if v == split) == size
    assert len(first) == 100


def test_sentence_partition_refuses_to_overcommit() -> None:
    records = _corpus({"M001": "train"}, [f"{i:04d}" for i in range(10)])
    with pytest.raises(SystemExit, match="o corpus tem 10"):
        sentence_partition(records, (8, 4, 4), seed=42)


def test_repeated_text_in_two_slots_stays_in_one_group() -> None:
    """Medido no CETUC: 3 textos aparecem em dois slots distintos.

    Particionar por indice deixaria o mesmo texto em treino e teste.
    """
    records = [_record("M001", slot, 0) for slot in ("0000", "0007", "0009", "0011")]
    records[1]["text_id"] = records[0]["text_id"]

    groups = content_groups(records)
    assert groups["0000"] == groups["0007"]
    assert groups["0009"] != groups["0000"]
    assert len(set(groups.values())) == 3

    split = sentence_partition(records, (1, 1, 1), seed=42)
    assert split["0000"] == split["0007"]


def test_divergent_text_in_one_slot_stays_in_one_group() -> None:
    """O caso inverso: um locutor divergindo do texto canonico do slot."""
    records = [_record("M001", "0000", 0), _record("M002", "0000", 0)]
    records[1]["text_id"] = "tdivergente"
    assert len(set(content_groups(records).values())) == 1


# ---------------------------------------------------------------------------
# Exclusoes: duplicata da fonte e janela
# ---------------------------------------------------------------------------


def test_duplicated_source_audio_is_excluded_from_both_classes() -> None:
    """Gravacao repetida do CETUC: mesmo audio em dois slots, textos diferentes.

    Uma das duas transcricoes esta errada e nao da para saber qual, entao as duas
    saem — inclusive os clones correspondentes.
    """
    records = _corpus({"M001": "train"}, ["0000", "0001", "0002"])
    shared = "duplicado".ljust(64, "0")
    for record in records:
        if record["sentence_index"] in ("0000", "0001") and record["label"] == 0:
            record["content_sha256"] = shared

    assert duplicate_audio_utterances(records) == {"M001-0000", "M001-0001"}

    assign(records, {"0000": "train", "0001": "train", "0002": "train"})
    excluded = [r for r in records if r["split"] == "excluded_duplicate_audio"]
    assert len(excluded) == 4
    assert {r["class"] for r in excluded} == {"real", "fake"}
    assert [r["split"] for r in records if r["sentence_index"] == "0002"] == [
        "train",
        "train",
    ]


def test_pair_shorter_than_the_window_is_dropped_whole() -> None:
    """O clone e mais curto que o original; sem isto, a repeticao (`tile`) seria
    assimetrica entre as classes e viraria pista de classe."""
    records = _corpus({"M001": "train"}, ["0000", "0001"])
    for record in records:
        if record["sentence_index"] == "0000" and record["label"] == 1:
            record["duration_sec"] = 2.4  # so o clone e curto

    assert short_utterances(records, 3.0) == {"M001-0000"}

    assign(records, {"0000": "train", "0001": "train"}, min_duration=3.0)
    dropped = [r for r in records if r["split"] == "excluded_shorter_than_window"]
    assert len(dropped) == 2  # o par inteiro, nao so o lado curto
    assert {r["class"] for r in dropped} == {"real", "fake"}


def test_no_minimum_duration_keeps_everything() -> None:
    records = _corpus({"M001": "train"}, ["0000"])
    records[0]["duration_sec"] = 0.5
    assign(records, {"0000": "train"}, min_duration=0.0)
    assert {r["split"] for r in records} == {"train"}


# ---------------------------------------------------------------------------
# Bloco diagonal
# ---------------------------------------------------------------------------


def test_assignment_is_block_diagonal_and_excludes_the_rest() -> None:
    speakers = {"M001": "train", "M002": "dev", "M003": "test"}
    sentences = ["0000", "0001", "0002"]
    records = _corpus(speakers, sentences)
    sentence_split = {"0000": "train", "0001": "val", "0002": "test"}
    speaker_split = {"M001": "train", "M002": "val", "M003": "test"}

    assign(records, sentence_split, speaker_split)

    for record in records:
        expected = speaker_split[record["speaker_id"]]
        if expected == sentence_split[record["sentence_index"]]:
            assert record["split"] == expected
        else:
            assert record["split"].startswith("excluded_offdiagonal")

    used = [r for r in records if r["split"] in SPLIT_NAMES]
    # 3 locutores x 3 frases x 2 classes = 18; so a diagonal (3 celulas) sobra.
    assert len(used) == 6


def test_double_disjointness_holds_and_is_detected_when_broken() -> None:
    speakers = {"M001": "train", "M002": "train", "M003": "dev", "M004": "test"}
    sentences = [f"{i:04d}" for i in range(9)]
    records = _corpus(speakers, sentences)
    sentence_split = {
        s: ("train" if i < 3 else "val" if i < 6 else "test")
        for i, s in enumerate(sentences)
    }
    speaker_split = {"M001": "train", "M002": "train", "M003": "val", "M004": "test"}
    assign(records, sentence_split, speaker_split)

    report = audit_disjointness(records)
    assert report["ok"], report
    for field in ("speaker_id", "sentence_index", "text_id", "content_sha256"):
        overlaps = {
            k: v
            for k, v in report[field]["overlaps"].items()
            if not k.endswith("examples")
        }
        assert set(overlaps.values()) == {0}

    # Move uma amostra de teste para o treino: a auditoria tem de acusar.
    for record in records:
        if record["split"] == "test":
            record["split"] = "train"
            break
    broken = audit_disjointness(records)
    assert not broken["ok"]
    assert not broken["speaker_id"]["ok"] or not broken["sentence_index"]["ok"]


# ---------------------------------------------------------------------------
# Balanceamento e atalhos de metadado
# ---------------------------------------------------------------------------


def test_pairing_gives_exact_balance_per_split_and_per_speaker() -> None:
    speakers = {"M001": "train", "M002": "dev", "M003": "test"}
    records = _corpus(speakers, ["0000", "0001", "0002"])
    assign(
        records,
        {"0000": "train", "0001": "val", "0002": "test"},
        {"M001": "train", "M002": "val", "M003": "test"},
    )

    report = audit_balance(records)
    assert report["ok"], report
    for split in SPLIT_NAMES:
        assert report[split]["real"] == report[split]["fake"]
        assert report[split]["speakers_unbalanced"] == 0


def test_unbalanced_speaker_is_detected() -> None:
    records = _corpus({"M001": "train"}, ["0000", "0001"])
    assign(records, {"0000": "train", "0001": "train"}, {"M001": "train"})
    records = [
        r for r in records if not (r["label"] == 1 and r["sentence_index"] == "0001")
    ]

    report = audit_balance(records)
    assert not report["ok"]
    assert report["train"]["speakers_unbalanced"] == 1


def test_speaker_and_text_do_not_predict_the_class() -> None:
    """O ponto central do protocolo: as duas identidades ficam no acaso."""
    records = _corpus({"M001": "train", "M002": "train"}, ["0000", "0001"])
    for field in ("source", "speaker_id", "sentence_index", "text_id"):
        accuracy, _ = majority_oracle(records, field)
        assert accuracy == pytest.approx(0.5)
    # `generator_id` DEFINE o rotulo — 100% por construcao, nao e atalho.
    accuracy, _ = majority_oracle(records, "generator_id")
    assert accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Normalizacao de nivel
# ---------------------------------------------------------------------------


def _rms_db(x: np.ndarray) -> float:
    return 20 * float(np.log10(np.sqrt((x**2).mean())))


def test_loudness_normalization_equalizes_levels_across_classes() -> None:
    """Reproduz o atalho medido: clones ~11 dB mais altos que o bonafide."""
    rng = np.random.default_rng(0)
    quiet = rng.normal(scale=0.03, size=16000).astype("float32")
    loud = rng.normal(scale=0.12, size=16000).astype("float32")

    assert _rms_db(loud) - _rms_db(quiet) > 8  # o atalho existe antes

    normalized_quiet, info_quiet = _normalize_loudness(quiet)
    normalized_loud, info_loud = _normalize_loudness(loud)

    assert _rms_db(normalized_quiet) == pytest.approx(TARGET_RMS_DB, abs=0.01)
    assert _rms_db(normalized_loud) == pytest.approx(TARGET_RMS_DB, abs=0.01)
    # O ganho aplicado fica registrado, entao a operacao e reversivel.
    assert info_quiet["applied_gain_db"] > info_loud["applied_gain_db"]


def test_loudness_normalization_never_clips() -> None:
    """Um sinal de crista alta seria estourado sem o teto de pico."""
    spiky = np.full(16000, 0.001, dtype="float32")
    spiky[100] = 0.9
    normalized, info = _normalize_loudness(spiky)
    assert float(np.abs(normalized).max()) <= 1.0
    assert info["peak_limited"] is True


def test_silent_audio_is_left_untouched() -> None:
    normalized, info = _normalize_loudness(np.zeros(1000, dtype="float32"))
    assert not np.any(normalized)
    assert info["applied_gain_db"] == 0.0


def test_window_is_normalized_not_just_the_file() -> None:
    """Normalizar o arquivo nao normaliza a janela.

    Reproduz o residuo medido (AUC 0,71): dois arquivos no mesmo RMS, um com
    silencio nas pontas, tem mioloes de energias diferentes. E o miolo que o
    modelo recebe.
    """
    rng = np.random.default_rng(0)
    speech = rng.normal(scale=0.05, size=48000).astype("float32")
    com_silencio = np.concatenate(
        [np.zeros(16000, dtype="float32"), speech, np.zeros(16000, dtype="float32")]
    )
    sem_silencio = np.concatenate([speech, speech]).astype("float32")[:80000]

    a, _ = _normalize_loudness(com_silencio)
    b, _ = _normalize_loudness(sem_silencio)

    # Os arquivos ficam no mesmo nivel...
    assert _rms_db(a) == pytest.approx(_rms_db(b), abs=0.01)
    # ...mas os miolos brutos, nao.
    span = 48000
    raw_a = a[(len(a) - span) // 2 :][:span]
    raw_b = b[(len(b) - span) // 2 :][:span]
    assert abs(_rms_db(raw_a) - _rms_db(raw_b)) > 0.5

    # `extract_window` nivela a janela, que e o que chega ao modelo.
    window_a, _, _ = extract_window(a, span)
    window_b, _, _ = extract_window(b, span)
    assert _rms_db(window_a) == pytest.approx(TARGET_RMS_DB, abs=0.01)
    assert _rms_db(window_b) == pytest.approx(TARGET_RMS_DB, abs=0.01)


def test_window_is_a_pure_center_crop_when_long_enough() -> None:
    audio = np.arange(100, dtype="float32") / 100.0
    window, original, start = extract_window(audio, 40)
    assert original == 100
    assert start == 30
    assert len(window) == 40


# ---------------------------------------------------------------------------
# Selecao com limite de tamanho
# ---------------------------------------------------------------------------


def test_cap_cuts_pairs_never_single_samples() -> None:
    speakers = {"M001": "train", "M002": "train", "M003": "train"}
    records = _corpus(speakers, [f"{i:04d}" for i in range(10)])

    kept = select_pairs(records, max_pairs=12, seed=42)

    assert len(kept) == 24
    assert sum(1 for r in kept if r["label"] == 0) == 12
    assert sum(1 for r in kept if r["label"] == 1) == 12
    by_utterance: dict[str, set[int]] = defaultdict(set)
    for record in kept:
        by_utterance[record["utterance_id"]].add(record["label"])
    assert all(labels == {0, 1} for labels in by_utterance.values())


def test_cap_spreads_across_speakers_instead_of_dropping_them() -> None:
    records = _corpus({f"M{i:03d}": "train" for i in range(4)}, [f"{i:04d}" for i in range(10)])

    kept = select_pairs(records, max_pairs=8, seed=42)

    per_speaker: Counter = Counter(r["speaker_id"] for r in kept)
    assert len(per_speaker) == 4, per_speaker
    assert set(per_speaker.values()) == {4}  # 2 pares por locutor


def test_cap_is_deterministic() -> None:
    records = _corpus(
        {"M001": "train", "M002": "train"}, [f"{i:04d}" for i in range(12)]
    )
    first = [r["path"] for r in select_pairs(records, max_pairs=10, seed=42)]
    second = [r["path"] for r in select_pairs(records, max_pairs=10, seed=42)]
    assert first == second


def test_no_cap_keeps_everything() -> None:
    records = _corpus({"M001": "train"}, ["0000", "0001"])
    assert len(select_pairs(records, max_pairs=0, seed=42)) == len(records)


# ---------------------------------------------------------------------------
# Barreira do exportador
# ---------------------------------------------------------------------------


def _clean_audits() -> dict:
    return {
        f"{field}_overlap": {"trainxval": 0, "trainxtest": 0, "valxtest": 0}
        for field in ("speaker_id", "sentence_index", "text_id", "content_sha256")
    } | {
        "class_balance": {s: {"real": 10, "fake": 10} for s in SPLIT_NAMES},
        "majority_oracle_accuracy": {"source": 0.5, "speaker_id": 0.5, "text_id": 0.5},
    }


def test_export_gate_passes_a_clean_partition() -> None:
    _enforce(_clean_audits())


def test_export_gate_blocks_a_shared_speaker() -> None:
    audits = _clean_audits()
    audits["speaker_id_overlap"]["trainxtest"] = 1
    with pytest.raises(SystemExit, match="speaker_id compartilhado entre trainxtest"):
        _enforce(audits)


def test_export_gate_blocks_a_repeated_sample() -> None:
    audits = _clean_audits()
    audits["content_sha256_overlap"]["valxtest"] = 3
    with pytest.raises(SystemExit, match="content_sha256 compartilhado"):
        _enforce(audits)


def test_export_gate_blocks_class_imbalance() -> None:
    audits = _clean_audits()
    audits["class_balance"]["test"] = {"real": 12, "fake": 8}
    with pytest.raises(SystemExit, match="test desbalanceado"):
        _enforce(audits)


def test_export_gate_blocks_a_metadata_shortcut() -> None:
    audits = _clean_audits()
    audits["majority_oracle_accuracy"]["speaker_id"] = 0.93
    with pytest.raises(SystemExit, match="oraculo por speaker_id"):
        _enforce(audits)


# ---------------------------------------------------------------------------
# AUC da auditoria de confundidores
# ---------------------------------------------------------------------------


def test_auc_is_symmetric_and_bounded() -> None:
    separated = auc(np.arange(10.0, 20.0), np.arange(0.0, 10.0))
    inverted = auc(np.arange(0.0, 10.0), np.arange(10.0, 20.0))
    assert separated == pytest.approx(1.0)
    # Separacao ao contrario separa igualmente bem: a metrica e simetrizada.
    assert inverted == pytest.approx(1.0)


def test_auc_of_identical_distributions_is_chance() -> None:
    rng = np.random.default_rng(0)
    assert auc(rng.normal(size=500), rng.normal(size=500)) == pytest.approx(
        0.5, abs=0.06
    )


# ---------------------------------------------------------------------------
# Rateio de tamanho (--target-samples)
# ---------------------------------------------------------------------------
# As variantes do dataset (completa e reduzida) saem do MESMO assignment; muda
# so a densidade de enunciados por celula. O rateio precisa entao ser derivado
# da particao, somar exato e nunca esvaziar uma particao.


def _fake_records(por_split: dict[str, int]) -> list[dict]:
    """Um registro por amostra, dois por enunciado (bonafide + clone)."""
    registros = []
    for split, pares in por_split.items():
        for i in range(pares):
            uid = f"{split}_{i:05d}"
            for gerador in ("bonafide", "xtts_v2"):
                registros.append(
                    {
                        "split": split,
                        "utterance_id": uid,
                        "speaker_id": f"S{i % 7:02d}",
                        "generator": gerador,
                    }
                )
    return registros


def test_target_samples_soma_exatamente_o_alvo() -> None:
    registros = _fake_records({"train": 16613, "val": 1988, "test": 1889})
    for alvo in (15000, 8000, 30000, 40980, 60):
        rateio = derive_max_pairs(registros, alvo)
        assert sum(rateio.values()) * 2 == alvo, alvo


def test_target_samples_preserva_a_proporcao_da_particao() -> None:
    disponivel = {"train": 16613, "val": 1988, "test": 1889}
    rateio = derive_max_pairs(_fake_records(disponivel), 15000)
    total = sum(disponivel.values())
    for split, pares in rateio.items():
        esperado = 7500 * disponivel[split] / total
        # Maior resto nunca desvia mais de um par inteiro da cota exata.
        assert abs(pares - esperado) < 1.0, (split, pares, esperado)


def test_target_igual_ao_disponivel_reproduz_a_particao_inteira() -> None:
    disponivel = {"train": 16613, "val": 1988, "test": 1889}
    rateio = derive_max_pairs(_fake_records(disponivel), sum(disponivel.values()) * 2)
    assert rateio == disponivel


def test_target_impar_e_recusado() -> None:
    # Cada par gera DUAS amostras; um alvo impar nao e realizavel.
    with pytest.raises(SystemExit):
        derive_max_pairs(_fake_records({"train": 10, "val": 4, "test": 4}), 15)


def test_target_acima_do_disponivel_e_recusado() -> None:
    with pytest.raises(SystemExit):
        derive_max_pairs(_fake_records({"train": 10, "val": 4, "test": 4}), 100)


def test_target_que_esvaziaria_uma_particao_e_recusado() -> None:
    # 2 amostras = 1 par: iria inteiro para o treino e val/teste ficariam vazios.
    with pytest.raises(SystemExit):
        derive_max_pairs(_fake_records({"train": 16613, "val": 1988, "test": 1889}), 2)
