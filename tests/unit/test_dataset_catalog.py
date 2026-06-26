from app.core.dataset_catalog import (
    DATASET_CATALOG,
    PRESET_SELECTIONS,
    infer_dataset_from_path,
    infer_prefix_from_path,
    summarize_dataset_paths,
)


def test_gradio_download_catalog_has_all_expected_sources():
    expected = {
        "BRSpeech-DF",
        "Fake Voices",
        "FLEURS",
        "CETUC",
        "MLAAD-PT",
        "Common Voice PT",
        "ASVspoof 2019",
        "WaveFake",
        "In-the-Wild",
        "ASVspoof 5",
    }

    assert set(DATASET_CATALOG) == expected
    assert DATASET_CATALOG["Fake Voices"].duration == "~140 h"
    assert DATASET_CATALOG["Fake Voices"].speakers == "101 falantes"
    assert DATASET_CATALOG["In-the-Wild"].speakers == "58 falantes/celebridades"
    assert "Benchmark Robusto Recomendado" in PRESET_SELECTIONS


def test_dataset_prefix_inference_keeps_machine_group_and_human_summary():
    paths = [
        "app/datasets/splits/train/fake/fkvoice_00001.wav",
        "app/datasets/splits/train/real/brspeech_00001.wav",
        "app/datasets/splits/train/fake/asv2019_00001.wav",
    ]

    assert infer_prefix_from_path(paths[0]) == "fkvoice"
    assert infer_dataset_from_path(paths[0]) == "Fake Voices"

    summary = summarize_dataset_paths(paths, duration_sec=5.0)
    assert summary["total_samples"] == 3
    assert summary["estimated_audio_hours"] == round(15.0 / 3600.0, 4)
    assert summary["sources"]["Fake Voices"]["samples"] == 1
    assert summary["sources"]["BRSpeech-DF"]["samples"] == 1
    assert summary["sources"]["ASVspoof 2019"]["samples"] == 1
