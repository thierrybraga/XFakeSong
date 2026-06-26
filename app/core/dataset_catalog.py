"""Catalogo central dos datasets de audio usados pelo XFakeSong.

Este modulo evita divergencia entre interface Gradio, scripts de download,
documentacao e manifesto do benchmark. Os valores de duracao/falantes sao
metadados publicos ou estimativas operacionais do projeto; quando a fonte nao
documenta o valor de forma estavel, o campo fica explicito como "nao informado".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    source_type: str
    cli_flag: str
    prefixes: tuple[str, ...]
    description: str
    repository: str
    url: str
    license: str
    language: str
    classes: str
    audio_count: str
    duration: str
    speakers: str
    access: str
    benchmark_use: str
    recommended_for_full_benchmark: bool = False
    notes: str = ""


DATASET_CATALOG: Dict[str, DatasetInfo] = {
    "BRSpeech-DF": DatasetInfo(
        name="BRSpeech-DF",
        source_type="both",
        cli_flag="--brspeech",
        prefixes=("brspeech",),
        description="Corpus PT-BR com amostras bonafide e spoof.",
        repository="AKCIT-Deepfake/BRSpeech-DF",
        url="https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF",
        license="Apache-2.0/CC BY 4.0 (HF divergente)",
        language="pt-BR",
        classes="real + fake",
        audio_count="459.137 amostras",
        duration="nao informado pela fonte",
        speakers="nao informado pela fonte",
        access="Hugging Face publico",
        benchmark_use="Fonte principal PT-BR; boa base para treino balanceado.",
        recommended_for_full_benchmark=True,
        notes="Use com split por fonte/gerador quando possivel para reduzir vazamento.",
    ),
    "Fake Voices": DatasetInfo(
        name="Fake Voices",
        source_type="fake",
        cli_flag="--fake-voices",
        prefixes=("fkvoice", "fakevoice"),
        description="Vozes sinteticas XTTS organizadas por falante.",
        repository="unfake/fake_voices",
        url="https://huggingface.co/datasets/unfake/fake_voices",
        license="MIT",
        language="pt-BR",
        classes="fake",
        audio_count="ZIPs por falante; ~30.5 GB",
        duration="~140 h",
        speakers="101 falantes",
        access="Hugging Face publico",
        benchmark_use="Fake PT-BR independente para teste cross-generator.",
        recommended_for_full_benchmark=True,
        notes="O downloader usa max_speakers e estima cerca de 80 amostras uteis por falante.",
    ),
    "FLEURS": DatasetInfo(
        name="FLEURS",
        source_type="real",
        cli_flag="--fleurs",
        prefixes=("fleurs",),
        description="Fala real do Google FLEURS no subconjunto pt_br.",
        repository="google/fleurs",
        url="https://huggingface.co/datasets/google/fleurs",
        license="consultar README/LICENSE oficiais",
        language="pt-BR",
        classes="real",
        audio_count="pt_br ~4,1 mil linhas",
        duration="nao informado no catalogo local",
        speakers="nao informado no catalogo local",
        access="Hugging Face publico",
        benchmark_use="Reforco de fala real PT-BR.",
        recommended_for_full_benchmark=True,
    ),
    "CETUC": DatasetInfo(
        name="CETUC",
        source_type="real",
        cli_flag="--cetuc",
        prefixes=("cetuc",),
        description="Fallback PT-BR real via CETUC/Common Voice/OpenSLR.",
        repository="CETUC/CommonVoice/OpenSLR fallback",
        url="https://www.openslr.org/132/",
        license="livre/variavel conforme fonte fallback",
        language="pt-BR",
        classes="real",
        audio_count="variavel",
        duration="variavel",
        speakers="variavel",
        access="publico; pode usar fallback em cascata",
        benchmark_use="Completa deficit de amostras reais.",
        recommended_for_full_benchmark=True,
    ),
    "MLAAD-PT": DatasetInfo(
        name="MLAAD-PT",
        source_type="fake",
        cli_flag="--mlaad-pt",
        prefixes=("mlaad",),
        description="Subconjunto em portugues do Multi-Language Anti-spoofing.",
        repository="OU-CSAIL/MLAAD",
        url="https://huggingface.co/datasets/OU-CSAIL/MLAAD",
        license="CC-BY-NC 4.0",
        language="pt",
        classes="fake",
        audio_count="subconjunto PT filtrado em streaming",
        duration="nao informado para o subconjunto PT",
        speakers="nao informado",
        access="Hugging Face publico",
        benchmark_use="Reforco fake multilíngue/PT; uso condicionado a licenca NC.",
        recommended_for_full_benchmark=False,
    ),
    "Common Voice PT": DatasetInfo(
        name="Common Voice PT",
        source_type="real",
        cli_flag="--common-voice-pt",
        prefixes=("cvpt", "cv"),
        description="Mozilla Common Voice v17 em portugues.",
        repository="Mozilla Data Collective / common_voice_17_0 legacy",
        url="https://commonvoice.mozilla.org/datasets",
        license="CC0",
        language="pt",
        classes="real",
        audio_count="variavel por release/configuracao; HF v17 legacy vazio",
        duration="variavel por release/configuracao",
        speakers="variavel por release/configuracao",
        access="Mozilla Data Collective; HF v17 legacy vazio desde 2025",
        benchmark_use="Diversidade de fala real; bom para treino geral.",
        recommended_for_full_benchmark=True,
    ),
    "ASVspoof 2019": DatasetInfo(
        name="ASVspoof 2019",
        source_type="both",
        cli_flag="--asvspoof2019",
        prefixes=("asv2019",),
        description="Benchmark LA/PA para sintese, conversao de voz e replay.",
        repository="LanceaKing/asvspoof2019; oficial Edinburgh DataShare",
        url="https://datashare.ed.ac.uk/handle/10283/3336",
        license="ODC-BY 1.0",
        language="ingles",
        classes="real + fake",
        audio_count="protocolo oficial LA/PA",
        duration="nao consolidado no catalogo local",
        speakers="derivado do VCTK; consultar protocolo oficial",
        access="DataShare/HF mirror",
        benchmark_use="Referencia externa padrao anti-spoofing.",
        recommended_for_full_benchmark=True,
    ),
    "WaveFake": DatasetInfo(
        name="WaveFake",
        source_type="fake",
        cli_flag="--wavefake",
        prefixes=("wavefake",),
        description="Audios fake de 6 vocoders neurais.",
        repository="RUB-SysSec/WaveFake + Zenodo 5642694",
        url="https://zenodo.org/records/5642694",
        license="CC-BY-SA 4.0",
        language="ingles e japones",
        classes="fake",
        audio_count="conjunto grande; download completo ~28.9 GB",
        duration="~175 h",
        speakers="LJSpeech/JSUT; poucos falantes base",
        access="HF mirrors + Zenodo fallback",
        benchmark_use="Reforco fake internacional e cross-vocoder.",
        recommended_for_full_benchmark=True,
    ),
    "In-the-Wild": DatasetInfo(
        name="In-the-Wild",
        source_type="both",
        cli_flag="--in-the-wild",
        prefixes=("itw",),
        description="Audios reais e deepfakes de figuras publicas.",
        repository="deepfake-total.com/in_the_wild",
        url="https://deepfake-total.com/in_the_wild",
        license="Apache-2.0",
        language="majoritariamente ingles",
        classes="real + fake",
        audio_count="58 celebridades",
        duration="~20.8 h real + ~17.2 h fake",
        speakers="58 falantes/celebridades",
        access="download publico",
        benchmark_use="Validacao externa realista; bom para generalizacao.",
        recommended_for_full_benchmark=True,
    ),
    "ASVspoof 5": DatasetInfo(
        name="ASVspoof 5",
        source_type="both",
        cli_flag="--asvspoof5",
        prefixes=("asv5",),
        description="Challenge 2024 com ataques modernos TTS/VC/deepfake.",
        repository="jungjee/asvspoof5; ASVspoof 5 oficial",
        url="https://huggingface.co/datasets/jungjee/asvspoof5",
        license="CC BY 4.0",
        language="ingles/multifonte",
        classes="real + fake",
        audio_count="protocolo ASVspoof 5; mirror HF ~142 GB",
        duration="nao consolidado no catalogo local",
        speakers="~2.000 falantes",
        access="Hugging Face; verificar README/LICENSE oficiais",
        benchmark_use="Referencia externa moderna; usar para avaliacao final.",
        recommended_for_full_benchmark=False,
        notes="Pode ser pesado/gated; nao entra no preset padrao sem confirmacao.",
    ),
}


PRESET_SELECTIONS: Dict[str, List[str]] = {
    "PT-BR Rápido": ["BRSpeech-DF", "Fake Voices"],
    "PT-BR Completo": ["BRSpeech-DF", "Fake Voices", "CETUC", "MLAAD-PT"],
    "Internacional Padrão": ["ASVspoof 2019", "WaveFake", "In-the-Wild"],
    "Máxima Cobertura": [
        "BRSpeech-DF",
        "Fake Voices",
        "CETUC",
        "MLAAD-PT",
        "ASVspoof 2019",
        "WaveFake",
        "In-the-Wild",
    ],
    "Benchmark Robusto Recomendado": [
        "BRSpeech-DF",
        "Fake Voices",
        "FLEURS",
        "Common Voice PT",
        "ASVspoof 2019",
        "WaveFake",
        "In-the-Wild",
    ],
    "Só Reforçar Real": ["FLEURS", "CETUC", "Common Voice PT"],
    "Só Reforçar Fake": ["Fake Voices", "MLAAD-PT", "WaveFake", "ASVspoof 5"],
}


# ---------------------------------------------------------------------------
# Tiers de dataset (test / small / medium / large)
# ---------------------------------------------------------------------------
# Fonte unica de verdade para o TAMANHO/FINALIDADE de um dataset, consumida pela
# UI Gradio, por scripts/build_dataset.py, pelo benchmark e pela documentacao.
# Os limiares de modelos seguem a prontidao em
# app/interfaces/gradio/tabs/dataset_management.py (Classico >=300, CNN leve
# >=1000, CNN/RNN >=2000, Transformer >=4000, Ensemble >=6000 por classe).


@dataclass(frozen=True)
class DatasetTier:
    name: str
    per_class: int
    purpose: str
    description: str
    sources: tuple[str, ...]
    split: Dict[str, float]
    split_strategy: str  # "stratified" | "speaker_disjoint"
    speaker_aware: bool
    models_enabled: str

    @property
    def total(self) -> int:
        return self.per_class * 2

    @property
    def skip_real_cv(self) -> bool:
        """test/small dispensam CommonVoice/FLEURS (real vem do BRSpeech bonafide)."""
        return not any(
            s in self.sources for s in ("Common Voice PT", "FLEURS", "CETUC")
        )


_STD_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

DATASET_TIERS: Dict[str, DatasetTier] = {
    "test": DatasetTier(
        name="test",
        per_class=100,
        purpose="Smoke: validar que treino e modelo funcionam de ponta a ponta.",
        description=(
            "Conjunto minimo (100 por classe). Nao serve para medir desempenho — "
            "apenas confirma que download, pre-processamento, treino e inferencia "
            "rodam sem erro. Download em segundos."
        ),
        sources=("BRSpeech-DF", "Fake Voices"),
        split=_STD_SPLIT,
        split_strategy="stratified",
        speaker_aware=False,
        models_enabled="Nenhum tier de desempenho — abaixo do minimo classico (300).",
    ),
    "small": DatasetTier(
        name="small",
        per_class=1_000,
        purpose="Treino rapido para iteracao (Classico + CNN leve).",
        description=(
            "Dataset enxuto (1.000 por classe) para ciclos rapidos de treino. "
            "Habilita SVM/RandomForest e CNNs leves (RawNet2, Sonic Sleuth, "
            "MultiscaleCNN). PT-BR via BRSpeech-DF + Fake Voices."
        ),
        sources=("BRSpeech-DF", "Fake Voices"),
        split=_STD_SPLIT,
        split_strategy="stratified",
        speaker_aware=False,
        models_enabled="Classico + CNN leve (ate ~1.000/classe).",
    ),
    "medium": DatasetTier(
        name="medium",
        per_class=3_000,
        purpose="Treino e teste mais completos (ate Transformer).",
        description=(
            "Dataset completo de treino+teste (3.000 por classe) com diversidade "
            "real adicional (Common Voice PT + FLEURS) e fake independente "
            "(Fake Voices XTTS). Habilita CNN/RNN e Transformers."
        ),
        sources=("BRSpeech-DF", "Common Voice PT", "FLEURS", "Fake Voices"),
        split=_STD_SPLIT,
        split_strategy="stratified",
        speaker_aware=False,
        models_enabled="Ate Transformer (>=2.000–4.000/classe).",
    ),
    "large": DatasetTier(
        name="large",
        per_class=10_000,
        purpose=(
            "Dataset completo com falantes identificados e protocolo de "
            "usuarios nao vistos."
        ),
        description=(
            "Dataset completo (10.000 por classe) com identificacao de falante "
            "(speaker_manifest.json) e split DISJUNTO POR FALANTE: nenhum falante "
            "aparece em treino e teste ao mesmo tempo, medindo generalizacao a "
            "usuarios nao vistos. Habilita todas as 14 arquiteturas, incluindo o "
            "Ensemble. Combina com o protocolo cross-generator (XTTS=fkvoice)."
        ),
        sources=("BRSpeech-DF", "Common Voice PT", "FLEURS", "Fake Voices"),
        split=_STD_SPLIT,
        split_strategy="speaker_disjoint",
        speaker_aware=True,
        models_enabled="Todas as 14 arquiteturas, incluindo Ensemble (>=6.000/classe).",
    ),
}


def get_tier(name: str) -> Optional[DatasetTier]:
    return DATASET_TIERS.get(str(name).strip().lower())


def tier_choices() -> List[str]:
    return list(DATASET_TIERS.keys())


def tier_sources(name: str) -> List[str]:
    tier = get_tier(name)
    return list(tier.sources) if tier else []


def tier_reference_markdown() -> str:
    lines = [
        "| Tier | Por classe | Total | Finalidade | Fontes | Split | Falante |",
        "|---|---:|---:|---|---|---|:---:|",
    ]
    for tier in DATASET_TIERS.values():
        split = "disjunto por falante" if tier.speaker_aware else "70/15/15 estratificado"
        spk = "sim" if tier.speaker_aware else "—"
        lines.append(
            f"| **{tier.name}** | {tier.per_class:,} | {tier.total:,} | "
            f"{tier.purpose} | {', '.join(tier.sources)} | {split} | {spk} |"
        )
    return "\n".join(lines)


def source_type_map() -> Dict[str, str]:
    return {name: info.source_type for name, info in DATASET_CATALOG.items()}


def prefix_to_dataset() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name, info in DATASET_CATALOG.items():
        for prefix in info.prefixes:
            mapping[prefix.lower()] = name
    return mapping


def infer_dataset_from_path(path: str | Path) -> str:
    stem = Path(str(path).replace("\\", "/")).stem.lower()
    token = stem.split("_", 1)[0]
    return prefix_to_dataset().get(token, "Outros")


def infer_prefix_from_path(path: str | Path) -> str:
    stem = Path(str(path).replace("\\", "/")).stem.lower()
    return stem.split("_", 1)[0] if stem else "unknown"


def summarize_dataset_paths(paths: Iterable[str | Path], duration_sec: float | None = None) -> dict:
    counts: Dict[str, int] = {}
    for path in paths:
        name = infer_dataset_from_path(path)
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return {
        "total_samples": total,
        "estimated_window_seconds": duration_sec,
        "estimated_audio_hours": (
            round(total * float(duration_sec) / 3600.0, 4)
            if duration_sec is not None
            else None
        ),
        "sources": {
            name: {
                "samples": count,
                "type": DATASET_CATALOG.get(name, DatasetInfo(
                    name=name,
                    source_type="unknown",
                    cli_flag="",
                    prefixes=(),
                    description="Fonte nao catalogada",
                    repository="",
                    url="",
                    license="",
                    language="",
                    classes="",
                    audio_count="",
                    duration="",
                    speakers="",
                    access="",
                    benchmark_use="",
                )).source_type,
                "license": DATASET_CATALOG[name].license if name in DATASET_CATALOG else "",
            }
            for name, count in sorted(counts.items())
        },
    }


def dataset_reference_markdown() -> str:
    lines = [
        "| Dataset | Tipo | Idioma | Arquivos/Duração | Falantes | Licença | Uso no benchmark |",
        "|---|:---:|---|---|---|---|---|",
    ]
    for info in DATASET_CATALOG.values():
        lines.append(
            f"| **{info.name}** | `{info.source_type}` | {info.language} | "
            f"{info.audio_count}; {info.duration} | {info.speakers} | "
            f"{info.license} | {info.benchmark_use} |"
        )
    return "\n".join(lines)


def benchmark_dataset_markdown() -> str:
    lines = [
        "# Catalogo de datasets de audio para benchmark",
        "",
        "Esta tabela e gerada a partir de `app/core/dataset_catalog.py`, a mesma fonte usada pela interface Gradio.",
        "",
        dataset_reference_markdown(),
        "",
        "## Presets de download",
        "",
        "| Preset | Fontes |",
        "|---|---|",
    ]
    for name, sources in PRESET_SELECTIONS.items():
        lines.append(f"| {name} | {', '.join(sources)} |")
    return "\n".join(lines) + "\n"
