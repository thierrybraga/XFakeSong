"""Cobertura da i18n da UI Gradio (`app/interfaces/gradio/utils/i18n.py`).

Inclui um teste de **paridade** entre os dicionários PT/EN — atua como portão
de qualidade: toda chave nova precisa existir nos dois idiomas.
"""

import pytest

from app.interfaces.gradio.utils.i18n import (
    TRANSLATIONS,
    get_language,
    set_language,
    supported_languages,
    t,
)


@pytest.fixture(autouse=True)
def _restore_language():
    """Salva e restaura o idioma global (estado de módulo) entre testes."""
    prev = get_language()
    yield
    set_language(prev)


def test_default_language_supported():
    assert get_language() in supported_languages()


def test_set_language_valid():
    assert set_language("en") == "en"
    assert get_language() == "en"


def test_set_language_normalizes_case_and_locale():
    assert set_language("EN") == "en"
    assert set_language("pt-BR") == "pt"


def test_set_language_invalid_keeps_current():
    set_language("pt")
    assert set_language("xx") == "pt"
    assert get_language() == "pt"


def test_t_returns_translation_per_language():
    assert t("common.success", lang="pt") == "Sucesso"
    assert t("common.success", lang="en") == "Success"


def test_t_uses_global_language():
    set_language("en")
    assert t("common.error") == "Error"


def test_t_missing_key_returns_raw_key():
    assert t("nonexistent.key.zzz") == "nonexistent.key.zzz"


def test_t_cross_language_fallback():
    TRANSLATIONS["en"]["_test.only_en"] = "OnlyEN"
    try:
        # pedindo em PT, mas a chave só existe em EN → fallback para EN
        assert t("_test.only_en", lang="pt") == "OnlyEN"
    finally:
        TRANSLATIONS["en"].pop("_test.only_en", None)


def test_t_format_kwargs_ignored_when_no_placeholder():
    assert t("common.save", lang="pt", anything=1) == "Salvar"


def test_supported_languages():
    assert supported_languages() == ("pt", "en")


def test_translation_dicts_have_parity():
    pt_keys = set(TRANSLATIONS["pt"])
    en_keys = set(TRANSLATIONS["en"])
    assert pt_keys == en_keys, (
        f"Chaves só em PT: {pt_keys - en_keys}; só em EN: {en_keys - pt_keys}"
    )
