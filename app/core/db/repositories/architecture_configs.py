"""Hiperparâmetros salvos por arquitetura (``architecture_configs``).

A linha ``variant_name="default"`` de cada arquitetura é semeada com o
``default_params`` do registry na composição (``app.dependencies``) e
sobrescrita pela aba de hiperparâmetros do Gradio.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.db.models.architecture_config import ArchitectureConfig
from app.core.db.session import SessionLocal

logger = logging.getLogger(__name__)


class ArchitectureConfigRepository:
    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal

    def find_default(self, architecture: str) -> ArchitectureConfig | None:
        with self._session_factory() as db:
            return (
                db.query(ArchitectureConfig)
                .filter_by(architecture_name=architecture, variant_name="default")
                .first()
            )

    def save_default_parameters(
        self, architecture: str, parameters: dict[str, Any], *, description: str
    ) -> None:
        """Grava ``parameters`` na linha ``default``, criando-a se faltar.

        ``description`` só é usada na criação; uma linha existente mantém a
        sua.
        """
        with self._session_factory() as db:
            config = (
                db.query(ArchitectureConfig)
                .filter_by(architecture_name=architecture, variant_name="default")
                .first()
            )
            if config is None:
                db.add(
                    ArchitectureConfig(
                        architecture_name=architecture,
                        variant_name="default",
                        parameters=parameters,
                        description=description,
                    )
                )
            else:
                config.parameters = parameters
                flag_modified(config, "parameters")
            db.commit()

    def seed_defaults(
        self, defaults: Mapping[str, tuple[str | None, dict[str, Any]]]
    ) -> int:
        """Insere a linha ``default`` de cada arquitetura se a tabela estiver vazia.

        ``defaults`` mapeia nome → ``(descrição, parâmetros)``. Devolve quantas
        linhas foram inseridas: 0 quando já havia qualquer configuração.
        """
        with self._session_factory() as db:
            if db.query(ArchitectureConfig).first():
                return 0
            logger.info(
                "Populando banco de dados com configurações de arquitetura padrão..."
            )
            for name, (description, parameters) in defaults.items():
                db.add(
                    ArchitectureConfig(
                        architecture_name=name,
                        variant_name="default",
                        description=description,
                        parameters=parameters,
                        is_active=True,
                    )
                )
            db.commit()
        logger.info(f"{len(defaults)} arquiteturas semeadas com sucesso.")
        return len(defaults)
