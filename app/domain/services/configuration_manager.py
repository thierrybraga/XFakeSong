"""Gerenciador de configurações centralizadas.

Este módulo implementa um sistema centralizado de configurações para todos
os componentes do pipeline, permitindo configuração flexível e validação.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import jsonschema
from datetime import datetime
import os
from copy import deepcopy

from app.domain.interfaces.pipeline_interfaces import (
    IConfigurationManager, ComponentType, ConfigurationError
)

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Formatos de configuração suportados."""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


@dataclass
class ConfigSchema:
    """Schema de configuração para validação."""
    component_type: ComponentType
    component_name: str
    schema: Dict[str, Any]
    version: str = "1.0.0"
    description: str = ""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Valida configuração contra o schema.

        Args:
            config: Configuração a validar

        Returns:
            Lista de erros de validação
        """
        errors = []

        try:
            jsonschema.validate(config, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Erro de validação: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Erro no schema: {e.message}")

        # Verificar campos obrigatórios
        for field_name in self.required_fields:
            if field_name not in config:
                errors.append(f"Campo obrigatório ausente: {field_name}")

        return errors


@dataclass
class ConfigProfile:
    """Perfil de configuração."""
    name: str
    description: str
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigProfile':
        """Cria instância a partir de dicionário."""
        return cls(
            name=data["name"],
            description=data["description"],
            config=data["config"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", [])
        )


class ConfigurationManager(IConfigurationManager):
    """Gerenciador de configurações centralizadas."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("app/config")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Diretórios específicos
        self.schemas_dir = self.config_dir / "schemas"
        self.profiles_dir = self.config_dir / "profiles"
        self.components_dir = self.config_dir / "components"

        for dir_path in [self.schemas_dir,
                         self.profiles_dir, self.components_dir]:
            dir_path.mkdir(exist_ok=True)

        # Caches
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, ConfigSchema] = {}
        self._profile_cache: Dict[str, ConfigProfile] = {}

        # Configuração principal
        self.main_config_file = self.config_dir / "main_config.yaml"
        self._main_config: Dict[str, Any] = {}

        # Variáveis de ambiente
        self._env_prefix = "DEEPFAKE_"

        self._load_main_config()
        self._load_schemas()
        self._load_profiles()

    def _load_main_config(self) -> None:
        """Carrega configuração principal."""
        if self.main_config_file.exists():
            try:
                self._main_config = self.load_config(self.main_config_file)
                logger.info("Configuração principal carregada")
            except Exception as e:
                logger.error(f"Erro ao carregar configuração principal: {e}")
                self._main_config = self._get_default_main_config()
        else:
            self._main_config = self._get_default_main_config()
            self.save_config(self._main_config, self.main_config_file)

    def _get_default_main_config(self) -> Dict[str, Any]:
        """Retorna configuração principal padrão."""
        return {
            "version": "1.0.0",
            "environment": "development",
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "pipeline": {
                "default_sample_rate": 16000,
                "default_channels": 1,
                "max_audio_length": 30.0,
                "batch_size": 32,
                "num_workers": 4
            },
            "models": {
                "cache_dir": "models",
                "auto_download": True,
                "default_architecture": "aasist"
            },
            "features": {
                "cache_dir": "features",
                "default_extractors": ["mfcc", "spectral_centroid"],
                "normalize": True
            },
            "plugins": {
                "enabled": True,
                "auto_load": True,
                "plugins_dir": "plugins"
            }
        }

    def _load_schemas(self) -> None:
        """Carrega schemas de configuração."""
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)

                schema = ConfigSchema(
                    component_type=ComponentType(
                        schema_data["component_type"]),
                    component_name=schema_data["component_name"],
                    schema=schema_data["schema"],
                    version=schema_data.get("version", "1.0.0"),
                    description=schema_data.get("description", ""),
                    required_fields=schema_data.get("required_fields", []),
                    optional_fields=schema_data.get("optional_fields", [])
                )

                key = f"{schema.component_type.value}:{schema.component_name}"
                self._schema_cache[key] = schema

            except Exception as e:
                logger.error(f"Erro ao carregar schema {schema_file}: {e}")

    def _load_profiles(self) -> None:
        """Carrega perfis de configuração."""
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)

                profile = ConfigProfile.from_dict(profile_data)
                self._profile_cache[profile.name] = profile

            except Exception as e:
                logger.error(f"Erro ao carregar perfil {profile_file}: {e}")

    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Carrega configuração de arquivo.

        Args:
            config_path: Caminho da configuração

        Returns:
            Dicionário de configuração
        """
        if not config_path.exists():
            raise ConfigurationError(
                f"Arquivo de configuração não encontrado: {config_path}")

        try:
            format_type = ConfigFormat(config_path.suffix.lower().lstrip('.'))

            with open(config_path, 'r', encoding='utf-8') as f:
                if format_type in [ConfigFormat.YAML, ConfigFormat.YML]:
                    config = yaml.safe_load(f)
                elif format_type == ConfigFormat.JSON:
                    config = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Formato não suportado: {
                            config_path.suffix}")

            # Aplicar variáveis de ambiente
            config = self._apply_env_variables(config)

            return config

        except Exception as e:
            raise ConfigurationError(
                f"Erro ao carregar configuração {config_path}: {e}")

    def save_config(self, config: Dict[str, Any], config_path: Path) -> bool:
        """Salva configuração em arquivo.

        Args:
            config: Configuração a salvar
            config_path: Caminho para salvar

        Returns:
            True se salvo com sucesso
        """
        try:
            # Criar diretório se não existir
            config_path.parent.mkdir(parents=True, exist_ok=True)

            format_type = ConfigFormat(config_path.suffix.lower().lstrip('.'))

            with open(config_path, 'w', encoding='utf-8') as f:
                if format_type in [ConfigFormat.YAML, ConfigFormat.YML]:
                    yaml.dump(config, f, default_flow_style=False,
                              allow_unicode=True, indent=2)
                elif format_type == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ConfigurationError(
                        f"Formato não suportado: {
                            config_path.suffix}")

            logger.info(f"Configuração salva em {config_path}")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar configuração {config_path}: {e}")
            return False

    def get_component_config(self, component_type: ComponentType,
                             component_name: str) -> Dict[str, Any]:
        """Obtém configuração de componente.

        Args:
            component_type: Tipo do componente
            component_name: Nome do componente

        Returns:
            Configuração do componente
        """
        cache_key = f"{component_type.value}:{component_name}"

        # Verificar cache
        if cache_key in self._config_cache:
            return deepcopy(self._config_cache[cache_key])

        # Buscar arquivo de configuração
        config_file = self.components_dir / \
            component_type.value / f"{component_name}.yaml"

        if config_file.exists():
            config = self.load_config(config_file)
        else:
            # Usar configuração padrão
            config = self._get_default_component_config(
                component_type, component_name)

            # Salvar configuração padrão
            config_file.parent.mkdir(parents=True, exist_ok=True)
            self.save_config(config, config_file)

        # Aplicar configurações globais
        config = self._merge_with_global_config(config, component_type)

        # Cache
        self._config_cache[cache_key] = deepcopy(config)

        return config

    def set_component_config(self, component_type: ComponentType,
                             component_name: str, config: Dict[str, Any]) -> bool:
        """Define configuração de componente.

        Args:
            component_type: Tipo do componente
            component_name: Nome do componente
            config: Configuração a definir

        Returns:
            True se definido com sucesso
        """
        # Validar configuração
        errors = self.validate_config(config, component_type, component_name)
        if errors:
            logger.error(f"Erros de validação: {errors}")
            return False

        # Salvar arquivo
        config_file = self.components_dir / \
            component_type.value / f"{component_name}.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.save_config(config, config_file):
            return False

        # Atualizar cache
        cache_key = f"{component_type.value}:{component_name}"
        self._config_cache[cache_key] = deepcopy(config)

        return True

    def validate_config(self, config: Dict[str, Any],
                        component_type: Optional[ComponentType] = None,
                        component_name: Optional[str] = None) -> List[str]:
        """Valida configuração.

        Args:
            config: Configuração a validar
            component_type: Tipo do componente (opcional)
            component_name: Nome do componente (opcional)

        Returns:
            Lista de erros de validação
        """
        errors = []

        if component_type and component_name:
            # Validar contra schema específico
            schema_key = f"{component_type.value}:{component_name}"
            if schema_key in self._schema_cache:
                schema = self._schema_cache[schema_key]
                errors.extend(schema.validate(config))

        # Validações gerais
        if not isinstance(config, dict):
            errors.append("Configuração deve ser um dicionário")

        return errors

    def _apply_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica variáveis de ambiente na configuração.

        Args:
            config: Configuração original

        Returns:
            Configuração com variáveis aplicadas
        """
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                default_value = None

                if ":" in env_var:
                    env_var, default_value = env_var.split(":", 1)

                full_env_var = f"{self._env_prefix}{env_var}"
                return os.getenv(full_env_var, os.getenv(
                    env_var, default_value))
            else:
                return obj

        return replace_env_vars(config)

    def _get_default_component_config(self, component_type: ComponentType,
                                      component_name: str) -> Dict[str, Any]:
        """Retorna configuração padrão para componente.

        Args:
            component_type: Tipo do componente
            component_name: Nome do componente

        Returns:
            Configuração padrão
        """
        base_config = {
            "enabled": True,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat()
        }

        if component_type == ComponentType.FEATURE_EXTRACTOR:
            base_config.update({
                "sample_rate": 16000,
                "normalize": True,
                "cache_features": True
            })
        elif component_type == ComponentType.ARCHITECTURE:
            base_config.update({
                "batch_size": 32,
                "use_gpu": True,
                "model_cache": True
            })

        return base_config

    def _merge_with_global_config(self, config: Dict[str, Any],
                                  component_type: ComponentType) -> Dict[str, Any]:
        """Mescla configuração com configurações globais.

        Args:
            config: Configuração do componente
            component_type: Tipo do componente

        Returns:
            Configuração mesclada
        """
        merged = deepcopy(config)

        # Aplicar configurações globais do pipeline
        if "pipeline" in self._main_config:
            pipeline_config = self._main_config["pipeline"]

            # Aplicar sample_rate padrão se não especificado
            if "sample_rate" not in merged and "default_sample_rate" in pipeline_config:
                merged["sample_rate"] = pipeline_config["default_sample_rate"]

            # Aplicar batch_size padrão se não especificado
            if "batch_size" not in merged and "batch_size" in pipeline_config:
                merged["batch_size"] = pipeline_config["batch_size"]

        return merged

    def create_profile(self, name: str, description: str,
                       config: Dict[str, Any], tags: List[str] = None) -> bool:
        """Cria perfil de configuração.

        Args:
            name: Nome do perfil
            description: Descrição do perfil
            config: Configuração do perfil
            tags: Tags do perfil

        Returns:
            True se criado com sucesso
        """
        if tags is None:
            tags = []

        profile = ConfigProfile(
            name=name,
            description=description,
            config=config,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags
        )

        # Salvar arquivo
        profile_file = self.profiles_dir / f"{name}.json"

        try:
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

            # Atualizar cache
            self._profile_cache[name] = profile

            logger.info(f"Perfil {name} criado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro ao criar perfil {name}: {e}")
            return False

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Carrega perfil de configuração.

        Args:
            profile_name: Nome do perfil

        Returns:
            Configuração do perfil
        """
        if profile_name not in self._profile_cache:
            raise ConfigurationError(f"Perfil {profile_name} não encontrado")

        return deepcopy(self._profile_cache[profile_name].config)

    def list_profiles(self, tags: List[str] = None) -> List[str]:
        """Lista perfis disponíveis.

        Args:
            tags: Filtrar por tags

        Returns:
            Lista de nomes de perfis
        """
        profiles = list(self._profile_cache.keys())

        if tags:
            filtered_profiles = []
            for profile_name in profiles:
                profile = self._profile_cache[profile_name]
                if any(tag in profile.tags for tag in tags):
                    filtered_profiles.append(profile_name)
            profiles = filtered_profiles

        return sorted(profiles)

    def get_main_config(self) -> Dict[str, Any]:
        """Retorna configuração principal.

        Returns:
            Configuração principal
        """
        return deepcopy(self._main_config)

    def update_main_config(self, updates: Dict[str, Any]) -> bool:
        """Atualiza configuração principal.

        Args:
            updates: Atualizações a aplicar

        Returns:
            True se atualizado com sucesso
        """
        try:
            # Aplicar atualizações
            self._main_config.update(updates)

            # Salvar arquivo
            return self.save_config(self._main_config, self.main_config_file)

        except Exception as e:
            logger.error(f"Erro ao atualizar configuração principal: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas das configurações.

        Returns:
            Dicionário com estatísticas
        """
        return {
            "config_dir": str(self.config_dir),
            "cached_configs": len(self._config_cache),
            "loaded_schemas": len(self._schema_cache),
            "loaded_profiles": len(self._profile_cache),
            "main_config_loaded": bool(self._main_config),
            "component_types": list(ComponentType)
        }


# Instância global do gerenciador de configurações
config_manager = ConfigurationManager()


# Funções de conveniência
def get_config(component_type: ComponentType,
               component_name: str) -> Dict[str, Any]:
    """Função de conveniência para obter configuração.

    Args:
        component_type: Tipo do componente
        component_name: Nome do componente

    Returns:
        Configuração do componente
    """
    return config_manager.get_component_config(component_type, component_name)


def get_main_config() -> Dict[str, Any]:
    """Função de conveniência para obter configuração principal.

    Returns:
        Configuração principal
    """
    return config_manager.get_main_config()


def load_profile(profile_name: str) -> Dict[str, Any]:
    """Função de conveniência para carregar perfil.

    Args:
        profile_name: Nome do perfil

    Returns:
        Configuração do perfil
    """
    return config_manager.load_profile(profile_name)
