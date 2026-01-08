"""Sistema de plugins para arquiteturas e extratores.

Este módulo implementa um sistema de plugins que permite carregamento dinâmico
de arquiteturas e extratores de features personalizados.
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import hashlib
from datetime import datetime

from app.domain.interfaces.pipeline_interfaces import (
    IPluginManager, IPipelineComponent, ComponentType,
    IFeatureExtractor, IArchitecture, PluginError
)
from app.domain.models.architectures.factory import ArchitectureSpec
from app.domain.features.extractor_registry import ExtractorSpec, FeatureType, ExtractorComplexity

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadados de um plugin."""
    name: str
    version: str
    author: str
    description: str
    component_type: ComponentType
    dependencies: List[str]
    entry_point: str
    config_schema: Dict[str, Any]
    created_at: datetime
    file_hash: str
    file_path: Path

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "component_type": self.component_type.value,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "created_at": self.created_at.isoformat(),
            "file_hash": self.file_hash,
            "file_path": str(self.file_path)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Cria instância a partir de dicionário."""
        return cls(
            name=data["name"],
            version=data["version"],
            author=data["author"],
            description=data["description"],
            component_type=ComponentType(data["component_type"]),
            dependencies=data["dependencies"],
            entry_point=data["entry_point"],
            config_schema=data["config_schema"],
            created_at=datetime.fromisoformat(data["created_at"]),
            file_hash=data["file_hash"],
            file_path=Path(data["file_path"])
        )


class IPlugin(ABC):
    """Interface base para plugins."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Retorna metadados do plugin."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Inicializa o plugin."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Limpa recursos do plugin."""
        pass

    @abstractmethod
    def get_component_class(self) -> Type[IPipelineComponent]:
        """Retorna classe do componente."""
        pass


class ArchitecturePlugin(IPlugin):
    """Plugin para arquiteturas."""

    @abstractmethod
    def get_architecture_spec(self) -> ArchitectureSpec:
        """Retorna especificação da arquitetura."""
        pass


class ExtractorPlugin(IPlugin):
    """Plugin para extratores de features."""

    @abstractmethod
    def get_extractor_spec(self) -> ExtractorSpec:
        """Retorna especificação do extrator."""
        pass


class PluginValidator:
    """Validador de plugins."""

    @staticmethod
    def validate_plugin_file(plugin_path: Path) -> List[str]:
        """Valida arquivo de plugin.

        Args:
            plugin_path: Caminho do plugin

        Returns:
            Lista de erros de validação
        """
        errors = []

        if not plugin_path.exists():
            errors.append(f"Arquivo não encontrado: {plugin_path}")
            return errors

        if not plugin_path.suffix == '.py':
            errors.append("Plugin deve ser um arquivo Python (.py)")

        try:
            # Verificar se o arquivo pode ser importado
            spec = importlib.util.spec_from_file_location(
                "temp_plugin", plugin_path)
            if spec is None:
                errors.append("Não foi possível criar especificação do módulo")
                return errors

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verificar se tem classe de plugin
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, IPlugin) and obj != IPlugin:
                    plugin_classes.append(obj)

            if not plugin_classes:
                errors.append("Plugin deve implementar interface IPlugin")

            # Verificar metadados
            for plugin_class in plugin_classes:
                try:
                    instance = plugin_class()
                    metadata = instance.get_metadata()
                    if not metadata.name:
                        errors.append(
                            f"Plugin {
                                plugin_class.__name__} deve ter nome")
                    if not metadata.version:
                        errors.append(
                            f"Plugin {
                                plugin_class.__name__} deve ter versão")
                except Exception as e:
                    errors.append(
                        f"Erro ao obter metadados de {
                            plugin_class.__name__}: {e}")

        except Exception as e:
            errors.append(f"Erro ao validar plugin: {e}")

        return errors

    @staticmethod
    def validate_dependencies(dependencies: List[str]) -> List[str]:
        """Valida dependências do plugin.

        Args:
            dependencies: Lista de dependências

        Returns:
            Lista de dependências não encontradas
        """
        missing = []

        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)

        return missing


class PluginManager(IPluginManager):
    """Gerenciador de plugins."""

    def __init__(self, plugins_dir: Optional[Path] = None):
        self.plugins_dir = plugins_dir or Path("plugins")
        self.plugins_dir.mkdir(exist_ok=True)

        self._loaded_plugins: Dict[str, IPlugin] = {}
        self._plugin_modules: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}

        # Arquivo de cache de metadados
        self.metadata_file = self.plugins_dir / "plugin_metadata.json"

        self._load_metadata_cache()

    def _load_metadata_cache(self) -> None:
        """Carrega cache de metadados."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for plugin_name, metadata_dict in data.items():
                    self._metadata_cache[plugin_name] = PluginMetadata.from_dict(
                        metadata_dict)

            except Exception as e:
                logger.warning(f"Erro ao carregar cache de metadados: {e}")

    def _save_metadata_cache(self) -> None:
        """Salva cache de metadados."""
        try:
            data = {name: metadata.to_dict()
                    for name, metadata in self._metadata_cache.items()}

            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erro ao salvar cache de metadados: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash do arquivo.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Hash SHA256 do arquivo
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def load_plugin(self, plugin_path: Path) -> bool:
        """Carrega plugin.

        Args:
            plugin_path: Caminho do plugin

        Returns:
            True se carregado com sucesso
        """
        try:
            # Validar plugin
            errors = PluginValidator.validate_plugin_file(plugin_path)
            if errors:
                logger.error(f"Erros de validação em {plugin_path}: {errors}")
                return False

            # Calcular hash do arquivo
            file_hash = self._calculate_file_hash(plugin_path)

            # Verificar se já está carregado e não mudou
            plugin_name = plugin_path.stem
            if (plugin_name in self._metadata_cache and
                    self._metadata_cache[plugin_name].file_hash == file_hash):
                logger.info(f"Plugin {plugin_name} já carregado e atualizado")
                return True

            # Importar módulo
            spec = importlib.util.spec_from_file_location(
                plugin_name, plugin_path)
            module = importlib.util.module_from_spec(spec)

            # Adicionar ao sys.modules para permitir imports relativos
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            # Encontrar classes de plugin
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, IPlugin) and
                        obj not in [IPlugin, ArchitecturePlugin, ExtractorPlugin]):
                    plugin_classes.append(obj)

            if not plugin_classes:
                logger.error(
                    f"Nenhuma classe de plugin encontrada em {plugin_path}")
                return False

            # Instanciar e registrar plugins
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    metadata = plugin_instance.get_metadata()

                    # Validar dependências
                    missing_deps = PluginValidator.validate_dependencies(
                        metadata.dependencies)
                    if missing_deps:
                        logger.error(
                            f"Dependências não encontradas para {
                                metadata.name}: {missing_deps}")
                        continue

                    # Atualizar metadados com informações do arquivo
                    metadata.file_path = plugin_path
                    metadata.file_hash = file_hash
                    metadata.created_at = datetime.now()

                    # Inicializar plugin
                    if not plugin_instance.initialize({}):
                        logger.error(
                            f"Falha ao inicializar plugin {
                                metadata.name}")
                        continue

                    # Registrar plugin
                    self._loaded_plugins[metadata.name] = plugin_instance
                    self._plugin_modules[metadata.name] = module
                    self._metadata_cache[metadata.name] = metadata

                    # Registrar componente nos registries apropriados
                    self._register_component(plugin_instance)

                    logger.info(
                        f"Plugin {
                            metadata.name} carregado com sucesso")

                except Exception as e:
                    logger.error(
                        f"Erro ao carregar plugin {
                            plugin_class.__name__}: {e}")

            # Salvar cache
            self._save_metadata_cache()
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar plugin {plugin_path}: {e}")
            return False

    def _register_component(self, plugin: IPlugin) -> None:
        """Registra componente do plugin nos registries.

        Args:
            plugin: Instância do plugin
        """
        metadata = plugin.get_metadata()

        try:
            if isinstance(plugin, ArchitecturePlugin):
                # Registrar arquitetura
                from app.domain.models.architectures.factory import architecture_factory_registry

                arch_spec = plugin.get_architecture_spec()
                architecture_factory_registry.register_architecture(arch_spec)
                logger.info(f"Arquitetura {metadata.name} registrada")

            elif isinstance(plugin, ExtractorPlugin):
                # Registrar extrator
                from app.domain.features.extractor_registry import extractor_registry

                ext_spec = plugin.get_extractor_spec()
                extractor_registry.register(ext_spec)
                logger.info(f"Extrator {metadata.name} registrado")

        except Exception as e:
            logger.error(f"Erro ao registrar componente {metadata.name}: {e}")

    def unload_plugin(self, plugin_name: str) -> bool:
        """Descarrega plugin.

        Args:
            plugin_name: Nome do plugin

        Returns:
            True se descarregado com sucesso
        """
        if plugin_name not in self._loaded_plugins:
            logger.warning(f"Plugin {plugin_name} não está carregado")
            return False

        try:
            plugin = self._loaded_plugins[plugin_name]

            # Cleanup do plugin
            plugin.cleanup()

            # Remover dos registries
            self._unregister_component(plugin)

            # Remover das estruturas internas
            del self._loaded_plugins[plugin_name]

            if plugin_name in self._plugin_modules:
                # Remover do sys.modules
                if plugin_name in sys.modules:
                    del sys.modules[plugin_name]
                del self._plugin_modules[plugin_name]

            if plugin_name in self._metadata_cache:
                del self._metadata_cache[plugin_name]

            # Salvar cache
            self._save_metadata_cache()

            logger.info(f"Plugin {plugin_name} descarregado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro ao descarregar plugin {plugin_name}: {e}")
            return False

    def _unregister_component(self, plugin: IPlugin) -> None:
        """Remove componente do plugin dos registries.

        Args:
            plugin: Instância do plugin
        """
        metadata = plugin.get_metadata()

        try:
            if isinstance(plugin, ArchitecturePlugin):
                # Remover arquitetura
                from app.domain.models.architectures.factory import architecture_factory_registry

                architecture_factory_registry.unregister_architecture(
                    metadata.name)
                logger.info(f"Arquitetura {metadata.name} removida")

            elif isinstance(plugin, ExtractorPlugin):
                # Remover extrator
                from app.domain.features.extractor_registry import extractor_registry

                extractor_registry.unregister(metadata.name)
                logger.info(f"Extrator {metadata.name} removido")

        except Exception as e:
            logger.error(f"Erro ao remover componente {metadata.name}: {e}")

    def list_plugins(self) -> List[str]:
        """Lista plugins carregados.

        Returns:
            Lista de nomes de plugins
        """
        return list(self._loaded_plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Obtém informações de plugin.

        Args:
            plugin_name: Nome do plugin

        Returns:
            Informações do plugin
        """
        if plugin_name not in self._loaded_plugins:
            raise PluginError(f"Plugin {plugin_name} não encontrado")

        metadata = self._metadata_cache[plugin_name]
        return metadata.to_dict()

    def discover_plugins(self) -> List[Path]:
        """Descobre plugins no diretório de plugins.

        Returns:
            Lista de caminhos de plugins
        """
        plugin_files = []

        for file_path in self.plugins_dir.rglob("*.py"):
            if file_path.name.startswith("__"):
                continue
            plugin_files.append(file_path)

        return plugin_files

    def load_all_plugins(self) -> Dict[str, bool]:
        """Carrega todos os plugins descobertos.

        Returns:
            Dicionário com resultado do carregamento
        """
        results = {}

        for plugin_path in self.discover_plugins():
            plugin_name = plugin_path.stem
            results[plugin_name] = self.load_plugin(plugin_path)

        return results

    def get_plugins_by_type(self, component_type: ComponentType) -> List[str]:
        """Obtém plugins por tipo de componente.

        Args:
            component_type: Tipo de componente

        Returns:
            Lista de nomes de plugins
        """
        return [name for name, metadata in self._metadata_cache.items()
                if metadata.component_type == component_type]

    def reload_plugin(self, plugin_name: str) -> bool:
        """Recarrega um plugin.

        Args:
            plugin_name: Nome do plugin

        Returns:
            True se recarregado com sucesso
        """
        if plugin_name not in self._metadata_cache:
            logger.error(f"Plugin {plugin_name} não encontrado")
            return False

        plugin_path = self._metadata_cache[plugin_name].file_path

        # Descarregar primeiro
        if plugin_name in self._loaded_plugins:
            self.unload_plugin(plugin_name)

        # Recarregar
        return self.load_plugin(plugin_path)

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas dos plugins.

        Returns:
            Dicionário com estatísticas
        """
        stats = {
            "total_plugins": len(self._loaded_plugins),
            "by_type": {},
            "loaded_plugins": list(self._loaded_plugins.keys()),
            "plugins_dir": str(self.plugins_dir)
        }

        for component_type in ComponentType:
            plugins = self.get_plugins_by_type(component_type)
            stats["by_type"][component_type.value] = len(plugins)

        return stats


# Instância global do gerenciador de plugins
plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Retorna instância global do gerenciador de plugins."""
    return plugin_manager


# Funções de conveniência
def load_plugin(plugin_path: Union[str, Path]) -> bool:
    """Função de conveniência para carregar plugin.

    Args:
        plugin_path: Caminho do plugin

    Returns:
        True se carregado com sucesso
    """
    return plugin_manager.load_plugin(Path(plugin_path))


def list_plugins() -> List[str]:
    """Função de conveniência para listar plugins.

    Returns:
        Lista de nomes de plugins
    """
    return plugin_manager.list_plugins()


def get_plugin_info(plugin_name: str) -> Dict[str, Any]:
    """Função de conveniência para obter informações de plugin.

    Args:
        plugin_name: Nome do plugin

    Returns:
        Informações do plugin
    """
    return plugin_manager.get_plugin_info(plugin_name)
