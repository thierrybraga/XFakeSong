"""Sistema de validação de compatibilidade entre componentes do pipeline.

Este módulo implementa validações para garantir que os componentes do pipeline
sejam compatíveis entre si e funcionem corretamente em conjunto.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from app.core.interfaces.base import ProcessingStatus
from app.domain.interfaces.pipeline_interfaces import (
    IPipelineComponent, IFeatureExtractor, IArchitecture,
    ProcessingContext
)
from app.core.interfaces.audio import AudioData
from app.domain.models.architectures.factory import ArchitectureSpec
from app.domain.features.extractor_registry import ExtractorSpec

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Níveis de validação."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class CompatibilityIssue(Enum):
    """Tipos de problemas de compatibilidade."""
    SHAPE_MISMATCH = "shape_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"
    CHANNEL_MISMATCH = "channel_mismatch"
    DEPENDENCY_MISSING = "dependency_missing"
    VERSION_INCOMPATIBLE = "version_incompatible"
    PERFORMANCE_WARNING = "performance_warning"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class ValidationResult:
    """Resultado de validação."""
    is_valid: bool
    issues: List[Dict[str, Any]]
    warnings: List[str]
    recommendations: List[str]
    performance_score: Optional[float] = None

    def add_issue(self, issue_type: CompatibilityIssue,
                  component: str, message: str, severity: str = "error"):
        """Adiciona problema de compatibilidade."""
        self.issues.append({
            "type": issue_type.value,
            "component": component,
            "message": message,
            "severity": severity
        })

        if severity == "error":
            self.is_valid = False

    def add_warning(self, message: str):
        """Adiciona aviso."""
        self.warnings.append(message)

    def add_recommendation(self, message: str):
        """Adiciona recomendação."""
        self.recommendations.append(message)


class ComponentValidator:
    """Validador de componentes individuais."""

    @staticmethod
    def validate_feature_extractor(extractor: IFeatureExtractor,
                                   spec: ExtractorSpec) -> ValidationResult:
        """Valida extrator de features.

        Args:
            extractor: Instância do extrator
            spec: Especificação do extrator

        Returns:
            Resultado da validação
        """
        result = ValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            recommendations=[])

        try:
            # Verificar interface
            if not hasattr(extractor, 'extract_features'):
                result.add_issue(
                    CompatibilityIssue.TYPE_MISMATCH,
                    spec.name,
                    "Extrator deve implementar método extract_features"
                )

            if not hasattr(extractor, 'get_feature_names'):
                result.add_issue(
                    CompatibilityIssue.TYPE_MISMATCH,
                    spec.name,
                    "Extrator deve implementar método get_feature_names"
                )

            # Verificar dependências
            missing_deps = ComponentValidator._check_dependencies(
                spec.dependencies)
            if missing_deps:
                result.add_issue(
                    CompatibilityIssue.DEPENDENCY_MISSING,
                    spec.name,
                    f"Dependências não encontradas: {missing_deps}"
                )

            # Teste básico de extração
            try:
                # Criar dados de teste
                sample_rate = spec.input_requirements.get("sample_rate", 16000)
                test_audio = np.random.randn(sample_rate)  # 1 segundo de áudio

                # Criar AudioData para teste
                audio_data = AudioData(
                    samples=test_audio,
                    sample_rate=sample_rate,
                    duration=len(test_audio)/sample_rate,
                    metadata={}
                )

                # Tentar extrair features usando o método principal extract
                extraction_result = extractor.extract(audio_data)

                if extraction_result.status != ProcessingStatus.SUCCESS:
                    result.add_issue(
                        CompatibilityIssue.CONFIGURATION_ERROR,
                        spec.name,
                        f"Falha na extração de teste: "
                        f"{extraction_result.errors}"
                    )
                elif not extraction_result.data:
                    result.add_issue(
                        CompatibilityIssue.CONFIGURATION_ERROR,
                        spec.name,
                        "Extração não retornou dados"
                    )
                else:
                    features = extraction_result.data.features
                    if features is None:
                        result.add_issue(
                            CompatibilityIssue.CONFIGURATION_ERROR,
                            spec.name,
                            "Objeto AudioFeatures vazio"
                        )
                    elif len(features) == 0:
                        result.add_issue(
                            CompatibilityIssue.CONFIGURATION_ERROR,
                            spec.name,
                            "Dicionário de features vazio"
                        )

            except Exception as e:
                result.add_issue(
                    CompatibilityIssue.CONFIGURATION_ERROR,
                    spec.name,
                    f"Erro no teste de extração: {e}"
                )

        except Exception as e:
            result.add_issue(
                CompatibilityIssue.CONFIGURATION_ERROR,
                spec.name,
                f"Erro na validação: {e}"
            )

        return result

    @staticmethod
    def validate_architecture(architecture: IArchitecture,
                              spec: ArchitectureSpec) -> ValidationResult:
        """Valida arquitetura.

        Args:
            architecture: Instância da arquitetura
            spec: Especificação da arquitetura

        Returns:
            Resultado da validação
        """
        result = ValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            recommendations=[])

        try:
            # Verificar interface
            if not hasattr(architecture, 'predict'):
                result.add_issue(
                    CompatibilityIssue.TYPE_MISMATCH,
                    spec.name,
                    "Arquitetura deve implementar método predict"
                )

            if not hasattr(architecture, 'load_model'):
                result.add_issue(
                    CompatibilityIssue.TYPE_MISMATCH,
                    spec.name,
                    "Arquitetura deve implementar método load_model"
                )

            # Verificar dependências
            missing_deps = ComponentValidator._check_dependencies(
                spec.dependencies)
            if missing_deps:
                result.add_issue(
                    CompatibilityIssue.DEPENDENCY_MISSING,
                    spec.name,
                    f"Dependências não encontradas: {missing_deps}"
                )

            # Verificar shape de entrada
            if spec.input_shape:
                try:
                    # Criar features de teste
                    test_features = np.random.randn(*spec.input_shape)

                    context = ProcessingContext(
                        session_id="validation_test"
                    )

                    # Tentar predição (se modelo estiver carregado)
                    if hasattr(architecture,
                               '_model') and architecture._model is not None:
                        prediction_result = architecture.predict(
                            test_features, context)

                        if not prediction_result.success:
                            result.add_warning(
                                f"Falha na predição de teste: {
                                    prediction_result.errors}"
                            )

                except Exception as e:
                    result.add_warning(f"Erro no teste de predição: {e}")

        except Exception as e:
            result.add_issue(
                CompatibilityIssue.CONFIGURATION_ERROR,
                spec.name,
                f"Erro na validação: {e}"
            )

        return result

    @staticmethod
    def _check_dependencies(dependencies: List[str]) -> List[str]:
        """Verifica dependências.

        Args:
            dependencies: Lista de dependências

        Returns:
            Lista de dependências não encontradas
        """
        missing = []

        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        return missing


class PipelineValidator:
    """Validador de pipeline completo."""

    def __init__(
            self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.component_validator = ComponentValidator()

    def validate_pipeline(
        self,
        extractors: List[Tuple[IFeatureExtractor, ExtractorSpec]],
        architectures: List[Tuple[IArchitecture, ArchitectureSpec]]
    ) -> ValidationResult:
        """Valida pipeline completo.

        Args:
            extractors: Lista de extratores e suas especificações
            architectures: Lista de arquiteturas e suas especificações

        Returns:
            Resultado da validação
        """
        result = ValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            recommendations=[])

        # Validar componentes individuais
        for extractor, spec in extractors:
            comp_result = self.component_validator.validate_feature_extractor(
                extractor, spec)
            result.issues.extend(comp_result.issues)
            result.warnings.extend(comp_result.warnings)
            result.recommendations.extend(comp_result.recommendations)

            if not comp_result.is_valid:
                result.is_valid = False

        for architecture, spec in architectures:
            comp_result = self.component_validator.validate_architecture(
                architecture, spec)
            result.issues.extend(comp_result.issues)
            result.warnings.extend(comp_result.warnings)
            result.recommendations.extend(comp_result.recommendations)

            if not comp_result.is_valid:
                result.is_valid = False

        # Validar compatibilidade entre componentes
        compat_result = self._validate_compatibility(extractors, architectures)
        result.issues.extend(compat_result.issues)
        result.warnings.extend(compat_result.warnings)
        result.recommendations.extend(compat_result.recommendations)

        if not compat_result.is_valid:
            result.is_valid = False

        # Calcular score de performance
        result.performance_score = self._calculate_performance_score(
            extractors, architectures)

        return result

    def _validate_compatibility(
        self,
        extractors: List[Tuple[IFeatureExtractor, ExtractorSpec]],
        architectures: List[Tuple[IArchitecture, ArchitectureSpec]]
    ) -> ValidationResult:
        """Valida compatibilidade entre componentes.

        Args:
            extractors: Lista de extratores
            architectures: Lista de arquiteturas

        Returns:
            Resultado da validação de compatibilidade
        """
        result = ValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            recommendations=[])

        # Verificar compatibilidade de sample rate
        sample_rates = set()
        for _, spec in extractors:
            if "sample_rate" in spec.input_requirements:
                sample_rates.add(spec.input_requirements["sample_rate"])

        if len(sample_rates) > 1:
            result.add_warning(
                f"Diferentes sample rates detectados: {sample_rates}. "
                "Isso pode causar problemas de compatibilidade."
            )

        # Verificar compatibilidade de shapes
        for extractor, ext_spec in extractors:
            for architecture, arch_spec in architectures:
                if ext_spec.output_shape and arch_spec.input_shape:
                    if not self._shapes_compatible(
                            ext_spec.output_shape, arch_spec.input_shape):
                        result.add_issue(
                            CompatibilityIssue.SHAPE_MISMATCH,
                            f"{ext_spec.name} -> {arch_spec.name}",
                            f"Shapes incompatíveis: {
                                ext_spec.output_shape} -> {
                                arch_spec.input_shape}"
                        )

        # Verificar dependências conflitantes
        all_deps = set()
        for _, spec in extractors + architectures:
            all_deps.update(spec.dependencies)

        # Verificar se há conflitos conhecidos
        conflicts = self._check_dependency_conflicts(list(all_deps))
        for conflict in conflicts:
            result.add_warning(f"Possível conflito de dependência: {conflict}")

        return result

    def _shapes_compatible(self, output_shape: tuple,
                           input_shape: tuple) -> bool:
        """Verifica se shapes são compatíveis.

        Args:
            output_shape: Shape de saída
            input_shape: Shape de entrada

        Returns:
            True se compatíveis
        """
        # Permitir batch dimension flexível
        if len(output_shape) == len(input_shape):
            for i, (out_dim, in_dim) in enumerate(
                    zip(output_shape, input_shape)):
                # Primeira dimensão pode ser batch size (flexível)
                if i == 0:
                    continue
                if out_dim != in_dim and out_dim != -1 and in_dim != -1:
                    return False
            return True

        # Verificar se pode ser reshaped
        if len(output_shape) == 1 and len(input_shape) > 1:
            # Flatten pode ser aplicado
            return True

        return False

    def _check_dependency_conflicts(
            self, dependencies: List[str]) -> List[str]:
        """Verifica conflitos de dependências.

        Args:
            dependencies: Lista de dependências

        Returns:
            Lista de conflitos encontrados
        """
        conflicts = []

        # Conflitos conhecidos
        known_conflicts = {
            ("tensorflow", "torch"): "TensorFlow e PyTorch podem conflitar",
            ("librosa", "torchaudio"): "Librosa e torchaudio podem ter conflitos de versão"
        }

        for (dep1, dep2), message in known_conflicts.items():
            if dep1 in dependencies and dep2 in dependencies:
                conflicts.append(message)

        return conflicts

    def _calculate_performance_score(
        self,
        extractors: List[Tuple[IFeatureExtractor, ExtractorSpec]],
        architectures: List[Tuple[IArchitecture, ArchitectureSpec]]
    ) -> float:
        """Calcula score de performance do pipeline.

        Args:
            extractors: Lista de extratores
            architectures: Lista de arquiteturas

        Returns:
            Score de performance (0-100)
        """
        score = 100.0

        # Penalizar por complexidade alta
        for _, spec in extractors:
            if spec.complexity.value == "very_high":
                score -= 10
            elif spec.complexity.value == "high":
                score -= 5

        # Penalizar por muitos componentes
        total_components = len(extractors) + len(architectures)
        if total_components > 5:
            score -= (total_components - 5) * 2

        # Bonificar por componentes otimizados
        for _, spec in architectures:
            if "optimized" in spec.description.lower():
                score += 5

        return max(0.0, min(100.0, score))

    def validate_configuration(
            self, config: Dict[str, Any]) -> ValidationResult:
        """Valida configuração do pipeline.

        Args:
            config: Configuração a validar

        Returns:
            Resultado da validação
        """
        result = ValidationResult(
            is_valid=True,
            issues=[],
            warnings=[],
            recommendations=[])

        # Verificar campos obrigatórios
        required_fields = ["sample_rate", "extractors", "architectures"]
        for field in required_fields:
            if field not in config:
                result.add_issue(
                    CompatibilityIssue.CONFIGURATION_ERROR,
                    "config",
                    f"Campo obrigatório ausente: {field}"
                )

        # Verificar valores válidos
        if "sample_rate" in config:
            sample_rate = config["sample_rate"]
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                result.add_issue(
                    CompatibilityIssue.CONFIGURATION_ERROR,
                    "config",
                    "Sample rate deve ser um inteiro positivo"
                )
            elif sample_rate < 8000 or sample_rate > 48000:
                result.add_warning(
                    f"Sample rate {sample_rate} pode não ser ideal "
                    "(recomendado: 16000-44100)"
                )

        # Verificar batch size
        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                result.add_issue(
                    CompatibilityIssue.CONFIGURATION_ERROR,
                    "config",
                    "Batch size deve ser um inteiro positivo"
                )
            elif batch_size > 128:
                result.add_warning(
                    f"Batch size {batch_size} pode ser muito alto "
                    "para alguns sistemas"
                )

        return result

    def suggest_optimizations(
        self,
        extractors: List[Tuple[IFeatureExtractor, ExtractorSpec]],
        architectures: List[Tuple[IArchitecture, ArchitectureSpec]]
    ) -> List[str]:
        """Sugere otimizações para o pipeline.

        Args:
            extractors: Lista de extratores
            architectures: Lista de arquiteturas

        Returns:
            Lista de sugestões de otimização
        """
        suggestions = []

        # Sugerir redução de extratores redundantes
        extractor_types = [spec.feature_type for _, spec in extractors]
        if len(set(extractor_types)) < len(extractor_types):
            suggestions.append(
                "Considere remover extratores redundantes do mesmo tipo"
            )

        # Sugerir otimizações de complexidade
        high_complexity = [
            spec.name for _, spec in extractors
            if spec.complexity.value in ["high", "very_high"]
        ]
        if high_complexity:
            suggestions.append(
                f"Considere alternativas mais eficientes para: "
                f"{', '.join(high_complexity)}"
            )

        # Sugerir cache de features
        suggestions.append(
            "Considere implementar cache de features para melhorar performance"
        )

        # Sugerir processamento em lote
        if len(extractors) > 2:
            suggestions.append(
                "Considere processamento em lote para múltiplos extratores"
            )

        return suggestions


# Instância global do validador
pipeline_validator = PipelineValidator()


# Funções de conveniência
def validate_pipeline(
    extractors: List[Tuple[IFeatureExtractor, ExtractorSpec]],
    architectures: List[Tuple[IArchitecture, ArchitectureSpec]]
) -> ValidationResult:
    """Função de conveniência para validar pipeline.

    Args:
        extractors: Lista de extratores e especificações
        architectures: Lista de arquiteturas e especificações

    Returns:
        Resultado da validação
    """
    return pipeline_validator.validate_pipeline(extractors, architectures)


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """Função de conveniência para validar configuração.

    Args:
        config: Configuração a validar

    Returns:
        Resultado da validação
    """
    return pipeline_validator.validate_configuration(config)


def suggest_optimizations(
    extractors: List[Tuple[IFeatureExtractor, ExtractorSpec]],
    architectures: List[Tuple[IArchitecture, ArchitectureSpec]]
) -> List[str]:
    """Função de conveniência para sugerir otimizações.

    Args:
        extractors: Lista de extratores e especificações
        architectures: Lista de arquiteturas e especificações

    Returns:
        Lista de sugestões
    """
    return pipeline_validator.suggest_optimizations(extractors, architectures)
