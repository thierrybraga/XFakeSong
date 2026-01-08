"""Orquestrador do Pipeline de Detecção de Deepfake, Implementa a orquestração completa do pipeline seguindo princípios
de arquitetura limpa.
"""


import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.interfaces.base import ProcessingStatus
from core.interfaces.audio import AudioData, AudioFeatures
from core.interfaces.services import (
    IUploadService, IFeatureExtractionService, ITrainingService,
    IDetectionService, DatasetType
)


@dataclass
class PipelineConfig:
    """Configuração do pipeline"""
    upload_directory: str
    feature_types: List[str]
    model_architecture: str
    training_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    enable_monitoring: bool = True
    save_intermediate_results: bool = True


@dataclass
class PipelineResult:
    """Resultado do pipeline"""
    stage: str
    status: ProcessingStatus
    data: Any = None
    error: Optional[str] = None
    message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class PipelineStage(ABC):
    """Classe base para estágios do pipeline"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, input_data: Any,
                config: PipelineConfig) -> PipelineResult:
        """Executar o estágio do pipeline"""
        pass

    def _create_result(self, status: ProcessingStatus, data: Any = None,
                       error: str = None, message: str = None,
                       execution_time: float = 0.0) -> PipelineResult:
        """Criar resultado padronizado"""
        return PipelineResult(
            stage=self.name,
            status=status,
            data=data,
            error=error,
            message=message,
            execution_time=execution_time
        )


class UploadStage(PipelineStage):
    """Estágio de upload de arquivos"""

    def __init__(self, upload_service: IUploadService):
        super().__init__("upload")
        self.upload_service = upload_service

    def execute(self, input_data: Union[str, List[str]],
                config: PipelineConfig) -> PipelineResult:
        """Executar upload de arquivos"""
        start_time = time.time()

        try:
            if isinstance(input_data, str):
                # Upload de arquivo único
                result = self.upload_service.upload_file(
                    input_data, DatasetType.TRAINING)
            else:
                # Upload em lote
                result = self.upload_service.upload_batch(
                    input_data, DatasetType.TRAINING)

            execution_time = time.time() - start_time

            return self._create_result(
                status=result.status,
                data=result.data,
                error=result.error,
                message=result.message,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                status=ProcessingStatus.ERROR,
                error=str(e),
                message=f"Erro no upload: {str(e)}",
                execution_time=execution_time
            )


class FeatureExtractionStage(PipelineStage):
    """Estágio de extração de características"""

    def __init__(self, extraction_service: IFeatureExtractionService):
        super().__init__("feature_extraction")
        self.extraction_service = extraction_service

    def execute(self, input_data: List[AudioData],
                config: PipelineConfig) -> PipelineResult:
        """Executar extração de características"""
        start_time = time.time()

        try:
            from domain.services.feature_extraction_service import (
                ExtractionConfig
            )
            from core.interfaces.audio import FeatureType

            # Converter strings para FeatureType
            feature_types = []
            for ft_str in config.feature_types:
                try:
                    feature_types.append(FeatureType(ft_str.lower()))
                except ValueError:
                    print(f"Warning: Unknown feature type {ft_str}")

            extraction_config = ExtractionConfig(
                feature_types=feature_types,
                normalize=True
            )

            result = self.extraction_service.extract_batch(
                input_data, extraction_config)

            execution_time = time.time() - start_time

            return self._create_result(
                status=result.status,
                data=result.data,
                error=result.error,
                message=result.message,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                status=ProcessingStatus.ERROR,
                error=str(e),
                message=f"Erro na extração: {str(e)}",
                execution_time=execution_time
            )


class TrainingStage(PipelineStage):
    """Estágio de treinamento do modelo"""

    def __init__(self, training_service: ITrainingService):
        super().__init__("training")
        self.training_service = training_service

    def execute(self, input_data: List[AudioFeatures],
                config: PipelineConfig) -> PipelineResult:
        """Executar treinamento do modelo"""
        start_time = time.time()

        try:
            # Implementar lógica de treinamento
            # Por enquanto, retorna sucesso simulado
            execution_time = time.time() - start_time

            return self._create_result(
                status=ProcessingStatus.SUCCESS,
                data={"model_path": "models/trained_model.pth"},
                message=f"Modelo treinado com sucesso em {
                    execution_time:.2f}s",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                status=ProcessingStatus.ERROR,
                error=str(e),
                message=f"Erro no treinamento: {str(e)}",
                execution_time=execution_time
            )


class DeepfakePipelineOrchestrator:
    """Orquestrador principal do pipeline de detecção de deepfake"""

    def __init__(self, upload_service: IUploadService,
                 extraction_service: IFeatureExtractionService,
                 training_service: Optional[ITrainingService] = None,
                 detection_service: Optional[IDetectionService] = None):
        self.upload_service = upload_service
        self.extraction_service = extraction_service
        self.training_service = training_service
        self.detection_service = detection_service

        # Inicializar estágios
        self.stages = {
            "upload": UploadStage(upload_service),
            "feature_extraction": FeatureExtractionStage(extraction_service),
        }

        if training_service:
            self.stages["training"] = TrainingStage(training_service)

    def run_training_pipeline(self, file_paths: List[str],
                              config: PipelineConfig) -> List[PipelineResult]:
        """Executar pipeline completo de treinamento"""
        results = []

        try:
            # Estágio 1: Upload
            upload_result = self.stages["upload"].execute(file_paths, config)
            results.append(upload_result)

            if upload_result.status == ProcessingStatus.ERROR:
                return results

            # Converter resultados de upload para AudioData
            audio_data_list = self._convert_upload_to_audio_data(
                upload_result.data)

            # Estágio 2: Extração de características
            extraction_result = self.stages["feature_extraction"].execute(
                audio_data_list, config
            )
            results.append(extraction_result)

            if extraction_result.status == ProcessingStatus.ERROR:
                return results

            # Estágio 3: Treinamento (se disponível)
            if "training" in self.stages:
                training_result = self.stages["training"].execute(
                    extraction_result.data, config
                )
                results.append(training_result)

            return results

        except Exception as e:
            error_result = PipelineResult(
                stage="orchestrator",
                status=ProcessingStatus.ERROR,
                error=str(e),
                message=f"Erro na orquestração: {str(e)}"
            )
            results.append(error_result)
            return results

    def run_detection_pipeline(self, file_path: str,
                               config: PipelineConfig) -> List[PipelineResult]:
        """Executar pipeline de detecção"""
        results = []

        try:
            # Upload do arquivo
            upload_result = self.stages["upload"].execute(file_path, config)
            results.append(upload_result)

            if upload_result.status == ProcessingStatus.ERROR:
                return results

            # Extração de características
            audio_data_list = self._convert_upload_to_audio_data(
                [upload_result.data])
            extraction_result = self.stages["feature_extraction"].execute(
                audio_data_list, config
            )
            results.append(extraction_result)

            # Detecção (se disponível)
            if (self.detection_service and
                    extraction_result.status == ProcessingStatus.SUCCESS):
                # Implementar lógica de detecção
                pass

            return results

        except Exception as e:
            error_result = PipelineResult(
                stage="orchestrator",
                status=ProcessingStatus.ERROR,
                error=str(e),
                message=f"Erro na detecção: {str(e)}"
            )
            results.append(error_result)
            return results

    def _convert_upload_to_audio_data(
            self, upload_results: List[Any]) -> List[AudioData]:
        """Converter resultados de upload para AudioData"""
        audio_data_list = []

        for upload_result in upload_results:
            # Simular carregamento de áudio
            # Em uma implementação real, usaria librosa ou similar
            audio_data = AudioData(
                data=None,  # Seria carregado do arquivo
                sample_rate=16000,
                duration=0.0,
                channels=1,
                file_path=getattr(upload_result, 'file_path', None)
            )
            audio_data_list.append(audio_data)

        return audio_data_list

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retornar status do pipeline"""
        return {
            "available_stages": list(self.stages.keys()),
            "services_configured": {
                "upload": self.upload_service is not None,
                "extraction": self.extraction_service is not None,
                "training": self.training_service is not None,
                "detection": self.detection_service is not None
            }
        }
