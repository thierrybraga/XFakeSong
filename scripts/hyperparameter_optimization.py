#!/usr/bin/env python3
"""
Script de Otimização de Hiperparâmetros para Todas as Arquiteturas

Este script implementa otimização sistemática de hiperparâmetros para todas as
arquiteturas disponíveis no projeto de detecção de deepfakes.

Arquiteturas suportadas:
- AASIST (Anti-spoofing Audio Spoofing and Deepfake Detection)
- RawGAT-ST (Raw Graph Attention Spatio-Temporal Network)
- Conformer (Convolution-augmented Transformer)
- EfficientNet-LSTM (EfficientNet with LSTM)
- MultiscaleCNN (Multi-Scale Convolutional Neural Network)
- SpectrogramTransformer (Transformer for Spectrogram Analysis)
- Ensemble (Combination of multiple architectures)
- SVM (Support Vector Machine)
- RandomForest (Random Forest Classifier)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.domain.features.segmented_feature_loader import SegmentedFeatureLoader
from app.domain.models.architectures.registry import architecture_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Classe principal para otimização de hiperparâmetros."""
    
    def __init__(self, data_path: str = "datasets/features/segmented"):
        self.data_path = data_path
        self.results = {}
        self.best_models = {}
        self.optimization_history = []
        
        # Configurar GPU se disponível
        self._setup_gpu()
        
        # Carregar dados
        self._load_data()
        
    def _setup_gpu(self):
        """Configura GPU para otimização."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configurada: {len(gpus)} dispositivos encontrados")
            except RuntimeError as e:
                logger.warning(f"Erro ao configurar GPU: {e}")
        else:
            logger.info("Executando em CPU")
    
    def _load_data(self):
        """Carrega dados para otimização."""
        try:
            logger.info("Carregando dados para otimização...")
            
            # Usar SegmentedFeatureLoader para carregar dados
            feature_loader = SegmentedFeatureLoader(base_path=self.data_path)
            
            # Carregar features e labels
            self.X_train, self.y_train = feature_loader.load_train_data()
            self.X_val, self.y_val = feature_loader.load_validation_data()
            self.X_test, self.y_test = feature_loader.load_test_data()
            
            logger.info(f"Dados carregados:")
            logger.info(f"  Treino: {self.X_train.shape} features, {len(self.y_train)} labels")
            logger.info(f"  Validação: {self.X_val.shape} features, {len(self.y_val)} labels")
            logger.info(f"  Teste: {self.X_test.shape} features, {len(self.y_test)} labels")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            # Fallback: gerar dados sintéticos para demonstração
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Gera dados sintéticos para demonstração."""
        logger.warning("Gerando dados sintéticos para demonstração...")
        
        n_samples = 1000
        n_features = 80
        
        # Gerar features sintéticas
        np.random.seed(42)
        self.X_train = np.random.randn(int(n_samples * 0.6), n_features)
        self.X_val = np.random.randn(int(n_samples * 0.2), n_features)
        self.X_test = np.random.randn(int(n_samples * 0.2), n_features)
        
        # Gerar labels sintéticos
        self.y_train = np.random.randint(0, 2, int(n_samples * 0.6))
        self.y_val = np.random.randint(0, 2, int(n_samples * 0.2))
        self.y_test = np.random.randint(0, 2, int(n_samples * 0.2))
        
        logger.info("Dados sintéticos gerados com sucesso")
    
    def get_hyperparameter_spaces(self) -> Dict[str, Dict[str, List]]:
        """Define espaços de hiperparâmetros para cada arquitetura."""
        
        return {
            "AASIST": {
                "dropout_rate": [0.1, 0.2, 0.3, 0.4],
                "l2_reg_strength": [0.0001, 0.0005, 0.001, 0.005],
                "hidden_dim": [256, 512, 1024],
                "num_layers": [6, 8, 10, 12],
                "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
                "batch_size": [16, 32, 64],
                "architecture": ["default", "cnn_baseline", "bidirectional_gru", "aasist"]
            },
            
            "RawGAT-ST": {
                "dropout_rate": [0.1, 0.2, 0.3, 0.4],
                "l2_reg_strength": [0.0001, 0.0005, 0.001, 0.005],
                "hidden_dim": [256, 512, 1024],
                "num_layers": [6, 8, 10],
                "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
                "batch_size": [16, 24, 32],
                "architecture": ["default", "cnn_baseline", "rawgat_st"]
            },
            
            "Conformer": {
                "dropout_rate": [0.05, 0.1, 0.15, 0.2],
                "l2_reg_strength": [0.0001, 0.0002, 0.0005],
                "attention_heads": [4, 8, 12, 16],
                "hidden_dim": [256, 512, 768],
                "num_layers": [4, 6, 8, 10],
                "learning_rate": [0.0001, 0.0003, 0.0005],
                "batch_size": [16, 20, 32],
                "conv_kernel_size": [3, 5, 7, 9]
            },
            
            "EfficientNet-LSTM": {
                "lstm_units": [128, 256, 512, 1024],
                "attention_units": [64, 128, 256, 512],
                "dropout_rate": [0.2, 0.3, 0.4, 0.5],
                "learning_rate": [0.0001, 0.0005, 0.001],
                "batch_size": [16, 32, 64],
                "architecture": ["efficientnet_lstm"]
            },
            
            "MultiscaleCNN": {
                "filters": [[32, 64, 128], [64, 128, 256], [64, 128, 256, 512]],
                "kernel_sizes": [[3, 5], [3, 5, 7], [3, 5, 7, 9]],
                "dropout_rate": [0.1, 0.2, 0.3, 0.4],
                "l2_reg_strength": [0.0001, 0.0003, 0.0005],
                "learning_rate": [0.0005, 0.001, 0.002],
                "batch_size": [32, 40, 64],
                "architecture": ["multiscale_cnn"]
            },
            
            "SpectrogramTransformer": {
                "d_model": [256, 512, 768],
                "num_heads": [8, 12, 16, 20],
                "num_blocks": [6, 8, 10, 12],
                "dropout_rate": [0.05, 0.1, 0.15],
                "learning_rate": [0.0001, 0.0003, 0.0005],
                "batch_size": [8, 16, 24],
                "patch_size": [(4, 4), (8, 8), (16, 16)],
                "ff_dim": [512, 1024, 2048]
            },
            
            "SVM": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                "degree": [2, 3, 4, 5],  # Para kernel poly
                "probability": [True, False]
            },
            
            "RandomForest": {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None, 0.5],
                "bootstrap": [True, False]
            }
        }
    
    def optimize_deep_learning_architecture(self, arch_name: str, param_space: Dict, 
                                          n_trials: int = 20) -> Dict[str, Any]:
        """Otimiza hiperparâmetros para arquiteturas de deep learning."""
        
        logger.info(f"Iniciando otimização para {arch_name}...")
        
        best_score = 0.0
        best_params = None
        best_model = None
        trial_results = []
        
        # Gerar combinações de parâmetros usando Random Search
        param_combinations = list(ParameterGrid(param_space))
        np.random.shuffle(param_combinations)
        
        # Limitar número de trials
        param_combinations = param_combinations[:n_trials]
        
        for i, params in enumerate(param_combinations):
            try:
                logger.info(f"Trial {i+1}/{len(param_combinations)} para {arch_name}")
                logger.info(f"Parâmetros: {params}")
                
                # Criar modelo
                model = self._create_model(arch_name, params)
                
                if model is None:
                    logger.warning(f"Falha ao criar modelo para {arch_name} com parâmetros {params}")
                    continue
                
                # Treinar modelo
                history = self._train_model(model, params)
                
                # Avaliar modelo
                score = self._evaluate_model(model)
                
                # Registrar resultado
                trial_result = {
                    "trial": i + 1,
                    "params": params.copy(),
                    "score": score,
                    "history": history.history if history else None
                }
                trial_results.append(trial_result)
                
                # Atualizar melhor resultado
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_model = model
                    
                    logger.info(f"Novo melhor score para {arch_name}: {best_score:.4f}")
                
                # Limpar memória
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                logger.error(f"Erro no trial {i+1} para {arch_name}: {e}")
                continue
        
        return {
            "architecture": arch_name,
            "best_score": best_score,
            "best_params": best_params,
            "best_model": best_model,
            "trial_results": trial_results,
            "total_trials": len(trial_results)
        }
    
    def optimize_classical_ml_architecture(self, arch_name: str, param_space: Dict) -> Dict[str, Any]:
        """Otimiza hiperparâmetros para arquiteturas de ML clássico."""
        
        logger.info(f"Iniciando otimização para {arch_name}...")
        
        try:
            # Preparar dados para sklearn
            X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
            X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
            
            # Criar modelo
            if arch_name == "SVM":
                from sklearn.svm import SVC
                base_model = SVC()
            elif arch_name == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier(random_state=42)
            else:
                raise ValueError(f"Arquitetura {arch_name} não suportada")
            
            # Usar RandomizedSearchCV para otimização
            search = RandomizedSearchCV(
                base_model,
                param_space,
                n_iter=50,
                cv=3,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
            # Executar busca
            search.fit(X_train_flat, self.y_train)
            
            # Avaliar melhor modelo
            best_model = search.best_estimator_
            val_score = best_model.score(X_val_flat, self.y_val)
            
            return {
                "architecture": arch_name,
                "best_score": val_score,
                "best_params": search.best_params_,
                "best_model": best_model,
                "cv_results": search.cv_results_,
                "total_trials": len(search.cv_results_['params'])
            }
            
        except Exception as e:
            logger.error(f"Erro na otimização de {arch_name}: {e}")
            return {
                "architecture": arch_name,
                "best_score": 0.0,
                "best_params": None,
                "best_model": None,
                "error": str(e)
            }
    
    def _create_model(self, arch_name: str, params: Dict) -> Optional[tf.keras.Model]:
        """Cria modelo baseado na arquitetura e parâmetros."""
        
        try:
            input_shape = self.X_train.shape[1:]
            num_classes = 2
            
            # Extrair parâmetros específicos
            batch_size = params.pop('batch_size', 32)
            learning_rate = params.pop('learning_rate', 0.001)
            
            # Criar modelo usando registry
            if arch_name in ["AASIST", "RawGAT-ST"]:
                # Importar função de criação específica
                if arch_name == "AASIST":
                    from app.domain.models.architectures.aasist import create_model
                else:
                    from app.domain.models.architectures.rawgat_st import create_model
                
                model = create_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **params
                )
                
            elif arch_name == "Conformer":
                from app.domain.models.architectures.conformer import create_conformer_model
                model = create_conformer_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **params
                )
                
            elif arch_name == "EfficientNet-LSTM":
                from app.domain.models.architectures.efficientnet_lstm import create_efficientnet_lstm_model
                model = create_efficientnet_lstm_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **params
                )
                
            elif arch_name == "MultiscaleCNN":
                from app.domain.models.architectures.multiscale_cnn import create_multiscale_cnn_model
                model = create_multiscale_cnn_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **params
                )
                
            elif arch_name == "SpectrogramTransformer":
                from app.domain.models.architectures.spectrogram_transformer import create_spectrogram_transformer_model
                model = create_spectrogram_transformer_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **params
                )
                
            else:
                logger.error(f"Arquitetura {arch_name} não implementada")
                return None
            
            # Compilar modelo
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Erro ao criar modelo {arch_name}: {e}")
            return None
    
    def _train_model(self, model: tf.keras.Model, params: Dict) -> Optional[tf.keras.callbacks.History]:
        """Treina modelo com early stopping."""
        
        try:
            batch_size = params.get('batch_size', 32)
            epochs = 50  # Número reduzido para otimização
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Treinar modelo
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return None
    
    def _evaluate_model(self, model: tf.keras.Model) -> float:
        """Avalia modelo no conjunto de validação."""
        
        try:
            # Predições
            y_pred_proba = model.predict(self.X_val, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)
            
            # Score combinado (pode ser customizado)
            combined_score = (accuracy + f1) / 2
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {e}")
            return 0.0
    
    def run_optimization(self, architectures: Optional[List[str]] = None, 
                        n_trials_per_arch: int = 20) -> Dict[str, Any]:
        """Executa otimização para todas as arquiteturas especificadas."""
        
        if architectures is None:
            architectures = [
                "AASIST", "RawGAT-ST", "Conformer", "EfficientNet-LSTM",
                "MultiscaleCNN", "SpectrogramTransformer", "SVM", "RandomForest"
            ]
        
        logger.info(f"Iniciando otimização para {len(architectures)} arquiteturas")
        logger.info(f"Arquiteturas: {architectures}")
        
        param_spaces = self.get_hyperparameter_spaces()
        optimization_results = {}
        
        for arch_name in architectures:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Otimizando {arch_name}")
                logger.info(f"{'='*50}")
                
                if arch_name not in param_spaces:
                    logger.warning(f"Espaço de parâmetros não definido para {arch_name}")
                    continue
                
                param_space = param_spaces[arch_name]
                
                # Escolher método de otimização
                if arch_name in ["SVM", "RandomForest"]:
                    result = self.optimize_classical_ml_architecture(arch_name, param_space)
                else:
                    result = self.optimize_deep_learning_architecture(
                        arch_name, param_space, n_trials_per_arch
                    )
                
                optimization_results[arch_name] = result
                
                # Salvar resultado intermediário
                self._save_intermediate_result(arch_name, result)
                
                logger.info(f"Otimização de {arch_name} concluída")
                logger.info(f"Melhor score: {result['best_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Erro na otimização de {arch_name}: {e}")
                optimization_results[arch_name] = {
                    "architecture": arch_name,
                    "error": str(e),
                    "best_score": 0.0
                }
        
        # Salvar resultados finais
        self._save_final_results(optimization_results)
        
        return optimization_results
    
    def _save_intermediate_result(self, arch_name: str, result: Dict[str, Any]):
        """Salva resultado intermediário."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{arch_name}_{timestamp}.json"
            
            # Preparar dados para serialização
            serializable_result = result.copy()
            
            # Remover objetos não serializáveis
            if 'best_model' in serializable_result:
                del serializable_result['best_model']
            
            if 'trial_results' in serializable_result:
                for trial in serializable_result['trial_results']:
                    if 'history' in trial and trial['history']:
                        # Manter apenas métricas finais
                        history = trial['history']
                        trial['final_metrics'] = {
                            'final_loss': history.get('loss', [])[-1] if history.get('loss') else None,
                            'final_accuracy': history.get('accuracy', [])[-1] if history.get('accuracy') else None,
                            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
                            'final_val_accuracy': history.get('val_accuracy', [])[-1] if history.get('val_accuracy') else None
                        }
                        del trial['history']
            
            with open(filename, 'w') as f:
                json.dump(serializable_result, f, indent=2, default=str)
            
            logger.info(f"Resultado intermediário salvo: {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultado intermediário: {e}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Salva resultados finais da otimização."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Preparar dados para serialização
            serializable_results = {}
            
            for arch_name, result in results.items():
                serializable_result = result.copy()
                
                # Remover objetos não serializáveis
                if 'best_model' in serializable_result:
                    del serializable_result['best_model']
                
                if 'trial_results' in serializable_result:
                    for trial in serializable_result['trial_results']:
                        if 'history' in trial and trial['history']:
                            # Manter apenas métricas finais
                            history = trial['history']
                            trial['final_metrics'] = {
                                'final_loss': history.get('loss', [])[-1] if history.get('loss') else None,
                                'final_accuracy': history.get('accuracy', [])[-1] if history.get('accuracy') else None,
                                'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
                                'final_val_accuracy': history.get('val_accuracy', [])[-1] if history.get('val_accuracy') else None
                            }
                            del trial['history']
                
                serializable_results[arch_name] = serializable_result
            
            # Salvar resultados completos
            results_filename = f"hyperparameter_optimization_results_{timestamp}.json"
            with open(results_filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Criar relatório resumido
            self._create_summary_report(serializable_results, timestamp)
            
            logger.info(f"Resultados finais salvos: {results_filename}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados finais: {e}")
    
    def _create_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Cria relatório resumido dos resultados."""
        
        try:
            report_filename = f"optimization_summary_report_{timestamp}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("RELATÓRIO DE OTIMIZAÇÃO DE HIPERPARÂMETROS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Ranking das arquiteturas
                f.write("RANKING DAS ARQUITETURAS\n")
                f.write("-" * 30 + "\n")
                
                # Ordenar por score
                sorted_results = sorted(
                    [(name, res) for name, res in results.items() if 'best_score' in res],
                    key=lambda x: x[1]['best_score'],
                    reverse=True
                )
                
                for i, (arch_name, result) in enumerate(sorted_results, 1):
                    f.write(f"{i}. {arch_name}: {result['best_score']:.4f}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
                
                # Detalhes por arquitetura
                f.write("DETALHES POR ARQUITETURA\n")
                f.write("-" * 30 + "\n\n")
                
                for arch_name, result in results.items():
                    f.write(f"Arquitetura: {arch_name}\n")
                    f.write(f"Melhor Score: {result.get('best_score', 'N/A')}\n")
                    
                    if 'best_params' in result and result['best_params']:
                        f.write("Melhores Parâmetros:\n")
                        for param, value in result['best_params'].items():
                            f.write(f"  {param}: {value}\n")
                    
                    if 'total_trials' in result:
                        f.write(f"Total de Trials: {result['total_trials']}\n")
                    
                    if 'error' in result:
                        f.write(f"Erro: {result['error']}\n")
                    
                    f.write("\n" + "-" * 30 + "\n\n")
                
                # Recomendações
                f.write("RECOMENDAÇÕES\n")
                f.write("-" * 15 + "\n\n")
                
                if sorted_results:
                    best_arch, best_result = sorted_results[0]
                    f.write(f"Melhor arquitetura: {best_arch}\n")
                    f.write(f"Score: {best_result['best_score']:.4f}\n\n")
                    
                    if 'best_params' in best_result and best_result['best_params']:
                        f.write("Parâmetros recomendados:\n")
                        for param, value in best_result['best_params'].items():
                            f.write(f"  {param}: {value}\n")
                
            logger.info(f"Relatório resumido criado: {report_filename}")
            
        except Exception as e:
            logger.error(f"Erro ao criar relatório resumido: {e}")

def main():
    """Função principal."""
    
    logger.info("Iniciando otimização de hiperparâmetros...")
    
    # Criar otimizador
    optimizer = HyperparameterOptimizer()
    
    # Definir arquiteturas para otimizar
    architectures = [
        "AASIST",
        "RawGAT-ST", 
        "Conformer",
        "EfficientNet-LSTM",
        "MultiscaleCNN",
        "SpectrogramTransformer",
        "SVM",
        "RandomForest"
    ]
    
    # Executar otimização
    results = optimizer.run_optimization(
        architectures=architectures,
        n_trials_per_arch=15  # Reduzido para demonstração
    )
    
    # Exibir resumo
    logger.info("\n" + "=" * 50)
    logger.info("RESUMO DA OTIMIZAÇÃO")
    logger.info("=" * 50)
    
    for arch_name, result in results.items():
        score = result.get('best_score', 0.0)
        logger.info(f"{arch_name}: {score:.4f}")
    
    logger.info("\nOtimização concluída!")
    logger.info("Verifique os arquivos gerados para detalhes completos.")

if __name__ == "__main__":
    main()