#!/usr/bin/env python3
"""
Script de Treinamento Avançado com Dataset Misto e Augmentation
Foco: Mitigação de vulnerabilidade contra deepfakes de alta qualidade.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import logging
import shutil
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdvancedTraining")

# Adicionar diretório app ao path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.domain.models.augmentation.advanced_audio_augmentation import create_robust_training_dataset
    from app.domain.models.architectures.registry import create_model_by_name
    from app.core.training.secure_training_pipeline import SecureTrainingPipeline, SecureTrainingConfig
    from stress_test_simulation import generate_tone
    import scipy.io.wavfile as wav
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)

def generate_synthetic_dataset(output_dir: Path, count: int = 50):
    """Gera dataset sintético 'Next Gen' para enriquecer o treinamento."""
    logger.info(f"Gerando {count} amostras sintéticas de alta qualidade...")
    
    fake_dir = output_dir / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 3.0
    
    for i in range(count):
        # Gerar tom com baixíssimo jitter/shimmer (Next Gen Simulation)
        freq = np.random.uniform(100, 800)
        # Jitter/Shimmer muito baixos para simular deepfake perfeito
        audio = generate_tone(freq, duration, sr, jitter=0.0001, shimmer=0.0001, noise_level=0.001)
        
        filename = fake_dir / f"synthetic_nextgen_{i:04d}.wav"
        wav.write(filename, sr, (audio * 32767).astype(np.int16))
    
    logger.info("Dataset sintético gerado com sucesso.")

def main():
    logger.info("🚀 INICIANDO TREINAMENTO AVANÇADO (MISTO + AUGMENTATION)")
    
    # 1. Configurar Diretórios
    datasets_dir = Path("datasets/raw")
    features_dir = Path("datasets/features")
    models_dir = Path("models")
    
    datasets_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Gerar Dados Sintéticos (Enriquecimento)
    # Se não tiver dados reais suficientes, o script pode falhar ou overfitar
    # Vamos assumir que o usuário já tem alguns dados reais ou vamos gerar mock reais também
    
    # Verificar dados reais
    real_dir = datasets_dir / "real"
    if not real_dir.exists() or len(list(real_dir.glob("*.wav"))) < 5:
        logger.warning("Poucos dados reais encontrados. Gerando mocks reais para demonstração...")
        real_dir.mkdir(parents=True, exist_ok=True)
        sr = 16000
        for i in range(10):
            # Real tem jitter/shimmer naturais maiores
            audio = generate_tone(np.random.uniform(100, 800), 3.0, sr, jitter=0.01, shimmer=0.05, noise_level=0.02)
            wav.write(real_dir / f"mock_real_{i:04d}.wav", sr, (audio * 32767).astype(np.int16))
            
    # Gerar Fake Next Gen
    generate_synthetic_dataset(datasets_dir, count=20)
    
    # 3. Extração de Features (Simplificada para Demo/Rapidez)
    # Em produção, usaria o SegmentedFeatureExtractor completo
    # Aqui vamos carregar os dados raw e extrair features on-the-fly ou usar um mock de features
    # Para usar o SecureTrainingPipeline, precisamos de X e y
    
    logger.info("Carregando e processando áudios...")
    
    X = []
    y = []
    
    # Helper para processar áudio
    def process_audio(data):
        if len(data) == 16000:
            return data
        elif len(data) > 16000:
            mid = len(data)//2
            return data[mid-8000:mid+8000]
        else:
            # Pad
            pad_len = 16000 - len(data)
            return np.pad(data, (0, pad_len), 'constant')

    # Carregar Reais (Label 0)
    for f in real_dir.glob("*.wav"):
        try:
            rate, data = wav.read(f)
            sample = process_audio(data)
            # Normalizar
            sample = sample.astype(np.float32) / 32768.0
            X.append(sample)
            y.append(0)
        except Exception as e:
            logger.error(f"Erro ao ler {f}: {e}")
            
    # Carregar Fakes (Label 1) - Incluindo os gerados
    fake_dir = datasets_dir / "fake"
    for f in fake_dir.glob("*.wav"):
        try:
            rate, data = wav.read(f)
            sample = process_audio(data)
            sample = sample.astype(np.float32) / 32768.0
            X.append(sample)
            y.append(1)
        except Exception as e:
            logger.error(f"Erro ao ler {f}: {e}")
            
    X = np.array(X)
    # Expandir dimensões para (N, 16000, 1) para ser compatível com Keras/RawNet2
    if len(X.shape) == 2:
        X = np.expand_dims(X, -1)
    y = np.array(y)
    
    logger.info(f"Dados carregados: {len(X)} amostras (Shape: {X.shape})")
    
    if len(X) == 0:
        logger.error("Nenhum dado para treinar.")
        return

    # 4. Configurar Pipeline de Treinamento Seguro
    secure_config = SecureTrainingConfig(
        test_size=0.2,
        validation_size=0.2,
        random_state=42
    )
    
    # Dividir dados manualmente pois o SecureTrainingPipeline espera features extraídas
    # Mas aqui estamos passando raw audio para usar augmentation
    # Vamos adaptar: vamos treinar um modelo RawNet2 ou AASIST que aceita raw audio
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * 0.8)
    
    X_train = X[indices[:split]]
    y_train = y[indices[:split]]
    X_val = X[indices[split:]]
    y_val = y[indices[split:]]
    
    # 5. Criar Dataset com Augmentation Avançada
    logger.info("Aplicando Advanced Audio Augmentation (incluindo Smoothing)...")
    
    # Configuração forte para forçar robustez
    aug_config = {
        'sample_rate': 16000,
        'max_duration': 1.0,
        'noise_factor': 0.01,
        'apply_probability': 0.8,
        'mixup_alpha': 0.4  # Mixup para misturar real e fake
    }
    
    train_dataset = create_robust_training_dataset(
        X_train, y_train,
        batch_size=8,
        augmentation_config=aug_config,
        use_mixup=True
    )
    
    # Converter validação para one-hot também, pois o loss será categorical_crossentropy
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=2)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot)).batch(8)
    
    # 6. Criar e Treinar Modelo
    logger.info("Inicializando arquitetura RawNet2 (otimizada para raw audio)...")
    
    # Usar registry para criar modelo de forma segura
    try:
        model = create_model_by_name(
            architecture_name="RawNet2",
            input_shape=(16000, 1),
            num_classes=2,
            safe_mode=True
        )
    except Exception as e:
        logger.error(f"Erro ao criar modelo: {e}")
        # Tentar fallback para AASIST se RawNet2 falhar
        logger.info("Tentando fallback para AASIST...")
        model = create_model_by_name(
            architecture_name="AASIST",
            input_shape=(16000, 1),
            num_classes=2,
            safe_mode=True
        )
    
    # Compilar
    # Usar categorical_crossentropy pois mixup/augmentation gera targets one-hot/soft labels
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar
    logger.info("Iniciando fit...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,  # Demo rápido
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # 7. Salvar Modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"rawnet2_robust_nextgen_{timestamp}"
    model_path = models_dir / f"{model_name}.h5"
    
    model.save(str(model_path))
    logger.info(f"✅ Modelo salvo em: {model_path}")
    
    # Salvar relatório simples
    acc = history.history['val_accuracy'][-1]
    logger.info(f"📊 Acurácia de Validação: {acc:.4f}")

    print("\n" + "="*50)
    print(f"🎉 TREINAMENTO CONCLUÍDO! Modelo: {model_name}")
    print("="*50)
    print("O modelo foi treinado com:")
    print("1. Dataset misto (Reais + Sintéticos Next Gen)")
    print("2. Augmentation avançada (Smoothing, Mixup, Noise)")
    print("3. Arquitetura RawNet2 (Raw Audio)")
    print("\nAgora você pode usar o painel Gradio para testar este modelo.")

if __name__ == "__main__":
    main()
