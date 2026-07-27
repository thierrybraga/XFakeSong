"""Registry de Arquiteturas Disponíveis

Este módulo centraliza o registro de todas as arquiteturas de deep learning
disponíveis no sistema, facilitando a integração com o pipeline de detecção.
"""

import inspect
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Chaves de ``default_params`` que pertencem ao PIPELINE DE TREINO, não ao
#: construtor do modelo. Fonte única compartilhada por ``registry.create_model``
#: e por ``factory.ArchitectureFactoryRegistry`` (que antes mantinha sua própria
#: cópia da lista).
_TRAINING_ONLY_PARAM_KEYS = frozenset({
    "patience",
    "lr_patience",
    "gradient_clip",
    "augmentation_strength",
})

# NOTA sobre ``crop_strategy`` (contrato de entrada de áudio bruto):
# AASIST e RawGAT-ST usam "train_random_eval_multicrop" (crop aleatório no
# treino, multicrop na avaliação — a receita dos respectivos papers, que reduz
# variância na métrica); RawNet2, WavLM, HuBERT e Ensemble usam "center".
# A divergência é DELIBERADA e não uma inconsistência: mudar o crop de uma
# arquitetura já treinada invalidaria a comparação com o run publicado. Ao
# incluir uma arquitetura nova no benchmark, escolha explicitamente e registre
# a escolha em docs/evaluation/benchmark.md.


@dataclass
class ArchitectureInfo:
    """Informações sobre uma arquitetura."""

    name: str
    module_path: str
    function_name: str
    description: str
    supported_variants: List[str]
    # NOTA: hiperparâmetros por arquitetura vivem em 3 lugares (cuidado com drift,
    # chaves como dropout_rate/l2_reg_strength se sobrepõem):
    #   1) este `default_params` (regularização/controle: dropout, l2, patience,
    #      gradient_clip, augmentation_strength) — usado pelo training_service;
    #   2) o `create_model(...)` de cada architectures/<nome>.py (LR/optimizer/loss);
    #   3) benchmarks/planning.py::NEURAL_BENCHMARK_HPARAMS (lr/batch/scheduler/
    #      warmup), aplicado pelo benchmark quando optimize_hyperparameters=True.
    default_params: Dict[str, Any]
    input_requirements: Dict[str, Any]


class ArchitectureRegistry:
    """Registry centralizado de arquiteturas."""

    # Mapeamento bidirecional: display_name ↔ snake_case
    _DISPLAY_TO_SNAKE: Dict[str, str] = {
        "AASIST": "aasist",
        "RawGAT-ST": "rawgat_st",
        "EfficientNet-LSTM": "efficientnet_lstm",
        "MultiscaleCNN": "multiscale_cnn",
        "SpectrogramTransformer": "spectrogram_transformer",
        "Conformer": "conformer",
        "Ensemble": "ensemble",
        "Sonic Sleuth": "sonic_sleuth",
        "RawNet2": "rawnet2",
        "WavLM": "wavlm",
        "HuBERT": "hubert",
        "Hybrid CNN-Transformer": "hybrid_cnn_transformer",
    }
    _SNAKE_TO_DISPLAY: Dict[str, str] = {v: k for k, v in _DISPLAY_TO_SNAKE.items()}

    def __init__(self):
        self._architectures: Dict[str, ArchitectureInfo] = {}
        self._register_default_architectures()

    def _register_default_architectures(self):
        """Registra as arquiteturas padrão do sistema."""

        # AASIST
        self.register(
            ArchitectureInfo(
                name="AASIST",
                module_path="app.domain.models.architectures.aasist",
                function_name="create_model",
                description="Anti-spoofing Audio Spoofing and Deepfake Detection - configuração otimizada para reduzir overfitting",
                supported_variants=[
                    "aasist",
                    "aasist_legacy",
                    "default",
                    "cnn_gru_simple",
                    "cnn_baseline",
                    "bidirectional_gru",
                    "resnet_gru",
                    "transformer",
                ],
                default_params={
                    # create_model params: dropout_rate, l2_reg_strength, hidden_dim, num_layers
                    # AJUSTE (retune): subajuste (val_loss travada ~1.0, acc ~0.92)
                    # + recall colapsa sob ruido (0.29 @10dB). Reduz regularizacao
                    # excessiva e reforca augmentation p/ robustez.
                    # CORREÇÃO 2026-07-15: learning_rate/l2_reg_strength tinham
                    # revertido para 1e-4/1e-4 (drift silencioso vs. o retune
                    # documentado aqui e em benchmarks/planning.py). Restaurado
                    # para 3e-4/2e-4 — mantém as 3 fontes de hparams em sincronia.
                    "dropout_rate": 0.2,
                    "l2_reg_strength": 0.0002,
                    "classifier_head": "cross_entropy",
                    "learning_rate": 0.0003,
                    "min_learning_rate": 0.000005,
                    "decay_steps": 100000,
                    # (hidden_dim/num_layers REMOVIDOS: a variante
                    # paper-faithful "aasist" tem topologia fixa pelo artigo e
                    # os ignora — só as variantes legadas CNN/GRU os usam.
                    # Eram config morto, mesma classe já limpa no Ensemble.)
                    # Training params (used by pipeline, not by create_model)
                    "patience": 25,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.35,
                },
                input_requirements={
                    "input_type": "raw_audio",
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "min_sequence_length": 16000,
                    "target_sequence_length": 48000,
                    "crop_strategy": "train_random_eval_multicrop",
                    "feature_frontend": "benchmark_raw_v1",
                    "source_samples": 48000,
                    "max_duration": 3.0,
                    "preprocessing": "normalize",
                },
            )
        )

        # RawGAT-ST
        self.register(
            ArchitectureInfo(
                name="RawGAT-ST",
                module_path="app.domain.models.architectures.rawgat_st",
                function_name="create_model",
                description="End-to-End Spectro-Temporal Graph Attention (Tak et al., 2021) — SincNet sobre áudio bruto + grafo espectral (Gs) e temporal (Gt) com fusão element-wise",
                supported_variants=[
                    "rawgat_st",
                    "rawgat_st_legacy",
                    "default",
                    "rawgat_st_paper",
                    "rawgat_st_fast",
                    "rawgat_st_stable",
                    "rawgat_st_optimized",
                    "cnn_gru_simple",
                    "cnn_baseline",
                    "bidirectional_gru",
                    "resnet_gru",
                    "transformer",
                ],
                default_params={
                    # create_model params: dropout_rate, l2_reg_strength, attention_heads, hidden_dim, num_layers
                    # AJUSTE (retune): pior modelo - val_acc cai apos epoca 4 e
                    # val_loss dispara (0.39->1.85) enquanto treino sobe = overfit/
                    # divergencia. Mais dropout + L2, clip mais apertado, mais
                    # augmentation e paciencia maior p/ achar minimo melhor.
                    "dropout_rate": 0.35,
                    "l2_reg_strength": 0.001,
                    "learning_rate": 0.00005,
                    "min_learning_rate": 0.000005,
                    "decay_steps": 100000,
                    # (attention_heads/hidden_dim/num_layers/
                    # temporal_pool_stride/fusion_mode REMOVIDOS: a variante
                    # paper-faithful "rawgat_st" segue a topologia do artigo
                    # — dois encoders 2D, GAT S/T, fusão element-wise e um
                    # terceiro GAT — e IGNORA todos eles. Eram config morto;
                    # continuam válidos apenas para as variantes legadas.)
                    # Training params (used by pipeline, not by create_model)
                    "patience": 25,
                    "lr_patience": 8,
                    "gradient_clip": 0.5,
                    "augmentation_strength": 0.4,
                },
                input_requirements={
                    # Opera sobre ÁUDIO BRUTO (SincNet front-end), igual ao
                    # AASIST. (Antes estava 'spectrogram', divergindo do paper.)
                    "input_type": "raw_audio",
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "min_sequence_length": 16000,
                    "target_sequence_length": 48000,
                    "crop_strategy": "train_random_eval_multicrop",
                    "feature_frontend": "benchmark_raw_v1",
                    "source_samples": 48000,
                    "max_duration": 3.0,
                    "preprocessing": "normalize",
                },
            )
        )

        # EfficientNet-LSTM
        self.register(
            ArchitectureInfo(
                name="EfficientNet-LSTM",
                module_path="app.domain.models.architectures.efficientnet_lstm",
                function_name="create_model",
                description="EfficientNet with LSTM for temporal modeling - configuração padrão para máxima acurácia",
                supported_variants=["efficientnet_lstm", "efficientnet_lstm_lite"],
                default_params={
                    # create_model params: lstm_units (int), dropout_rate
                    # AJUSTE (retune): acc limpa baixa (0.929) porem robusto -
                    # reduz dropout e da mais paciencia p/ ganhar acuracia.
                    "lstm_units": 256,
                    "dropout_rate": 0.25,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 20,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3,
                },
                input_requirements={
                    # CORREÇÃO: a implementação (efficientnet_lstm.py) consome
                    # ESPECTROGRAMA (redimensionado p/ o backbone EfficientNet);
                    # o valor antigo "raw_audio" divergia do contrato real da
                    # factory e da própria rede.
                    "input_type": "spectrogram",
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80,
                    "sample_rate": 16000,
                    # 3 s: mesma janela canônica do protocolo (48.000 amostras
                    # a 16 kHz) usada pelas demais arquiteturas.
                    "max_duration": 3.0,
                },
            )
        )

        # MultiscaleCNN (Res2Net — Gao et al., TPAMI 2021)
        self.register(
            ArchitectureInfo(
                name="MultiscaleCNN",
                module_path="app.domain.models.architectures.multiscale_cnn",
                function_name="create_model",
                description="Res2Net: Multi-scale backbone with hierarchical residual connections (Gao et al., TPAMI 2021). Res2Net-50 config: scale=4, baseWidth=26, [3,4,6,3] blocks.",
                supported_variants=[
                    "multiscale_cnn",
                    "multiscale_cnn_lite",
                    "multiscale_cnn_se",
                    "multiscale_cnn_optimized",
                ],
                default_params={
                    # create_model params: base_width, scale, layer_config, dropout_rate
                    # AJUSTE 2026-07-14: dropout 0.2->0.5 — overfit severo
                    # (train 100% / val 64,5%); o 0.5 do plano de benchmark
                    # nunca chegava ao modelo (config morto). Acompanha
                    # Adam->AdamW com weight_decay real no builder
                    # (multiscale_cnn.py). Em sincronia com planning.py.
                    "base_width": 26,
                    "scale": 4,
                    "use_se": False,
                    "dropout_rate": 0.5,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.2,
                    "augmentation_strength": 0.35,
                },
                input_requirements={
                    "input_type": "spectrogram",
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80,
                },
            )
        )

        # SpectrogramTransformer
        self.register(
            ArchitectureInfo(
                name="SpectrogramTransformer",
                module_path="app.domain.models.architectures.spectrogram_transformer",
                function_name="create_model",
                description="Audio Spectrogram Transformer (AST) - ViT-Base with overlapping patches for audio deepfake detection",
                supported_variants=[
                    "spectrogram_transformer",        # ViT-Base do paper AST
                    "spectrogram_transformer_small",  # ViT-Small p/ treino do zero
                    "spectrogram_transformer_lite",
                ],
                default_params={
                    # create_model params: AST/ViT-Base trained from scratch.
                    # AJUSTE 2026-07-14: blocos pre-LN (paper) + LR de pico
                    # 5e-5→1e-5 e weight_decay 1e-4→1e-5 — o treino degradava
                    # lentamente até chute aleatório (EER final ~51%) mesmo
                    # após o fix de decay_steps. Em sincronia com
                    # spectrogram_transformer.py::create_spectrogram_transformer_model
                    # e benchmarks/planning.py.
                    "patch_size": (16, 16),
                    "stride": (10, 10),
                    "embed_dim": 768,
                    "num_blocks": 12,
                    "num_heads": 12,
                    "ff_dim": 3072,
                    "dropout_rate": 0.3,
                    "learning_rate": 1e-5,
                    "warmup_steps": 2000,
                    "decay_steps": 50000,
                    "weight_decay": 1e-5,
                    "alpha": 1e-7,
                    "clipnorm": 1.0,
                    # Default FALSE de propósito: `pretrained=True` baixa o
                    # checkpoint AudioSet (~350 MB) e exige rede, o que
                    # quebraria testes/CI offline. O BENCHMARK liga a flag via
                    # benchmarks/planning.py — é lá que a decisão científica
                    # (partir de pesos pré-treinados, como o artigo) é tomada.
                    "pretrained": False,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 25,
                    "lr_patience": 12,
                    "gradient_clip": 0.5,
                    "augmentation_strength": 0.2,
                },
                input_requirements={
                    # CORREÇÃO 2026-07-27 (conformidade com Gong et al., 2021):
                    # o contrato anterior (100×80) produzia apenas 63 tokens
                    # com patch 16×16/stride 10 — contra os 1212 do artigo —
                    # alimentando um ViT-Base de 85M parâmetros. Era o regime
                    # que degradava o treino até chute aleatório (EER ~51%).
                    # 300 quadros × 128 mel = o front-end do paper (hop de
                    # 10 ms, 128 bandas) aplicado à janela canônica de 3 s,
                    # e resulta em 29×12 = 348 tokens.
                    "input_type": "spectrogram",
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 300,
                    "feature_dim": 128,
                    # Janela de 25 ms (400 amostras a 16 kHz), como no artigo.
                    # O default do projeto é 512 (32 ms).
                    "n_fft": 400,
                },
            )
        )

        # Conformer
        self.register(
            ArchitectureInfo(
                name="Conformer",
                module_path="app.domain.models.architectures.conformer",
                function_name="create_model",
                description="Conformer: Convolution-augmented Transformer - configuração padrão para máxima acurácia",
                # CONSOLIDAÇÃO 2026-07-27: configuração ÚNICA (Conformer-M do
                # paper: 16 blocos). 'conformer_lite' permanece apenas como
                # ALIAS legado e resolve para a mesma topologia — antes as duas
                # variantes tinham os nomes invertidos (a "lite" era 2× maior).
                supported_variants=["conformer", "conformer_m", "conformer_lite"],
                default_params={
                    # create_model params: dropout_rate, learning_rate, weight_decay,
                    # warmup_steps, decay_steps, alpha, clipnorm, label_smoothing
                    # Training params (used by pipeline, not by create_model)
                    "patience": 22,
                    "lr_patience": 11,
                    "gradient_clip": 0.6,
                    "augmentation_strength": 0.25,
                },
                input_requirements={
                    "input_type": "spectrogram",
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80,
                },
            )
        )

        # Ensemble
        self.register(
            ArchitectureInfo(
                name="Ensemble",
                module_path="app.domain.models.architectures.ensemble",
                function_name="create_model",
                description="Multi-spectrogram ensemble (Mel+LFCC+CQT) with MLP fusion — Pham et al. 2024",
                supported_variants=[
                    "ensemble",
                    "ensemble_score",
                    "ensemble_lite",
                    "ensemble_adaptive",
                ],
                default_params={
                    "dropout_rate": 0.3,
                    # (use_mfcc_branch/use_cross_attention/use_gated_fusion/
                    # use_se_blocks/aux_loss_weight removidos: NENHUM builder
                    # de ensemble.py os aceita como parâmetro — eram specs
                    # documentais sem efeito, e o repasse cego via **kwargs
                    # quebrava a criação com TypeError. Os quatro branches +
                    # cross-attention + gated fusion são fixos na variante
                    # "ensemble"; ver _create_ensemble_feature_fusion.)
                    # Training params
                    # AJUSTE (retune): falha catastrofica de robustez - a 10dB a
                    # acc vira 0.50 e recall->0 (predicts tudo como "real"). Precisa
                    # ver fakes ruidosos no treino: augmentation forte + paciencia.
                    "patience": 20,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.45,
                },
                input_requirements={
                    # AJUSTE: era "spectrogram", divergindo do contrato real
                    # em architecture_factory_registry (factory.py) e da
                    # implementação (ensemble.py computa Mel/LFCC/CQT
                    # internamente via camadas TF custom a partir de audio
                    # bruto — SharedSTFTLayer et al.). feature_preparer.py usa
                    # este registry como fallback de inferencia quando o
                    # input_contract do modelo salvo esta ausente; a entrada
                    # divergente fazia esse fallback preparar espectrograma
                    # em vez de audio bruto.
                    "input_type": "raw_audio",
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "min_sequence_length": 16000,
                    "target_sequence_length": 48000,
                    "crop_strategy": "center",
                },
            )
        )

        # Sonic Sleuth (Alshehri et al., MDPI Computers 2024)
        self.register(
            ArchitectureInfo(
                name="Sonic Sleuth",
                module_path="app.domain.models.architectures.sonic_sleuth",
                function_name="create_model",
                description="Sonic Sleuth (Alshehri et al., 2024): LFCC/MFCC/CQT feature extraction + 3×Conv2D(32→64→128) + Dense(256→128) + Dropout(0.1). Best: LFCC 98.27% accuracy.",
                supported_variants=[
                    "sonic_sleuth",
                    "sonic_sleuth_paper",
                    "sonic_sleuth_mfcc",
                    "sonic_sleuth_cqt",
                    "sonic_sleuth_lfcc_cqt",
                ],
                default_params={
                    # Estes parâmetros eram CONFIG MORTO: o builder só lia
                    # `sample_rate` e a topologia/dropout eram fixos no código.
                    # Agora todos têm efeito real (ver sonic_sleuth.py); os
                    # valores abaixo reproduzem o modelo já treinado. A
                    # configuração LITERAL da Figura 3 do artigo está na
                    # variante "sonic_sleuth_paper".
                    "sample_rate": 16000,
                    "use_batch_norm": True,
                    "num_conv_blocks": 5,
                    "use_residual": True,
                    "use_se_blocks": True,
                    "use_gap_gmp": True,
                    "dropout_rate": 0.3,
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3,
                },
                input_requirements={
                    "input_type": "spectrogram",
                    "type": "features",
                    "format": "spectrogram",
                    "max_duration": 3.0,
                    "sample_rate": 16000,
                    "preprocessing": "log_mel",
                    # Alvo de espectrograma (como as demais arquiteturas de
                    # espectrograma). Sem isto, o adaptador do benchmark mantinha
                    # a forma de entrada crua e os 5 blocos de pooling colapsavam
                    # entradas pequenas (< 32×32) → "Negative dimension".
                    "min_sequence_length": 100,
                    "feature_dim": 80,
                },
            )
        )

        # RawNet2
        self.register(
            ArchitectureInfo(
                name="RawNet2",
                module_path="app.domain.models.architectures.rawnet2",
                function_name="create_model",
                description=(
                    "RawNet2 sobre áudio bruto (SincNet + blocos residuais com "
                    "FMS + GRU). ATENÇÃO à variante: o default segue o "
                    "'Improved RawNet' de VERIFICAÇÃO DE LOCUTOR (Jung et al., "
                    "2020 — Sinc 128, canais 128/256, 1×GRU); o baseline de "
                    "ANTI-SPOOFING do ASVspoof 2021 (Sinc 20, canais 20/128, "
                    "3×GRU), que é o comparado na literatura da tarefa, está em "
                    "'rawnet2_antispoofing'."
                ),
                supported_variants=[
                    "rawnet2",                # Improved RawNet (Jung 2020, SV)
                    "rawnet2_antispoofing",   # baseline ASVspoof 2021
                    "rawnet2_lite",
                ],
                default_params={
                    # create_model params: sinc_filters, sinc_kernel_size, res_filters, gru_units, gru_layers, dense_units, dropout_rate
                    "sinc_filters": 128,
                    "sinc_kernel_size": 1024,
                    "res_filters": [128, 128, 256, 256, 256, 256],
                    "gru_units": 1024,
                    "gru_layers": 1,
                    "dense_units": 1024,
                    "dropout_rate": 0.3,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3,
                },
                input_requirements={
                    "input_type": "raw_audio",
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "min_sequence_length": 16000,
                    "target_sequence_length": 48000,
                    "crop_strategy": "center",
                    # max_duration COERENTE com target_sequence_length (48.000
                    # amostras a 16 kHz = 3 s). Estava em 5.0 e o fallback do
                    # FeaturePreparer (usado quando o modelo salvo não traz
                    # input_shape) montava 80.000 amostras — uma janela que o
                    # modelo nunca viu no treino.
                    "max_duration": 3.0,
                    "preprocessing": "normalize",
                },
            )
        )

        # WavLM
        self.register(
            ArchitectureInfo(
                name="WavLM",
                module_path="app.domain.models.architectures.wavlm",
                function_name="create_model",
                description="Arquitetura de dois estágios com WavLM pré-treinado como extrator de características e classificador MLP - configuração padrão para máxima acurácia",
                supported_variants=["wavlm", "wavlm_lite", "wavlm_aasist"],
                default_params={
                    # create_model params: wavlm_model, freeze_wavlm, classifier_units, dropout_rate
                    # Checkpoint REAL: desde 2026-07-27 os pesos são lidos do
                    # checkpoint PyTorch e portados para Keras (o caminho TF do
                    # `transformers` não funciona com Keras 3). O backbone fica
                    # CONGELADO e só a cabeça treina — receita padrão de
                    # downstream com SSL. Ver ssl_backbone.py.
                    "wavlm_model": "microsoft/wavlm-base",
                    "freeze_wavlm": True,
                    "classifier_units": [1024, 512, 256],
                    "dropout_rate": 0.2,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3,
                },
                input_requirements={
                    "input_type": "raw_audio",
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "min_sequence_length": 16000,
                    "target_sequence_length": 48000,
                    "crop_strategy": "center",
                    # Coerente com target_sequence_length (3 s @ 16 kHz);
                    # estava 10.0 e contradizia a própria janela declarada.
                    "max_duration": 3.0,
                    "preprocessing": "normalize",
                },
            )
        )

        # HuBERT
        self.register(
            ArchitectureInfo(
                name="HuBERT",
                module_path="app.domain.models.architectures.hubert",
                function_name="create_model",
                description="Arquitetura baseada em HuBERT (Hidden-Unit BERT) para detecção de deepfakes em áudio bruto - fidelidade ao paper",
                supported_variants=["hubert", "hubert_lite", "hubert_aasist"],
                default_params={
                    # create_model params: model_name, freeze_hubert, classifier_hidden_dim, dropout_rate
                    "model_name": "facebook/hubert-base-ls960",
                    "freeze_hubert": True,
                    "classifier_hidden_dim": 256,
                    "dropout_rate": 0.3,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3,
                },
                input_requirements={
                    # CONTRATO COMPLETADO: faltavam min_sequence_length,
                    # target_sequence_length e crop_strategy — o HuBERT era a
                    # única arquitetura de áudio bruto sem contrato temporal, e
                    # o fallback do FeaturePreparer derivava 10 s (160.000
                    # amostras) de max_duration, contra os 48.000 do protocolo.
                    "input_type": "raw_audio",
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "min_sequence_length": 16000,
                    "target_sequence_length": 48000,
                    "crop_strategy": "center",
                    "max_duration": 3.0,
                    "preprocessing": "normalize",
                },
            )
        )

        # Hybrid CNN-Transformer
        self.register(
            ArchitectureInfo(
                name="Hybrid CNN-Transformer",
                module_path="app.domain.models.architectures.hybrid_cnn_transformer",
                function_name="create_model",
                description="CCT (Compact Convolutional Transformer) — Hassani et al. 2021, adapted for audio deepfake per Bartusiak & Delp 2022",
                supported_variants=[
                    "hybrid_cnn_transformer",
                    "hybrid_cnn_transformer_lite",
                ],
                default_params={
                    # create_model params: projection_dim, num_heads, transformer_layers,
                    #   conv_channels, dropout_rate, stochastic_depth_rate, use_positional_emb
                    "projection_dim": 256,
                    "num_heads": 4,
                    "transformer_layers": 4,
                    "conv_channels": [64, 128],
                    # AJUSTE (retune): robustez moderada (0.97->0.785 @10dB).
                    # Mais dropout/stochastic-depth e augmentation p/ generalizar.
                    "dropout_rate": 0.2,
                    "stochastic_depth_rate": 0.15,
                    "use_positional_emb": True,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 18,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.4,
                },
                input_requirements={
                    "input_type": "spectrogram",
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80,
                    "sample_rate": 16000,
                    "max_duration": 3.0,
                    "preprocessing": "spectrogram_or_raw",
                    "supports_1d_input": True,
                    "supports_2d_input": True,
                },
            )
        )

    def register(self, architecture_info: ArchitectureInfo):
        """Registra uma nova arquitetura."""
        self._architectures[architecture_info.name] = architecture_info
        logger.info(f"Arquitetura {architecture_info.name} registrada com sucesso")

    def get_architecture(self, name: str) -> ArchitectureInfo:
        """Obtém informações de uma arquitetura."""
        if name not in self._architectures:
            raise ValueError(
                f"Arquitetura '{name}' não encontrada. Disponíveis: {list(self._architectures.keys())}"
            )
        return self._architectures[name]

    def list_architectures(self) -> List[str]:
        """Lista todas as arquiteturas disponíveis."""
        return list(self._architectures.keys())

    def get_all_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Retorna todas as arquiteturas registradas."""
        return self._architectures.copy()

    # ── Name Mapping Layer ──────────────────────────────────────────────

    @classmethod
    def to_snake_case(cls, name: str) -> str:
        """Converte display name → snake_case. Se já for snake, retorna como está."""
        if name in cls._DISPLAY_TO_SNAKE:
            return cls._DISPLAY_TO_SNAKE[name]
        # Já é snake_case ou desconhecido
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def to_display_name(cls, snake: str) -> str:
        """Converte snake_case → display name para UI."""
        if snake in cls._SNAKE_TO_DISPLAY:
            return cls._SNAKE_TO_DISPLAY[snake]
        return snake

    @classmethod
    def normalize_architecture_name(cls, name: str) -> str:
        """Normaliza qualquer formato de nome para o display name do registry.

        Aceita tanto 'sonic_sleuth' quanto 'Sonic Sleuth' e retorna
        o display name canônico registrado no registry.
        """
        # Se já é um display name válido
        if name in cls._DISPLAY_TO_SNAKE:
            return name
        # Tentar converter de snake_case
        if name in cls._SNAKE_TO_DISPLAY:
            return cls._SNAKE_TO_DISPLAY[name]
        # Fallback: tentar case-insensitive
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        for snake, display in cls._SNAKE_TO_DISPLAY.items():
            if snake == name_lower:
                return display
        compact = re.sub(r"[^a-z0-9]+", "", name.lower())
        for display in cls._DISPLAY_TO_SNAKE:
            if re.sub(r"[^a-z0-9]+", "", display.lower()) == compact:
                return display
        raise ValueError(
            f"Arquitetura '{name}' não encontrada. "
            f"Disponíveis (snake): {list(cls._SNAKE_TO_DISPLAY.keys())}"
        )

    def get_architecture_by_any_name(self, name: str) -> ArchitectureInfo:
        """Busca arquitetura aceitando display name OU snake_case."""
        try:
            return self.get_architecture(name)
        except ValueError:
            display = self.normalize_architecture_name(name)
            return self.get_architecture(display)

    def list_architectures_snake(self) -> List[str]:
        """Lista todas as arquiteturas em snake_case (para dropdowns/API)."""
        return [self.to_snake_case(name) for name in self._architectures.keys()]

    def list_architecture_choices(self) -> List[Tuple[str, str]]:
        """Retorna pares (display_label, snake_case) para dropdowns Gradio."""
        choices = []
        for display_name in self._architectures:
            snake = self.to_snake_case(display_name)
            info = self._architectures[display_name]
            input_type = info.input_requirements.get("type", "unknown")
            label = f"{display_name} ({input_type})"
            choices.append((label, snake))
        return choices

    def get_active_config(
        self, architecture_name: str, variant: str = "default"
    ) -> Dict[str, Any]:
        """Obtém a configuração ativa do banco de dados (ou default se falhar)."""
        try:
            from app.core.db.session import SessionLocal
            from app.domain.models.architecture_config import ArchitectureConfig

            db_session = SessionLocal()
            try:
                # Tentar buscar no DB usando SQLAlchemy nativo
                config = (
                    db_session.query(ArchitectureConfig)
                    .filter_by(
                        architecture_name=architecture_name,
                        variant_name=variant,
                        is_active=True,
                    )
                    .first()
                )

                if config:
                    return config.parameters

                # Fallback se não encontrar variante específica: tentar default
                if variant != "default":
                    config = (
                        db_session.query(ArchitectureConfig)
                        .filter_by(
                            architecture_name=architecture_name,
                            variant_name="default",
                            is_active=True,
                        )
                        .first()
                    )
                    if config:
                        return config.parameters
            finally:
                db_session.close()

        except Exception as e:
            logger.warning(
                f"Não foi possível carregar config do DB para {architecture_name}: {e}. Usando hardcoded."
            )

        # Fallback final: Hardcoded
        return self.get_architecture(architecture_name).default_params

    def create_model(
        self,
        architecture_name: str,
        input_shape: Tuple[int, ...],
        num_classes: int = 2,
        variant: str = None,
        safe_mode: bool = True,
        **kwargs,
    ):
        """Cria um modelo usando a arquitetura especificada.

        Args:
            architecture_name: Nome da arquitetura
            input_shape: Forma do input
            num_classes: Número de classes
            variant: Variante da arquitetura
            safe_mode: Se True, aplica correções para prevenir data leakage
            **kwargs: Parâmetros adicionais
        """
        arch_info = self.get_architecture(architecture_name)

        # Importar dinamicamente o módulo
        module = __import__(arch_info.module_path, fromlist=[arch_info.function_name])
        create_model_func = getattr(module, arch_info.function_name)

        # CORREÇÃO: este caminho IGNORAVA o próprio `default_params` do
        # registry — construía o modelo apenas com os defaults da assinatura de
        # cada `create_model`. Resultado: `registry.create_model(...)` e
        # `factory.create_model_by_name(...)` produziam modelos com
        # hiperparâmetros diferentes. Agora ambos partem do mesmo default,
        # excluindo as chaves que pertencem ao pipeline de treino (a factory
        # aplica exatamente a mesma exclusão).
        params = {
            key: value
            for key, value in arch_info.default_params.items()
            if key not in _TRAINING_ONLY_PARAM_KEYS
        }

        # Adicionar variant se especificado
        if variant:
            if variant not in arch_info.supported_variants:
                raise ValueError(
                    f"Variant '{variant}' não suportada para {architecture_name}. "
                    f"Disponíveis: {arch_info.supported_variants}"
                )
            params["architecture"] = variant
        else:
            # Usar primeira variante como padrão
            if arch_info.supported_variants:
                params["architecture"] = arch_info.supported_variants[0]

        # Adicionar kwargs do usuário
        params.update(kwargs)

        # Filtrar params para apenas os aceitos pela função (quando não há **kwargs)
        sig = inspect.signature(create_model_func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            accepted = set(sig.parameters.keys())
            filtered_out = [k for k in params if k not in accepted]
            if filtered_out:
                logger.debug(
                    f"{architecture_name}: ignoring unsupported params: {filtered_out}"
                )
            params = {k: v for k, v in params.items() if k in accepted}

        # Criar modelo
        model = create_model_func(input_shape, num_classes, **params)

        # Validação de leakage REAL (Lambdas de pré-processamento suspeitas).
        # NOTA: BatchNormalization não é mais tratada como leakage nem
        # reescrita automaticamente — ver docstring de architecture_patcher.
        if safe_mode:
            # Import tardio: architecture_patcher exige TensorFlow no topo do
            # arquivo. Um import eager aqui forçaria TF em qualquer consumidor
            # de registry.py (incl. ambientes classical-ml sem TF instalado).
            from .architecture_patcher import patch_architecture_for_safety

            model = patch_architecture_for_safety(model)

        return model

    def validate_input_shape(
        self, architecture_name: str, input_shape: Tuple[int, ...]
    ) -> bool:
        """Valida se o input_shape é compatível com a arquitetura.

        CORREÇÃO: esta função rejeitava áudio bruto 1-D — ``(48000,)`` caía no
        ``len(input_shape) < 2`` e voltava False, enquanto a factory
        (``BaseArchitectureFactory.validate_input_shape``) aceitava a mesma
        forma. Duas respostas opostas para o mesmo contrato. Agora ambas seguem
        a semântica de ``input_type``:

        - ``raw_audio``   → ``(T,)`` ou ``(T, 1)``
        - ``spectrogram`` → ``(T, F)`` ou ``(T, F, 1)``
        - ausente/legado  → validação frouxa por ``min_sequence_length``/
          ``feature_dim`` (compat com SVM/RF e contratos antigos).
        """
        arch_info = self.get_architecture(architecture_name)
        requirements = arch_info.input_requirements
        input_type = requirements.get("input_type", "any")

        if not input_shape:
            return False

        min_len = requirements.get("min_sequence_length")

        if input_type == "raw_audio":
            if min_len and input_shape[0] < min_len:
                return False
            # (T,) e (T, 1) são válidos; (T, K>1) não é áudio bruto.
            if len(input_shape) == 2 and input_shape[-1] != 1:
                return False
            return True

        if input_type == "spectrogram":
            if len(input_shape) < 2:
                return False
            if min_len and input_shape[0] < min_len:
                return False
            expected_feature_dim = requirements.get("feature_dim")
            if expected_feature_dim and int(input_shape[1]) != int(expected_feature_dim):
                return False
            if len(input_shape) == 3 and input_shape[2] != 1:
                return False
            return True

        # Caminho legado (contratos sem input_type).
        if len(input_shape) < 2:
            return False

        sequence_length, feature_dim = input_shape[0], input_shape[1]
        if min_len and sequence_length < min_len:
            return False

        # A chave era "min_feature_dim", que nenhuma arquitetura define (a
        # validação sempre passava). O contrato real usa "feature_dim" com
        # igualdade exata (mesma regra da factory).
        expected_feature_dim = requirements.get("feature_dim")
        if (
            expected_feature_dim
            and feature_dim not in (None, 1)
            and int(feature_dim) != int(expected_feature_dim)
        ):
            return False

        return True

    def sync_defaults_to_db(self):
        """Sincroniza os parâmetros padrão do registry para o banco de dados."""
        try:
            from app.core.db.session import SessionLocal
            from app.domain.models.architecture_config import ArchitectureConfig

            db_session = SessionLocal()
            try:
                # Iterar sobre todas as arquiteturas registradas
                for name, info in self._architectures.items():
                    # Verificar se já existe configuração default
                    config = (
                        db_session.query(ArchitectureConfig)
                        .filter_by(architecture_name=name, variant_name="default")
                        .first()
                    )

                    if not config:
                        logger.info(f"Criando configuração default no DB para {name}")
                        new_config = ArchitectureConfig(
                            architecture_name=name,
                            variant_name="default",
                            description=info.description,
                            parameters=info.default_params,
                            is_active=True,
                        )
                        db_session.add(new_config)

                db_session.commit()
                logger.info("Sincronização de configurações padrão concluída.")
            finally:
                db_session.close()
        except Exception as e:
            logger.error(f"Erro ao sincronizar defaults para o DB: {e}")


# Instância global do registry
architecture_registry = ArchitectureRegistry()

# Funções de conveniência


def get_available_architectures() -> List[str]:
    """Retorna lista de arquiteturas disponíveis."""
    return architecture_registry.list_architectures()


def create_model_by_name(
    architecture_name: str,
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    variant: str = None,
    safe_mode: bool = True,
    **kwargs,
):
    """Cria um modelo pela nome da arquitetura.

    Args:
        architecture_name: Nome da arquitetura
        input_shape: Forma do input
        num_classes: Número de classes
        variant: Variante da arquitetura
        safe_mode: Se True, aplica correções para prevenir data leakage
        **kwargs: Parâmetros adicionais
    """
    return architecture_registry.create_model(
        architecture_name, input_shape, num_classes, variant, safe_mode, **kwargs
    )


def create_safe_model_by_name(
    architecture_name: str,
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    variant: str = None,
    **kwargs,
):
    """Cria um modelo seguro (sem data leakage) pela nome da arquitetura."""
    return create_model_by_name(
        architecture_name, input_shape, num_classes, variant, safe_mode=True, **kwargs
    )


def get_architecture_info(architecture_name: str) -> ArchitectureInfo:
    """Obtém informações sobre uma arquitetura."""
    return architecture_registry.get_architecture(architecture_name)


def get_architecture_by_any_name(architecture_name: str) -> ArchitectureInfo:
    """Obtém arquitetura aceitando display name, snake_case ou alias comum."""
    return architecture_registry.get_architecture_by_any_name(architecture_name)


def validate_architecture_input(
    architecture_name: str, input_shape: Tuple[int, ...]
) -> bool:
    """Valida input para uma arquitetura."""
    return architecture_registry.validate_input_shape(architecture_name, input_shape)


@lru_cache(maxsize=64)
def load_hyperparameters_json(
    architecture_name: str, results_dir: str
) -> Dict[str, Any]:
    """Carrega hiperparâmetros recomendados de um arquivo JSON.

    Cacheado por ``(architecture_name, results_dir)``: o caminho de inferência
    (FeaturePreparer) chamava isto a cada request, lendo disco repetidamente.
    O retorno é tratado como somente-leitura pelos callers.

    Args:
        architecture_name: Nome da arquitetura
        results_dir: Diretório onde estão os resultados/configs

    Returns:
        Dict com hiperparâmetros ou dict vazio se não encontrar
    """
    import json
    import os

    # Normalizar nome para busca de arquivo (ex: "Random Forest" ->
    # "random_forest")
    safe_name = architecture_name.lower().replace(" ", "_")

    # Tentar variações de nomes de arquivo
    possible_files = [
        f"{safe_name}_hyperparameters.json",
        f"{safe_name}_params.json",
        f"{safe_name}.json",
    ]

    for filename in possible_files:
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao ler hiperparâmetros de {file_path}: {e}")
                return {}

    return {}


def get_architecture_choices() -> List[Tuple[str, str]]:
    """Retorna pares (display_label, snake_case) para dropdowns."""
    return architecture_registry.list_architecture_choices()


def get_valid_snake_names() -> set:
    """Retorna set de snake_case names válidos (para validação de schemas)."""
    return set(architecture_registry.list_architectures_snake())


def normalize_arch_name(name: str) -> str:
    """Normaliza qualquer formato para display name canônico."""
    return ArchitectureRegistry.normalize_architecture_name(name)


def to_snake(name: str) -> str:
    """Converte qualquer formato para snake_case."""
    return ArchitectureRegistry.to_snake_case(name)


# Exportar principais classes e funções
__all__ = [
    "ArchitectureInfo",
    "ArchitectureRegistry",
    "architecture_registry",
    "get_available_architectures",
    "create_model_by_name",
    "create_safe_model_by_name",
    "get_architecture_info",
    "get_architecture_by_any_name",
    "validate_architecture_input",
    "load_hyperparameters_json",
    "get_architecture_choices",
    "get_valid_snake_names",
    "normalize_arch_name",
    "to_snake",
]
