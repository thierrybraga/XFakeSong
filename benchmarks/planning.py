"""Preflight e plano executável do benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict

import numpy as np

from benchmarks.config import BenchmarkConfig

CLASSICAL_ARCHES = {"svm", "randomforest"}
HEAVY_ARCHES = {
    "rawnet2",
    "aasist",
    "rawgatst",
    "spectrogramtransformer",
}
ARCH_ALIASES = {
    "cct": "hybridcnntransformer",
    "ast": "spectrogramtransformer",
    "audiospectrogramtransformer": "spectrogramtransformer",
    "res2net": "multiscalecnn",
    "randomforest": "randomforest",
}

# Hiperparâmetros recomendados por arquitetura para o benchmark (aplicados
# quando cfg.optimize_hyperparameters=True; ver _merge_effective_hparams).
# Este é 1 dos 3 locais de hparams por arquitetura — mantenha em sincronia com:
#   - app/domain/models/architectures/registry.py::default_params (regularização);
#   - o create_model(...) de cada architectures/<nome>.py (LR/optimizer/loss).
# Chaves sobrepostas (dropout_rate, l2_reg_strength) podem divergir — revise as 3
# fontes ao ajustar um modelo.
#: Custo de treino por arquitetura — a base dos timeouts por modelo.
#:
#: MEDIDO em 2026-07-28 num Ryzen 5 7600X (12 threads), com o LOTE e a PRECISÃO
#: que o plano efetivamente aplica (ver `_fit_to_device`), sobre a janela real
#: de 48.000 amostras. Valores em horas para o orçamento canônico: 100 épocas
#: sobre `_REFERENCE_FIT_SAMPLES` amostras.
#:
#: A coluna GPU é ESTIMADA por roofline (RTX 3060: ~12,7 TFLOPS FP32 e 360 GB/s
#: contra ~1,0–1,3 TFLOPS e ~83 GB/s do CPU), com multiplicador por família:
#: operação densa chega perto de 20–30×, atenção em grafo — que materializa
#: tensores pareados nó×nó e fica limitada por banda — fica em 12–25×. Os
#: valores de GPU são o CENÁRIO PROVÁVEL; o conservador é ~2× maior, e é por
#: isso que o fator de segurança do timeout não é pequeno.
#:
#: Ao mudar lote, precisão, janela ou arquitetura, remeça: um timeout derivado
#: de número velho mata um treino bom.
_REFERENCE_FIT_SAMPLES = 66452  # 33.226 de treino + 1 cópia AWGN
_REFERENCE_EPOCHS = 100

#: Coluna ``gpu`` RECALIBRADA em 2026-08-02 com a campanha de 2026-08-01/02
#: (RTX 3060 12 GB, dataset de 40.980, 100 épocas). Os valores anteriores eram
#: estimativas e erravam para os dois lados — o RawNet2 em 3,6× para MENOS
#: (18 h declaradas contra 65,1 h reais), acima do fator de segurança de 3×, o
#: que faria o timeout matar o treino por volta da época 83; WavLM e HuBERT em
#: 25× para MAIS, porque a estimativa assumia o backbone rodando a cada época
#: quando o runner extrai embeddings uma vez e treina só a cabeça.
#:
#: ``medido`` = tempo real do run. ``extrapolado`` = não concluiu nenhuma vez
#: neste protocolo; o valor antigo foi corrigido pelo erro de calibração
#: observado na arquitetura MEDIDA da mesma família, que é a melhor informação
#: disponível — melhor que manter um número que já se provou errado por
#: construção. Substituir por medida assim que rodarem até o fim.
#:
#: A coluna ``cpu`` vem de uma campanha anterior e NÃO foi remedida: o
#: benchmark roda em GPU. Os valores seguem só como ordem de grandeza.
EXPECTED_TRAINING_HOURS: Dict[str, Dict[str, float]] = {
    # arquitetura (chave compacta): {"cpu": campanha antiga, "gpu": ver acima}
    "hubertoriginal": {"cpu": 201.4, "gpu": 0.21},  # medido
    "wavlmoriginal": {"cpu": 205.9, "gpu": 0.24},  # medido
    # As variantes com fine-tuning (`hubertaasist`/`wavlmaasist`) saíram do
    # manifesto oficial em 2026-08-11 — ver a justificativa em
    # `benchmarks/config.py`. As estimativas ficam aqui porque o caminho de
    # código continua disponível como ablação: o backbone entra no grafo de
    # gradiente e roda a cada época, em vez de uma vez só para gerar embeddings
    # em cache, daí as duas ordens de grandeza contra as congeladas acima.
    "hubertaasist": {"cpu": 900.0, "gpu": 10.0},
    "wavlmaasist": {"cpu": 900.0, "gpu": 10.0},
    "sonicsleuth": {"cpu": 7.2, "gpu": 0.4},  # escopo estendido, não medido
    # AJUSTE 2026-08-09 — o custo dos clássicos MUDOU DE ORDEM com o retune:
    # o grid saiu de 24 para 108 candidatos (RF) e de 12 para 24 (SVM), as
    # dobras de 3 para 5, o conjunto da busca de 12.162 amostras limpas para
    # 24.324 (limpo + cópia AWGN) e o vetor de 63 para 183 colunas. São 540
    # ajustes de floresta contra 72, e 120 de SVC contra 36 — cada um sobre um
    # problema 2x maior em amostras e 3x em features.
    #
    # Os valores antigos (0,42 h e 0,93 h) derivavam um timeout de 30 min, que
    # MATOU o RandomForest aos 1800,6 s antes de fechar a busca.
    #
    # MEDIDO no retreino de 2026-08-09 sobre 24.324 amostras de ajuste:
    # RandomForest 2.117,1 s (0,588 h) e SVM 3.614,8 s (1,004 h). Normalizado à
    # referência de 66.452 (a escala é linear em `fit_samples`): 1,61 h e
    # 2,74 h. O perfil de CPU segue extrapolado do fator observado na campanha
    # antiga, já que esta medição foi em máquina com GPU ociosa mas trabalho
    # 100% em CPU — para SVM/RF os dois perfis são o mesmo trabalho.
    "randomforest": {"cpu": 1.61, "gpu": 1.61},  # medido (retune 2026-08-09)
    "svm": {"cpu": 2.74, "gpu": 2.74},  # medido (retune 2026-08-09)
    "hybridcnntransformer": {"cpu": 83.0, "gpu": 2.45},  # medido
    "conformer": {"cpu": 41.0, "gpu": 2.88},  # medido
    "efficientnetlstm": {"cpu": 59.3, "gpu": 3.0},  # escopo estendido
    "ensemble": {"cpu": 89.2, "gpu": 3.6},  # escopo estendido
    "multiscalecnn": {"cpu": 40.4, "gpu": 3.67},  # medido (2 épocas no 15k)
    "hubert": {"cpu": 201.4, "gpu": 5.8},  # escopo estendido
    "wavlm": {"cpu": 205.9, "gpu": 5.9},  # escopo estendido
    # AJUSTE 2026-08-15 — as quatro entradas abaixo eram EXTRAPOLAÇÕES e
    # superestimavam o custo real em 1,9x a 5,5x. Substituídas pelo MEDIDO no
    # run `clean_benchmark_15k` (100 épocas, 24.324 amostras de ajuste),
    # normalizado a `_REFERENCE_FIT_SAMPLES`:
    #
    #   arquitetura   real (h)   normalizado   antigo   erro
    #   AST              11,01          30,1     57,0   1,9x
    #   RawNet2           4,29          11,7     65,1   5,5x
    #   AASIST           15,44          42,2    101,0   2,4x
    #   RawGAT-ST        26,85          73,4    196,0   2,7x
    #
    # Superestimar não mata treino (o timeout fica frouxo), mas desinforma o
    # planejamento de custo — e o `CLAUDE.md` exige que esta tabela acompanhe
    # qualquer mudança de protocolo. Com o fator de segurança 3x, o timeout
    # derivado do RawGAT-ST cai de 215 h para ~81 h, ainda 3x o real.
    "spectrogramtransformer": {"cpu": 996.0, "gpu": 30.1},  # medido
    # RawNet2 — ESTIMATIVA REVISADA em 2026-08-20, não medida.
    #
    # As 11,7 h medidas são do "Improved RawNet" de verificação de locutor
    # (Sinc 128, canais 128/256, 1×GRU(1024)), que era o construído até então.
    # O escopo oficial passou a treinar o baseline anti-spoofing de Tak et al.
    # (Sinc 20, canais 20/128, 3×GRU(1024)): as convoluções ficam MUITO mais
    # baratas (20/128 contra 128/256 canais) mas a recorrência TRIPLICA, e a
    # GRU é a parte sequencial que não paraleliza na GPU — no RawNet2 ela
    # domina o passo. Estimativa conservadora de 1,6x sobre o medido, à espera
    # da primeira medição real. Subestimar aqui é o modo de falha caro: o
    # timeout derivado (3x) mataria o treino perto do fim.
    "rawnet2": {"cpu": 580.0, "gpu": 18.7},  # estimado (era 364.0/11.7, SV)
    "aasist": {"cpu": 842.4, "gpu": 42.2},  # medido
    "rawgatst": {"cpu": 1359.4, "gpu": 73.4},  # medido
}

#: Margem sobre a estimativa. 3× cobre o cenário conservador (~2×) e ainda
#: sobra folga: o timeout existe para matar um treino TRAVADO, não um lento.
DEFAULT_TIMEOUT_SAFETY_FACTOR = 3.0

#: Arquiteturas que treinam em float32 puro na GPU. Não é preferência de
#: precisão: com `mixed_float16` o processo morre de SIGSEGV no backward.
#:
#: - ``rawnet2``: SincConv + GRU.
#: - ``multiscalecnn``: reproduzido em 2026-08-02 num repro mínimo (entrada
#:   log-mel 100x80, batch 32, RTX 3060, XLA já desligado). O primeiro passo
#:   de treino COMPLETA e o processo morre no segundo, sempre — foi o
#:   `returncode=-11` aos 336 s no benchmark de 2026-08-01. Isolado assim:
#:   float32 roda cinco passos limpos; forward puro em float16 roda seis
#:   passos limpos; `TF_CUDNN_USE_AUTOTUNE=0` não muda nada. Ou seja, o crash
#:   está no backward em fp16, não na escolha de kernel do autotune nem no
#:   forward. A suspeita é o gradiente do split/concat hierárquico de canais
#:   do bloco Bottle2neck do Res2Net, o único padrão que esta arquitetura tem
#:   e as outras não; o build de 2026-08-01 subiu `nvidia-cudnn-cu12` de
#:   9.1.0.70 para 9.24.0.43.
#:
#: O custo é velocidade, não resultado: nenhuma métrica depende da precisão do
#: acumulador. Revisar quando o cuDNN/TF do container mudar.
_MIXED_PRECISION_UNSAFE_ARCHITECTURES = frozenset(
    {"rawnet2", "multiscalecnn", "aasist"}
)


def expected_training_timeout_min(
    arch: str,
    device_profile: str = "gpu",
    epochs: int = _REFERENCE_EPOCHS,
    fit_samples: int = _REFERENCE_FIT_SAMPLES,
    safety_factor: float = DEFAULT_TIMEOUT_SAFETY_FACTOR,
    minimum_min: float = 30.0,
) -> float:
    """Timeout em minutos para UMA arquitetura, derivado do custo estimado.

    Escala linearmente com épocas e com o tamanho do conjunto de treino, então
    um run reduzido (`--epochs 20`) não herda o timeout do run completo.

    Arquiteturas desconhecidas caem no maior valor da tabela: é preferível
    esperar demais a matar um treino de dias por causa de um nome novo.
    """
    key = _compact(arch)
    profile = "cpu" if str(device_profile).lower() == "cpu" else "gpu"
    entry = EXPECTED_TRAINING_HOURS.get(key)
    if entry is None:
        base_hours = max(v[profile] for v in EXPECTED_TRAINING_HOURS.values())
    else:
        base_hours = entry[profile]

    if key not in CLASSICAL_ARCHES:
        base_hours *= max(1, int(epochs)) / _REFERENCE_EPOCHS
    base_hours *= max(1, int(fit_samples)) / _REFERENCE_FIT_SAMPLES
    return max(float(minimum_min), base_hours * 60.0 * float(safety_factor))


NEURAL_BENCHMARK_HPARAMS: Dict[str, Dict[str, Any]] = {
    "rawnet2": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.3,
        "l2_reg_strength": 1e-4,
        "optimizer": "Adam",
        "scheduler": "architecture_default",
        "use_augmentation": False,
        "use_mixed_precision": False,
        "early_stopping": False,
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        # TOPOLOGIA DO BASELINE ANTI-SPOOFING (Tak et al., 2021).
        #
        # Estes valores precisam viver AQUI, e não só no runner: a interface
        # Gradio resolve os defaults por `effective_hyperparameters()`, que lê
        # este dicionário sobre o `registry.default_params`. O registry descreve
        # o "Improved RawNet" de VERIFICAÇÃO DE LOCUTOR (sinc 128, canais
        # 128/256, 1×GRU) — outra arquitetura com o mesmo nome. Enquanto a
        # topologia anti-spoofing existiu apenas em `benchmarks/runner.py`, a
        # interface exibia e treinava a variante de locutor enquanto o benchmark
        # treinava a de anti-spoofing: a "quarta fonte de verdade" que a
        # CLAUDE.md alerta, agora com duas arquiteturas diferentes sob um nome.
        #
        # A definição canônica é `rawnet2.py::_RAWNET2_ANTISPOOFING_PARAMS`;
        # `tests/unit/test_xla_guard_manifest.py` não cobre isto, mas
        # `tests/unit/test_rawnet2_variante_oficial.py` trava a igualdade entre
        # as duas cópias.
        "sinc_filters": 20,
        "sinc_kernel_size": 1024,
        "res_filters": [20, 20, 128, 128, 128, 128],
        "gru_units": 1024,
        "gru_layers": 3,
        "dense_units": 1024,
        "notes": "Raw waveform + Sinc/GRU: recorte central 1s, mixed precision desligado; LR segue o compile da arquitetura. 2026-07-14: topologia corrigida p/ paridade com o paper — MaxPool(3) após cada bloco residual (GRU passa a ver ~7 passos, não ~590) e FMS mul+add. 2026-08-20: escopo oficial passou ao baseline ANTI-SPOOFING de Tak et al. (sinc 20, canais 20/128, 3×GRU), no lugar do Improved RawNet de verificação de locutor.",
    },
    "aasist": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 24,
        # AJUSTE (retune): LR 1e-4->3e-4 e l2 1e-4->2e-4, em sincronia com
        # aasist.py::create_model e registry.py::default_params (ver
        # docs/evaluation/retraining-adjustments.md). Augmentation ligado — subajuste + colapso
        # de recall sob ruído (0.29 @10dB) no diagnóstico original.
        # CORREÇÃO 2026-07-15: o valor estava revertido para 1e-4/1e-4 (drift
        # silencioso — o comentário acima já documentava 3e-4/2e-4 como a
        # decisão vigente). Restaurado para bater com o que o comentário e o
        # docs/evaluation/retraining-adjustments.md sempre descreveram.
        "learning_rate": 3e-4,
        "min_learning_rate": 5e-6,
        "decay_steps": 100000,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 2e-4,
        "classifier_head": "cross_entropy",
        # (attention_heads/hidden_units removidos: o create_model do AASIST
        # não os aceita — eram filtrados pela assinatura, config morto.)
        "optimizer": "AdamW",
        "scheduler": "CosineDecay",
        "use_augmentation": True,
        # 2026-08-04: era True. O AASIST divergia para NaN de forma
        # determinística no batch 502 da época 1, três execuções seguidas
        # (TerminateOnNaN). A justificativa anterior — "Sinc e logits em
        # float32 nas próprias camadas, o encoder 2D/GAT usa Tensor Cores com
        # loss scaling automático" — cobre só metade do problema: o loss
        # scaling protege o BACKWARD (detecta inf/NaN no gradiente e pula o
        # passo) e não faz nada quando o overflow nasce no FORWARD, que é o
        # risco do softmax de atenção do GAT em float16. Confirmado por A/B
        # com mesma LR (3e-4), mesmo lote (24), mesmo seed e mesmos dados:
        # em float32 a época 1 fecha com loss=0.669 e val_accuracy=0.709.
        # Sem custo de tempo — a época levou 11,1 min contra ~12 min em fp16.
        "use_mixed_precision": False,
        "recommended_epochs": 100,
        "notes": (
            "Sinc 2D + GAT S/T + master/HS-GAL/MGO; janela canônica 48.000 "
            "(3 s @ 16 kHz), crop aleatório no treino e multicrop na avaliação."
        ),
    },
    "rawgatst": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        # AJUSTE (retune): LR 1e-4->5e-5, dropout 0.2->0.35 e l2 1e-4->1e-3,
        # em sincronia com rawgat_st.py::create_model e
        # registry.py::default_params (ver docs/evaluation/retraining-adjustments.md).
        # Augmentation ligado — pior modelo do recorte, overfit/divergência
        # após a época 4 no diagnóstico original.
        #
        # AJUSTE 2026-08-06 (clean_benchmark_15k): o retune anterior não bastou.
        # Continua o pior do escopo oficial — acurácia 87,55%, EER 11,87% e min
        # t-DCF 0,3149, ABAIXO de SVM e RandomForest no t-DCF. O padrão é
        # sobreajuste, não subajuste: treino 0,998 contra val 0,85 no melhor
        # checkpoint (época 17), `val_loss` mínimo 0,511 (10x o do Hybrid)
        # subindo a 1,18 no fim. A robustez ainda sai NÃO MONOTÔNICA (77,21%
        # a 10 dB contra 77,64% a 5 dB), sinal de superfície de decisão
        # instável. Sobe dropout 0.35->0.5 e weight_decay 1e-3->3e-3; LR fica
        # em 5e-5 (o problema não é passo grande, é capacidade sem freio).
        #
        # ─── REVERSÃO PARCIAL 2026-08-17, medida ───────────────────────────
        # O ajuste acima mudou DOIS fatores de uma vez e nunca foi executado.
        # O fatorial de `scripts/benchmark/run_rawgat_retune.py` rodou os dois
        # braços isolados, ambos já com `decay_steps` = 152.100:
        #
        #   publicado  dropout 0,35 + L2 1e-3 -> pico val_acc 0,8997 (ép. 88)
        #   braço (d)  dropout 0,50 + L2 1e-3 -> val_acc 0,5000 EXATO, ép. 1-25
        #   braço (l)  dropout 0,35 + L2 3e-3 -> pico val_acc 0,8984 (ép. 40)
        #
        # Dropout 0,50 é o fator LETAL: com ele o treino chegou a 95,4%
        # enquanto a validação ficou colada no acaso — memoriza e não
        # generaliza, não é subajuste. L2 3e-3 isolado não move o teto
        # (-0,13 p.p.) e só agrava a oscilação. A célula combinada (0,50 +
        # 3e-3) que estava aqui é a única não medida, e o fatorial prevê que
        # ela falha pelo dropout.
        #
        # Volta a 0,35/1e-3 (a célula com o melhor teto medido). O que fica
        # das correções é o que realmente tem evidência: `decay_steps`
        # completo e, para runs novos, `checkpoint_monitor=val_eer` — que
        # sozinho recupera os 5,29 p.p. perdidos na seleção da época.
        "learning_rate": 5e-5,
        "min_learning_rate": 5e-6,
        # 100.000 era menor que o orçamento real: com batch 16 são
        # ceil(24.324/16) = 1.521 passos/época x 100 = 152.100 passos, então o
        # cosseno zerava na época ~66 e as últimas 34 rodavam no piso de 5e-6.
        "decay_steps": 152100,
        "epochs": 100,
        "dropout_rate": 0.35,
        "l2_reg_strength": 1e-3,
        # Passou a ser knob de verdade em 2026-08-06 (era o literal 0.7 no
        # compile de rawgat_st.py, enquanto o registry declarava 0.5 e o valor
        # nunca chegava ao modelo). 0.5 = o que o registry já dizia.
        "global_clipnorm": 0.5,
        "optimizer": "AdamW",
        "scheduler": "CosineDecay",
        "use_augmentation": True,
        # A/B retunado em 2026-07-17: o híbrido oscilou até loss=6.75,
        # enquanto o controle float32 ficou em loss=0.63/0.69/0.64 e
        # val_loss=0.58/0.58/0.58. Mantém o pipeline confirmatório em
        # float32; a implementação híbrida permanece disponível para estudo.
        "use_mixed_precision": False,
        "recommended_epochs": 100,
        "notes": (
            "Dois encoders 2D + GAT S/T + produto + terceiro GAT; janela "
            "canônica 48.000 (3 s @ 16 kHz), crop aleatório no treino e "
            "multicrop na avaliação."
        ),
    },
    "conformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        # AJUSTE 2026-08-06 (colapso irreversível em clean_benchmark_15k): com
        # LR de pico 1e-4 e warmup de 1.500 passos (~2 épocas com batch 32), o
        # treino divergiu na época ~14 e caiu para `loss = ln 2 = 0.693` /
        # `val_accuracy = 0.500`, ficando morto da época 22 à 100. Duas sessões
        # independentes colapsaram do mesmo jeito, então não é acaso de seed. O
        # número publicado (99,49%) vem do checkpoint da época 10 — 10 das 100
        # épocas do orçamento declarado. LR 1e-4->5e-5 e warmup 1500->3000
        # (~4 épocas): a topologia é pre-LN Macaron (estável por construção),
        # o que restava era o pico de LR sustentado.
        "learning_rate": 5e-5,
        "warmup_steps": 3000,
        # `decay_steps` era omitido aqui, então caía no default 50.000 de
        # `create_conformer_model`. Com batch 32 são ceil(24.324/32) = 761
        # passos/época x 100 = 76.100 passos REAIS: o cosseno zerava (alpha) na
        # época ~66 e as últimas 34 rodavam a 1e-7. Mesmo mismatch já
        # diagnosticado no Hybrid CNN-Transformer e no AST.
        "decay_steps": 76100,
        "alpha": 1e-7,
        "epochs": 100,
        # AJUSTE 2026-07-27 (consolidação do Conformer): dropout_rate 0.3->0.1.
        # O valor 0.3 nunca chegou ao encoder — a variante sobrescrevia o
        # dropout por módulo (ff=0.2/attn=0.1/conv=0.1) e o ConvSubsampling
        # tinha 0.1 hardcoded, de modo que 0.3 só afetava a cabeça de
        # classificação. Agora `dropout_rate` vale para o encoder inteiro,
        # então usamos o P_drop=0.1 do paper (Gulati et al.) — mantém o
        # comportamento efetivo anterior e passa a ser um knob real.
        "dropout_rate": 0.1,
        "l2_reg_strength": 1e-4,
        "weight_decay": 1e-4,
        "clipnorm": 1.0,
        "label_smoothing": 0.05,
        # (attention_heads/hidden_units REMOVIDOS: o runner só promove
        # lr/weight_decay/warmup/decay/alpha/dropout/clipnorm/label_smoothing
        # para `parameters`, então essas duas chaves NUNCA chegavam ao
        # construtor — o modelo sempre rodou com num_heads=4/d_model=256 do
        # Conformer-M. Eram config morto, como as já removidas do AASIST.)
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": (
            "Conformer-M do paper (16 blocos, d_model=256, 4 cabeças, "
            "d_ff=1024, kernel 31, P_drop=0.1) — configuração ÚNICA desde "
            "2026-07-27. Compile-respect: LR/weight_decay/clipnorm/dropout são "
            "passados ao construtor."
        ),
    },
    "hybridcnntransformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        # AJUSTE 2026-07-14: LR 1e-3->3e-4 e schedule agora passado ao
        # construtor do CCT (antes o compile era hardcoded lr=1e-3 e
        # decay_steps=50000 — com batch 32 sao ~657 passos/epoca x 100 =
        # 65700 passos reais, o LR zerava (alpha) na epoca ~76; mesmo
        # mismatch de decay_steps ja diagnosticado no AST). Colapso sob
        # ruido (AUC 0.46 @10dB) tratado com pico de LR menor + retreino
        # com a copia ruidosa do protocolo.
        "learning_rate": 3e-4,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1500,
        # AJUSTE 2026-08-16: era 65700, que supõe 21.024 amostras de ajuste
        # (65700/100x32). O conjunto real tem 24.324 (12.162 limpas + 1 cópia
        # AWGN), logo ceil(24324/32)x100 = 76.100. Com 65700 o cosseno atingia
        # o piso na época ~86 e as 14 últimas rodavam com LR congelado.
        "decay_steps": 76_100,
        "alpha": 1e-7,
        "clipnorm": 1.0,
        # (base_filters/num_residual_blocks/num_transformer_layers/
        # attention_heads removidos: o builder CCT usa projection_dim/
        # num_heads/transformer_layers/conv_channels do registry — as chaves
        # antigas nunca chegavam ao modelo, eram config morto.)
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "CCT compile-respect: LR/warmup/decay/weight_decay/clipnorm passados ao construtor (2026-07-14).",
    },
    "spectrogramtransformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 8,
        # AJUSTE 2026-07-14: LR 2e-5->1e-5 e weight_decay 5e-5->1e-5. Mesmo
        # com decay_steps corrigido o treino degradava lentamente ate chute
        # aleatorio (EER final ~51%); 87M params do zero pedem passo menor.
        # Acompanha a mudanca estrutural p/ blocos pre-LN (paper) em
        # spectrogram_transformer.py — em sincronia com registry.py.
        "learning_rate": 1e-5,
        "epochs": 100,
        "dropout_rate": 0.25,
        "l2_reg_strength": 1e-5,
        "weight_decay": 1e-5,
        "warmup_steps": 3000,
        # decay_steps cobre o total real de passos (100 epocas x 2625
        # passos/epoca, ja considerando o 1 copy de ruido do
        # train_noise_copies=1 que dobra 10500->21000 amostras). O valor
        # antigo (100000) fazia o LR zerar (alpha=1e-6) por volta da epoca
        # 38 e o treino degradava ate accuracy=chute aleatorio dali ate a
        # epoca 100 (diagnosticado em 2026-07-13).
        # AJUSTE 2026-08-16: era 262500, que supõe 21.000 amostras de ajuste
        # (262500/100x8) — a mesma premissa defasada do CCT. O real são 24.324,
        # logo ceil(24324/8)x100 = 304.100. Idem: o cosseno terminava na época
        # ~86 e o LR ficava no piso até o fim.
        "decay_steps": 304_100,
        "alpha": 1e-6,
        "clipnorm": 1.0,
        # AJUSTE 2026-07-27: pesos AudioSet ligados. O AST do artigo PARTE de
        # inicialização pré-treinada (ImageNet→AudioSet); treinar 85M params do
        # zero sobre este dataset é o regime que degradava até chute aleatório
        # (EER ~51%). A transferência lê o checkpoint PyTorch e escreve nas
        # camadas Keras (app/domain/models/architectures/ast_pretrained.py) —
        # validada contra o PyTorch bloco a bloco (max|dif| ~1e-5).
        # Exige rede na 1ª execução (~350 MB, cacheado depois) e FALHA ALTO se
        # indisponível — nunca degrada em silêncio para treino do zero.
        "pretrained": True,
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "use_augmentation": False,
        "early_stopping": True,
        "early_stopping_patience": 20,
        "checkpoint_best": True,
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "AST pre-LN (paper) com cabeça linear sobre o CLS, entrada 300x128 (128 mel, hop 10 ms, janela 25 ms) e pesos AudioSet transferidos do checkpoint PyTorch; LR de pico 1e-5 e weight_decay 1e-5, warmup 3000, clipnorm=1.0 e checkpoint obrigatório com restauração guardada (validada no val).",
    },
    "multiscalecnn": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 64,
        # AJUSTE 2026-07-14: LR 2e-3->1e-3 (alinha com o compile do builder)
        # e regularização EFETIVA contra o overfit train 100% / val 64,5%:
        # o dropout_rate/l2_reg_strength antigos deste plano eram config
        # morto (nunca chegavam ao create_model). Agora o dropout 0.5 flui
        # pelo registry (default_params) e o weight_decay real vem do AdamW
        # do builder (multiscale_cnn.py, default 1e-2 acoplado ao LR).
        "learning_rate": 1e-3,
        "epochs": 100,
        "dropout_rate": 0.5,
        "weight_decay": 1e-2,
        # (l2_reg_strength/hidden_units removidos: nenhum caminho os
        # consumia para esta arquitetura — config morto.)
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "recommended_epochs": 100,
        "notes": "Res2Net-50 com AdamW (weight decay real) + dropout 0.5 e checkpoint com restauração guardada (2026-07-14).",
    },
}


def _compact(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _canonical_arch_key(arch: str) -> str:
    compact = _compact(arch)
    return ARCH_ALIASES.get(compact, compact)


def _base_recommended_hparams(arch: str) -> Dict[str, Any]:
    compact = _canonical_arch_key(arch)
    if compact in CLASSICAL_ARCHES:
        return {
            "model_family": "classical",
            "fit": "sklearn",
            "feature_scaling": True,
            "training_budget": "GridSearchCV + final refit",
            "epochs": None,
        }

    if compact not in NEURAL_BENCHMARK_HPARAMS:
        raise ValueError(
            f"Arquitetura sem hiperparâmetros de plano: {arch}. "
            "O escopo OFICIAL cobre RawNet2, AASIST, RawGAT-ST, Conformer, "
            "CCT/Hybrid CNN-Transformer, AST/SpectrogramTransformer, "
            "Res2Net/MultiscaleCNN, SVM e RandomForest (WavLM/HuBERT Original "
            "rodam pelo runner PyTorch dedicado). Sonic Sleuth, "
            "EfficientNet-LSTM e Ensemble pertencem ao escopo ESTENDIDO: use "
            "`--experiment-scope extended` (que já desliga "
            "optimize_hyperparameters) ou `--no-optimize-hparams`."
        )

    params = dict(NEURAL_BENCHMARK_HPARAMS[compact])
    return params


def _fit_to_device(
    params: Dict[str, Any], arch: str, device: Dict[str, Any]
) -> Dict[str, Any]:
    tuned = dict(params)
    compact = _canonical_arch_key(arch)
    if compact in CLASSICAL_ARCHES:
        return tuned

    # Os caps abaixo foram calibrados com a janela antiga de 64.600 amostras
    # (~4,04 s). A janela canonica passou a 48.000 (3 s), 26% menor, entao eles
    # seguem SEGUROS — uma entrada menor so consome menos memoria. Conservadores
    # de proposito: subi-los e otimizacao, e otimizacao sem medir na GPU alvo
    # troca tempo de execucao por risco de OOM no meio de um run de horas.
    batch = int(tuned.get("batch_size", 32))
    if device.get("resolved_profile") == "cpu":
        if compact in {"rawnet2", "aasist", "rawgatst"}:
            cap = 4
        elif compact in HEAVY_ARCHES:
            cap = 8
        else:
            cap = 16
        tuned["batch_size"] = min(batch, cap)
        tuned["device_adjustment"] = "cpu_batch_cap"
        tuned["use_mixed_precision"] = False
    else:
        if compact == "rawnet2":
            cap = 16
        elif compact == "aasist":
            cap = 24
        elif compact == "rawgatst":
            cap = 16
        elif compact == "spectrogramtransformer":
            cap = 16
        else:
            cap = 32
        tuned["batch_size"] = min(batch, cap)
        tuned["device_adjustment"] = "gpu_vram_safe_cap"
        if compact in _MIXED_PRECISION_UNSAFE_ARCHITECTURES:
            tuned["use_mixed_precision"] = False
        else:
            tuned.setdefault("use_mixed_precision", True)

    return tuned


def _device_snapshot(profile: str) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "requested_profile": profile,
        "resolved_profile": "cpu",
        "platform": f"{platform.system()} {platform.release()}",
        "gpu_available": False,
        "gpu_names": [],
    }
    if profile == "cpu":
        snap["resolved_profile"] = "cpu"
        return snap

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        snap["gpu_available"] = bool(gpus)
        snap["gpu_names"] = [getattr(gpu, "name", str(gpu)) for gpu in gpus]
        if profile == "gpu":
            snap["resolved_profile"] = "gpu" if gpus else "cpu"
            if not gpus:
                snap["gpu_request_unavailable"] = True
        elif profile == "auto":
            snap["resolved_profile"] = "gpu" if gpus else "cpu"
    except Exception as exc:
        snap["tensorflow_probe_error"] = str(exc)
    return snap


def _merge_effective_hparams(
    cfg: BenchmarkConfig,
    arch: str,
    device: Dict[str, Any],
) -> Dict[str, Any]:
    compact = _compact(arch)
    if cfg.optimize_hyperparameters:
        params = _base_recommended_hparams(arch)
    elif compact in CLASSICAL_ARCHES:
        params = {
            "model_family": "classical",
            "fit": "sklearn",
            "feature_scaling": True,
            "training_budget": "single fit",
            "epochs": None,
        }
    else:
        params = {"epochs": cfg.epochs, "batch_size": cfg.batch_size}

    params = _fit_to_device(params, arch, device)
    if compact not in CLASSICAL_ARCHES:
        params["epochs"] = int(cfg.epochs)
        params.setdefault("batch_size", int(cfg.batch_size))
        params.setdefault("learning_rate", 1e-3)
        params.setdefault("early_stopping", True)
        params.setdefault("lr_scheduler", "architecture_default")
        params["epochs_source"] = "benchmark_cli"

    overrides = cfg.training_overrides.get(arch, {})
    params.update(overrides)
    if compact not in CLASSICAL_ARCHES:
        # Controles do protocolo sobrescrevem apenas aspectos de comparabilidade.
        params["epochs"] = int(cfg.epochs)
        params["epochs_source"] = "standardized_benchmark_budget"
        # `early_stopping` derivado do orçamento fixo, MAS um override explícito
        # do chamador vence: a linha incondicional anterior descartava em
        # silêncio o `--no-early-stopping` sempre que fixed_epoch_budget=False.
        if "early_stopping" not in overrides:
            params["early_stopping"] = not bool(cfg.fixed_epoch_budget)
        params["select_best_checkpoint"] = bool(cfg.select_best_checkpoint)
        params["validation_condition"] = "clean"
        params["calibrate_under_noise"] = False
        params["decision_threshold"] = float(cfg.decision_threshold)
    return params


def effective_hyperparameters(arch: str) -> Dict[str, Any]:
    """Hiperparâmetros do PIPELINE para uma arquitetura, sem levantar erro.

    Existe para que consumidores fora do benchmark — a interface Gradio, em
    primeiro lugar — mostrem a MESMA configuração que o benchmark treina, em
    vez de literais próprios.

    O CLAUDE.md já documenta que os hiperparâmetros vivem em três lugares
    (`registry.default_params`, o `create_model` de cada arquitetura e este
    módulo) e alerta para o drift. A interface era um QUARTO: `load_defaults`
    caía em `batch_size=32, epochs=10, learning_rate=0.001` para toda
    arquitetura — AASIST, Conformer e Sonic Sleuth apareciam idênticas, e
    nenhuma batia com o que o benchmark usa (AASIST: lote 24, LR 3e-4).

    Precedência:
      1. `NEURAL_BENCHMARK_HPARAMS` — a configuração que o benchmark treina;
      2. `registry.default_params` — para as arquiteturas do escopo estendido
         (Sonic Sleuth, EfficientNet-LSTM, Ensemble, WavLM, HuBERT), que não
         têm entrada de plano, e para as chaves que o plano não carrega.

    Devolve `{}` para arquiteturas desconhecidas — o chamador decide o
    fallback. Nunca levanta: uma interface não pode quebrar porque alguém
    selecionou um modelo fora do recorte oficial.
    """
    compact = _canonical_arch_key(arch)
    resolved: Dict[str, Any] = {}

    try:
        from app.domain.models.architectures.registry import architecture_registry

        base = (
            architecture_registry.get_architecture_by_any_name(arch).default_params
            or {}
        )
        resolved.update({k: v for k, v in base.items() if v is not None})
    except Exception:  # noqa: BLE001 — registry ausente não impede o plano
        pass

    if compact in NEURAL_BENCHMARK_HPARAMS:
        plano = NEURAL_BENCHMARK_HPARAMS[compact]
        resolved.update({k: v for k, v in plano.items() if v is not None})
    elif compact in CLASSICAL_ARCHES:
        resolved.update({"model_family": "classical", "epochs": None})

    return resolved


def build_benchmark_plan(
    cfg: BenchmarkConfig, data: Any | None = None
) -> Dict[str, Any]:
    """Cria o plano de execução antes do treino."""
    device = _device_snapshot(cfg.device_profile)
    dataset = {}
    if data is not None:
        y = np.asarray(data.y)
        dataset = {
            "name": getattr(data, "name", None),
            "n_total": int(len(y)),
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "balance": {
                "real": int((y == 0).sum()),
                "fake": int((y == 1).sum()),
            },
            "metadata": getattr(data, "metadata", {}) or {},
        }

    architectures = {}
    for arch in cfg.architectures:
        compact = _compact(arch)
        architectures[arch] = {
            "type": "classical" if compact in CLASSICAL_ARCHES else "neural",
            "training_config": _merge_effective_hparams(cfg, arch, device),
        }

    return {
        "preset": cfg.preset_name,
        "device": device,
        "dataset": dataset,
        "benchmark": {
            "architectures": list(cfg.architectures),
            "snr_levels_db": list(cfg.snr_levels_db),
            "latency_runs": int(cfg.latency_runs),
            "run_api_probe": bool(cfg.run_api_probe),
            "optimize_hyperparameters": bool(cfg.optimize_hyperparameters),
            "standardized_controls": {
                "epochs": int(cfg.epochs),
                "fixed_epoch_budget": bool(cfg.fixed_epoch_budget),
                "select_best_checkpoint": bool(cfg.select_best_checkpoint),
                "validation_condition": "clean",
                "decision_threshold": float(cfg.decision_threshold),
                "metric_threshold_policy": cfg.metric_threshold_policy,
                "experiment_scope": cfg.experiment_scope,
                "preserve_predefined_splits": bool(cfg.preserve_predefined_splits),
                "fail_on_split_overlap": bool(cfg.fail_on_split_overlap),
                "waveform_awgn_before_frontend": True,
            },
            "convergence": {
                "auc_roc_min": float(cfg.converge_auc_threshold),
                "accuracy_min": float(cfg.converge_accuracy_threshold),
            },
        },
        "architectures": architectures,
    }


def apply_plan_to_config(cfg: BenchmarkConfig, plan: Dict[str, Any]) -> BenchmarkConfig:
    cfg.training_overrides = {
        arch: dict(info.get("training_config") or {})
        for arch, info in (plan.get("architectures") or {}).items()
    }
    return cfg


def write_benchmark_plan(plan: Dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized = json.dumps(
        plan, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    effective = {
        "schema": "xfakesong-effective-training-config-v1",
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "plan": plan,
    }
    (out / "effective_training_config.json").write_text(
        json.dumps(effective, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Plano de Benchmark",
        "",
        f"- Preset: `{plan.get('preset')}`",
        f"- Perfil de dispositivo: `{plan.get('device', {}).get('resolved_profile')}`",
        f"- Dataset: `{plan.get('dataset', {}).get('name')}`",
        f"- Amostras: `{plan.get('dataset', {}).get('n_total')}`",
        f"- SNRs: `{plan.get('benchmark', {}).get('snr_levels_db')}`",
        f"- API probe: `{plan.get('benchmark', {}).get('run_api_probe')}`",
        "",
        "## Hiperparâmetros Efetivos",
        "",
        "| Arquitetura | Tipo | Treino | Batch | LR | Ajuste |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arch, info in (plan.get("architectures") or {}).items():
        hp = info.get("training_config") or {}
        lines.append(
            f"| {arch} | {info.get('type')} | "
            f"{hp.get('training_budget') or hp.get('epochs', '-')} | "
            f"{hp.get('batch_size', '-')} | {hp.get('learning_rate', '-')} | "
            f"{hp.get('device_adjustment', '-')} |"
        )
    (out / "benchmark_plan.md").write_text("\n".join(lines), encoding="utf-8")
