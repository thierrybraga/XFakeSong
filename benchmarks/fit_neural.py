"""Treino das arquiteturas neurais do benchmark, via ``TrainingService``.

Traduz o plano em ``train_config`` (o que é da arquitetura vai para
``parameters``), treina e devolve os callables de inferência e o bloco de
proveniência do ajuste. Saiu de ``runner.py`` (P2-15).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import (
    _architecture_dir,
    _compact_slug,
    _preserve_run_artifact,
    _slug,
)
from benchmarks.config import BenchmarkConfig
from benchmarks.efficiency import count_params, file_size_mb
from benchmarks.protocol import _stamp_benchmark_frontend

logger = logging.getLogger("benchmark")


def _run_neural(
    arch: str,
    cfg: BenchmarkConfig,
    splits,
    tmp: Path,
    models_dir: Path,
    training_seed: int | None = None,
):
    """Treina via TrainingService e devolve callables de inferência."""
    from app.core.contracts.base import ProcessingStatus
    from app.domain.services.training_service import TrainingService

    Xtr, ytr, Xv, yv, _Xte, _yte = splits[:6]
    protocol = splits[7] if len(splits) > 7 else {}
    name = f"bench_{_slug(arch)}"
    npz = tmp / "ds.npz"
    np.savez(npz, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)

    models_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = _architecture_dir(cfg, arch, training_seed)
    svc = TrainingService(models_dir=str(models_dir))
    train_config = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "model_name": name,
        "verbose": 0,
        "progress_log_interval": 1,
        "progress_label": arch,
    }
    train_config.update(cfg.training_overrides.get(arch, {}))
    # O orçamento é um controle experimental, não um hiperparâmetro do modelo.
    train_config["epochs"] = int(cfg.epochs)
    train_config["model_name"] = name
    train_config["verbose"] = 0
    compact = _compact_slug(arch)
    if compact == "rawnet2":
        # Mantém os hiperparâmetros do retreino anterior. No protocolo comum,
        # o aumento interno será desligado após esta seleção porque a cópia
        # ruidosa já foi gerada no domínio da forma de onda.
        train_config.update(
            {
                "learning_rate": 1e-4,
                "use_augmentation": True,
                "use_mixed_precision": False,
            }
        )
        # TOPOLOGIA: baseline ANTI-SPOOFING (Tak et al., 2021), não o
        # "Improved RawNet" de VERIFICAÇÃO DE LOCUTOR (Jung et al., 2020).
        #
        # São arquiteturas diferentes com o mesmo nome. O escopo oficial roda
        # RawNet2 e AASIST como baselines da Track 1 do ASVspoof, e é o de Tak
        # que a literatura de anti-spoofing compara por EER — Sinc 20, blocos
        # [20,20,128,128,128,128], 3×GRU(1024). O `registry.default_params`
        # descreve a variante de locutor, então sem esta promoção o benchmark
        # construía ~7,0M parâmetros da arquitetura errada.
        #
        # Os valores vêm de `NEURAL_BENCHMARK_HPARAMS["rawnet2"]` (via
        # `train_config`), NÃO de um import da arquitetura: a interface Gradio
        # resolve seus defaults por `planning.effective_hyperparameters()`, que
        # lê o mesmo dicionário. Uma topologia que vivesse só aqui faria a
        # interface treinar uma arquitetura e o benchmark outra.
        #
        # ESTE BLOCO PRECISA VIVER NO PRIMEIRO RAMO. Ele nasceu (2026-08-20)
        # como um `elif compact == "rawnet2"` mais abaixo na mesma cadeia — e
        # portanto INALCANÇÁVEL, porque este `if` já casa. O efeito era mudo: o
        # benchmark seguia construindo a variante de locutor enquanto manifesto,
        # plano e tabela de custo declaravam a anti-spoofing.
        model_params = train_config.setdefault("parameters", {})
        for key in (
            "sinc_filters",
            "sinc_kernel_size",
            "res_filters",
            "gru_units",
            "gru_layers",
            "dense_units",
        ):
            if key in train_config:
                model_params[key] = train_config[key]
    elif compact in {"aasist", "rawgatst"}:
        # Compile-respect: LR e CosineDecay pertencem ao construtor.
        model_params = train_config.setdefault("parameters", {})
        for key in (
            "learning_rate",
            "min_learning_rate",
            "decay_steps",
            "dropout_rate",
            "l2_reg_strength",
            "classifier_head",
            # Só o RawGAT-ST declara `global_clipnorm` no plano; o AASIST
            # divide este ramo mas não o expõe no construtor, e a promoção é
            # condicionada a `if key in train_config`, então não o alcança.
            "global_clipnorm",
        ):
            if key in train_config:
                model_params.setdefault(key, train_config[key])
        train_config.pop("learning_rate", None)
        train_config.update(
            {
                "use_augmentation": True,
                "reduce_lr_on_plateau": False,
            }
        )
    elif compact == "efficientnetlstm":
        model_params = train_config.setdefault("parameters", {})
        if "dropout_rate" in train_config:
            model_params.setdefault("dropout_rate", float(train_config["dropout_rate"]))
        if "lstm_units" in train_config:
            model_params.setdefault("lstm_units", int(train_config["lstm_units"]))
        elif "hidden_units" in train_config:
            first_units = str(train_config["hidden_units"]).split("/", 1)[0]
            try:
                model_params.setdefault("lstm_units", int(first_units))
            except ValueError:
                pass
        if "pretrained" in train_config:
            model_params.setdefault("pretrained", bool(train_config["pretrained"]))
        train_config.update({"learning_rate": 1e-4})
    elif compact in {"hybridcnntransformer", "conformer", "spectrogramtransformer"}:
        train_config.update({"reduce_lr_on_plateau": False})
        # Compile-respect: LR/schedule vão ao CONSTRUTOR da arquitetura
        # (WarmupCosineDecay próprio), não ao TrainingConfig — inclusive o
        # CCT (2026-07-14; antes o compile do CCT era hardcoded e o
        # learning_rate/decay_steps do plano nunca chegavam ao modelo).
        model_params = train_config.setdefault("parameters", {})
        for key in (
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "decay_steps",
            "alpha",
            "dropout_rate",
        ):
            if key in train_config:
                model_params.setdefault(key, train_config[key])
        train_config.pop("learning_rate", None)
        if compact in {"conformer", "hybridcnntransformer"}:
            for key in ("clipnorm", "label_smoothing"):
                if key in train_config:
                    model_params.setdefault(key, train_config[key])
        elif compact == "spectrogramtransformer":
            # `pretrained` é parâmetro do CONSTRUTOR do AST (transferência dos
            # pesos AudioSet) — sem promovê-lo para `parameters` a flag do
            # plano não chegaria ao modelo.
            if "pretrained" in train_config:
                model_params.setdefault("pretrained", bool(train_config["pretrained"]))
            # P1 — retreino obrigatório. Liga augmentation (ruído SNR + SpecAug)
            # e força a restauração do MELHOR checkpoint (val_loss, agora
            # GUARDADA — validada no val antes de aceitar), atacando o colapso
            # val→teste por sobreajuste. Hiperparâmetros vêm do plano/defaults
            # (2026-07-14: pre-LN, lr de pico 1e-5, weight_decay 1e-5).
            train_config.update({"use_augmentation": True, "checkpoint_best": True})

    # Controles experimentais comuns; não alteram LR, dropout, regularização,
    # otimizador, scheduler ou batch customizados por arquitetura.
    train_config["calibrate_under_noise"] = False
    if cfg.fixed_epoch_budget:
        train_config["early_stopping"] = False
    # O AWGN científico já foi aplicado à forma de onda antes do frontend.
    # Desativa o AudioAugmenter legado para impedir novo ruído no log-Mel ou
    # em outra representação. SpecAugment/RawBoost são ablações separadas.
    # Exceção (2026-07-15): AASIST/RawGAT-ST usam o AudioAugmenter DINÂMICO
    # na forma de onda no lugar da cópia estática (ver
    # _prepare_protocol_splits) — para eles use_augmentation permanece True.
    if protocol.get("training_augmentation_domain") == "waveform":
        train_config["use_augmentation"] = False
        train_config["waveform_noise_protocol"] = protocol
    elif protocol.get("training_augmentation_domain") == "waveform_dynamic_augmenter":
        train_config["use_augmentation"] = True
        train_config["waveform_noise_protocol"] = protocol

    checkpoint_path = None
    checkpoint_requested = bool(train_config.pop("checkpoint_best", False))
    if cfg.select_best_checkpoint or checkpoint_requested:
        checkpoint_dir = arch_dir / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # `.weights.h5` → ModelCheckpoint salva SÓ os pesos (2026-07-14):
        # o artefato serve apenas à restauração guardada via load_weights;
        # o modelo completo de inferência é salvo separadamente pelo
        # TrainingService (save_inference_keras).
        checkpoint_path = checkpoint_dir / "best_checkpoint.weights.h5"
        train_config["checkpoint_path"] = str(checkpoint_path)
        # Sem esta linha o `checkpoint_monitor` do BenchmarkConfig não chega ao
        # ModelTrainer: o TrainingService descarta toda chave que não seja
        # campo do TrainingConfig, e o trainer lê `self.config.checkpoint_
        # monitor`. O smoke `_smoke_eer` (2026-08-17) gravou `val_loss` no
        # best.json pedindo `val_eer` justamente por causa disso.
        train_config["checkpoint_monitor"] = str(
            getattr(cfg, "checkpoint_monitor", "") or "val_loss"
        )

    # Backup operacional separado do checkpoint de selecao cientifica.
    # Preserva pesos, otimizador e contador de epocas para retomada apos
    # reinicio do host/Docker. E removido automaticamente ao concluir fit().
    backup_dir = arch_dir / "training_backup"
    train_config["backup_dir"] = str(backup_dir)

    res = svc.train_model(
        architecture=arch,
        dataset_path=str(npz),
        config=train_config,
    )
    if res.status != ProcessingStatus.SUCCESS:
        raise RuntimeError(f"train_model falhou: {res.errors}")
    train_data = (res.data.metrics or {}) if res.data else {}
    history = train_data.get("history")
    model = (res.metadata or {}).get("model")
    if model is None:
        raise RuntimeError("TrainingService não retornou o modelo treinado em memória")

    # AJUSTE 2026-07-14 (rigor da avaliação): modelos com saída LINEAR emitem
    # LOGITS crus (ex.: AASIST/AMSoftmax, faixa ≈[-15, 15]). Antes, pred[:, 1]
    # ia direto para _finite_scores, que CLIPA em [0, 1] — quantizando os
    # scores em ~{0, 1} (observado: 4 valores distintos em 2250 predições) e
    # invalidando EER/ROC/min-tDCF, que exigem scores contínuos. Além do clip,
    # o ranking de p_fake é monotônico em (z1 − z0), não em z1 isolado.
    # Normaliza (softmax/sigmoid) ANTES de extrair p_fake.
    # Mesma funcao que alimenta `output_is_logits` no contrato e que a producao
    # consulta — um criterio so, para os dois caminhos nao divergirem.
    from app.domain.services.detection.predictor import model_emits_logits

    _from_logits = bool(model_emits_logits(model))

    def _normalize_probs(pred: np.ndarray) -> np.ndarray:
        if not _from_logits:
            return pred
        if pred.ndim > 1 and pred.shape[-1] > 1:
            z = pred - pred.max(axis=-1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=-1, keepdims=True)
        return 1.0 / (1.0 + np.exp(-pred))

    config_path = models_dir / f"{name}_config.json"
    input_contract: dict[str, Any] = {}
    if config_path.exists():
        try:
            input_contract = json.loads(config_path.read_text(encoding="utf-8")).get(
                "input_contract", {}
            )
        except (OSError, json.JSONDecodeError):
            input_contract = {}

    # CORREÇÃO 2026-07-27 — escala de score coerente com o limiar calibrado.
    #
    # O `eer_threshold` do contrato é derivado no conjunto de VALIDAÇÃO sobre
    # probabilidades COM temperature scaling (trainer._compute_eer_threshold), e
    # `auto_calibrate_temperature` é True por padrão. O benchmark aplicava esse
    # limiar a scores SEM temperatura: como o scaling é monótono, o EER não muda,
    # mas o PONTO DE OPERAÇÃO se desloca — `accuracy_at_calibrated_threshold`,
    # justamente a métrica não-oráculo, ficava medida no limiar errado.
    #
    # Aplicar T aqui também alinha o benchmark ao detector implantado (o
    # Predictor aplica a mesma T) e faz o ECE medir a calibração do sistema
    # entregue, não a da saída crua. EER/AUC/min-tDCF não mudam (transformação
    # monótona); `accuracy` no limiar fixo 0,5 pode mudar, e deve — é a decisão
    # que o sistema realmente toma.
    _temperature = input_contract.get("temperature")
    try:
        _temperature = float(_temperature) if _temperature is not None else 1.0
    except (TypeError, ValueError):
        _temperature = 1.0
    if not np.isfinite(_temperature) or _temperature <= 0:
        _temperature = 1.0

    def _apply_temperature(probs: np.ndarray) -> np.ndarray:
        if _temperature == 1.0:
            return probs
        from app.domain.services.detection.predictor import (
            apply_temperature_scaling,
        )

        return np.asarray(
            apply_temperature_scaling(probs, _temperature), dtype="float64"
        )

    def predict_p_fake(X: np.ndarray) -> np.ndarray:
        pred = model.predict(np.asarray(X, dtype="float32"), verbose=0)
        pred = _normalize_probs(np.asarray(pred, dtype="float64"))
        pred = _apply_temperature(pred)
        if pred.ndim > 1 and pred.shape[-1] > 1:
            return pred[:, 1]
        return pred.reshape(-1)

    def predict_fn(xb):  # latência: forward puro do modelo
        return model.predict(np.asarray(xb, dtype="float32"), verbose=0)

    model_path = models_dir / f"{name}.keras"
    reported_training_config = dict(train_data.get("training_config") or train_config)
    model_parameters = dict(train_config.get("parameters") or {})
    for key in (
        "epochs",
        "batch_size",
        "use_augmentation",
        "use_mixed_precision",
        "reduce_lr_on_plateau",
        "checkpoint_path",
    ):
        if key in train_config and reported_training_config.get(key) is None:
            reported_training_config[key] = train_config[key]
    if checkpoint_path is not None:
        reported_training_config["checkpoint_best"] = True
        reported_training_config["best_checkpoint_path"] = str(checkpoint_path)
        # Qual métrica elegeu a época. O TrainingService filtra as chaves que
        # não são campos de TrainingConfig, então o valor precisa ser
        # reafirmado aqui para chegar ao metrics.json — é ele que o
        # diagnóstico de estabilidade e o rodapé do TCC consultam.
        reported_training_config["checkpoint_monitor"] = str(
            getattr(cfg, "checkpoint_monitor", "") or "val_loss"
        )
    if model_parameters:
        if "learning_rate" in model_parameters:
            reported_training_config["learning_rate"] = model_parameters[
                "learning_rate"
            ]
        reported_training_config["model_parameters"] = model_parameters
    # A temperatura faz parte da definição do score avaliado — sem registrá-la,
    # os números não são reproduzíveis a partir do modelo salvo.
    reported_training_config["calibrated_temperature"] = float(_temperature)
    reported_training_config["scores_temperature_scaled"] = bool(_temperature != 1.0)

    input_contract = _stamp_benchmark_frontend(
        config_path, input_contract, protocol, arch
    )

    checkpoint_restore = dict(train_data.get("checkpoint_restore") or {})
    if checkpoint_restore.get("attempted") and not checkpoint_restore.get("restored"):
        # Não aborta: os pesos da última época ainda são um resultado, e matar
        # um treino de horas na última linha seria pior. Mas o run.log precisa
        # dizer, na altura em que alguém lê, que este artefato NÃO é o
        # checkpoint selecionado — o aviso equivalente do guard fica a dezenas
        # de milhares de linhas de distância.
        logger.warning(
            "[%s] o checkpoint eleito por %s foi DESCARTADO na restauração: o "
            "artefato avaliado tem os pesos da ÚLTIMA ÉPOCA. O resultado não "
            "corresponde ao critério de seleção declarado — retreine antes de "
            "publicar.",
            arch,
            checkpoint_restore.get("monitor", "?"),
        )

    run_artifact, artifact_fingerprint = _preserve_run_artifact(model_path, arch_dir)

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        # Necessário para RECARIMBAR o contrato depois da avaliação, quando a
        # estratégia de crop deixa de ser o placeholder "resolved_at_eval".
        #
        # SÃO DUAS CÓPIAS. `config_path` é o sidecar GLOBAL
        # (`data/models/bench_<arch>_config.json`), que a inferência carrega;
        # `run_config_path` é a cópia dentro do run, feita por
        # `_preserve_run_artifact` ANTES da avaliação — e é ela que
        # `sync_completed_benchmark_artifacts.py` promove para
        # `data/models/benchmark_final/`. Recarimbar só a global deixava o
        # artefato PROMOVIDO sem `crop_strategy`, e a inferência caía em 1 crop
        # enquanto o benchmark mediu com 3.
        "config_path": config_path,
        "run_config_path": (
            run_artifact.with_name(f"{run_artifact.stem}_config.json")
            if run_artifact is not None
            else None
        ),
        "params": count_params(model),
        "size_mb": file_size_mb(model_path),
        "history": history,
        "training_config": reported_training_config,
        "model_parameters": model_parameters,
        "final_metrics": train_data.get("final_metrics") or {},
        "input_contract": input_contract,
        "model_artifact": str(run_artifact or model_path),
        "model_artifact_shared_copy": str(model_path),
        "model_artifact_fingerprint": artifact_fingerprint,
        # Contraparte do bloco declarado no caminho clássico: o neural ajusta só
        # no treino e reserva a validação para escolher a época. Sem os dois
        # lados declarados, o `fit_samples` de 25.780 dos clássicos contra os
        # 24.324 dos neurais parecia divergência de dados, não de protocolo.
        "fit_strategy": {
            "kind": "fixed_epoch_budget_then_checkpoint_selection",
            "estimator": "keras",
            "fit_splits": ["train"],
            "fit_samples": int(len(ytr)),
            # DERIVADO do monitor real. O literal "menor val_loss limpa" que
            # ficava aqui contradizia os runs com `--checkpoint-monitor
            # val_eer`: o metrics.json declarava perda enquanto o
            # ModelCheckpoint selecionava por EER.
            "validation_role": (
                "seleção de checkpoint (menor "
                f"{str(getattr(cfg, 'checkpoint_monitor', '') or 'val_loss')}"
                " limpa)"
            ),
            "checkpoint_monitor": str(
                getattr(cfg, "checkpoint_monitor", "") or "val_loss"
            ),
            # QUAIS pesos foram avaliados. Um artefato que declara seleção por
            # val_eer mas guarda os pesos da última época não é um detalhe de
            # implementação: é o resultado sob outro protocolo.
            "checkpoint_restore": checkpoint_restore,
            "val_samples": int(len(yv)),
        },
    }
