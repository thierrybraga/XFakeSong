"""Cabeça classificadora compartilhada dos modelos SSL originais (WavLM/HuBERT).

Fonte ÚNICA da topologia da cabeça usada (1) pelo runner de treino
(`scripts/benchmark/run_wavlm_original_benchmark.py`) e (2) pelo wrapper de
inferência (`TorchSSLOriginalModel` em model_loader.py) — evita drift de
paridade treino↔inferência quando o contrato de embedding muda.

Contrato de embedding (gravado no checkpoint `.pt` como `embedding_config`):
    target_samples: janela da forma de onda (amostras @16 kHz)
    layer_pooling:  'weighted' (soma ponderada aprendida via softmax sobre
                    TODAS as hidden_states do backbone — protocolo padrão de
                    avaliação downstream de modelos SSL: Yang et al., "SUPERB:
                    Speech processing Universal PERformance Benchmark",
                    Interspeech 2021, arXiv:2105.01051. As camadas
                    intermediárias carregam mais artefatos de síntese que a
                    última) ou 'last' (comportamento legado: só a última
                    camada)
    time_pooling:   'meanstd' (média ⊕ desvio por dimensão) ou 'mean' (legado)
    num_layers:     nº de hidden_states agregadas (13 p/ *-base: 12+embedding)
    feature_dim:    dimensão por camada após o time pooling
                    (hidden_size × 2 no 'meanstd')

Checkpoints antigos não trazem `embedding_config` — o consumidor deve cair no
legado: {16000, 'last', 'mean', 1, hidden_size}.

Torch é importado tardiamente: este módulo pode ser importado em ambientes
sem PyTorch (ex.: caminho Keras puro) sem custo.
"""

from __future__ import annotations

from typing import Any, Dict

LEGACY_EMBEDDING_CONFIG: Dict[str, Any] = {
    "target_samples": 16000,
    "layer_pooling": "last",
    "time_pooling": "mean",
    "num_layers": 1,
    "feature_dim": None,  # legado: hidden_size do backbone
}

_CLS_CACHE: Dict[str, Any] = {}


def _weighted_head_cls():
    """Define (uma vez) a classe nn.Module da cabeça com soma ponderada."""
    if "cls" in _CLS_CACHE:
        return _CLS_CACHE["cls"]

    import torch
    import torch.nn as nn

    class SSLWeightedLayerHead(nn.Module):
        """Soma ponderada (softmax) sobre camadas + MLP de classificação.

        Entrada: (B, L, F) — L camadas time-pooled — ou (B, F) (degrada para
        o MLP puro, útil em smoke tests).
        """

        def __init__(self, num_layers: int, feature_dim: int,
                     dropout: float = 0.2, hidden: int = 256):
            super().__init__()
            self.layer_logits = nn.Parameter(torch.zeros(int(num_layers)))
            self.mlp = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(int(feature_dim), int(hidden)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), 2),
            )

        def forward(self, x):  # noqa: D102 - contrato do nn.Module
            if x.dim() == 3:
                weights = torch.softmax(self.layer_logits, dim=0)
                x = torch.einsum("l,blf->bf", weights, x)
            return self.mlp(x)

    _CLS_CACHE["cls"] = SSLWeightedLayerHead
    return SSLWeightedLayerHead


def resolve_embedding_config(
    checkpoint: Dict[str, Any], backbone_hidden_size: int
) -> Dict[str, Any]:
    """Extrai o contrato de embedding de um checkpoint, com fallback legado."""
    cfg = dict(LEGACY_EMBEDDING_CONFIG)
    cfg.update(dict(checkpoint.get("embedding_config") or {}))
    if not cfg.get("feature_dim"):
        cfg["feature_dim"] = int(backbone_hidden_size)
    cfg["target_samples"] = int(cfg["target_samples"])
    cfg["num_layers"] = int(cfg["num_layers"])
    cfg["feature_dim"] = int(cfg["feature_dim"])
    return cfg


def build_ssl_classifier(embedding_config: Dict[str, Any], dropout: float = 0.2):
    """Constrói a cabeça correspondente ao contrato de embedding."""
    import torch.nn as nn

    feature_dim = int(embedding_config["feature_dim"])
    if str(embedding_config.get("layer_pooling", "last")) == "weighted":
        head_cls = _weighted_head_cls()
        return head_cls(
            num_layers=int(embedding_config["num_layers"]),
            feature_dim=feature_dim,
            dropout=float(dropout),
        )
    return nn.Sequential(
        nn.Dropout(float(dropout)),
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(256, 2),
    )


def pool_hidden_states(outputs, embedding_config: Dict[str, Any]):
    """Aplica layer/time pooling à saída do backbone → tensor de features.

    Retorna (B, L, F) para 'weighted' ou (B, F) para 'last'. Executar sob
    torch.no_grad() quando o backbone estiver congelado.
    """
    import torch

    time_pooling = str(embedding_config.get("time_pooling", "mean"))

    def _pool_time(h):
        mean = h.mean(dim=1)
        if time_pooling == "meanstd":
            return torch.cat([mean, h.std(dim=1, unbiased=False)], dim=-1)
        return mean

    if str(embedding_config.get("layer_pooling", "last")) == "weighted":
        layers = outputs.hidden_states  # tupla (num_layers+1) de (B, T, H)
        return torch.stack([_pool_time(h) for h in layers], dim=1)
    return _pool_time(outputs.last_hidden_state)
