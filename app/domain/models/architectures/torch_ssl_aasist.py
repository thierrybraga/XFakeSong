"""Back-end de grafo AASIST em PyTorch, sobre a sequência de um front-end SSL.

MOTIVAÇÃO (2026-08-09). O runner SSL oficial
(`scripts/benchmark/run_wavlm_original_benchmark.py`) usava a receita de
*probing* do SUPERB: backbone congelado, pooling global (média⊕desvio sobre
~150 frames) e um MLP de duas camadas. Isso mede quanta informação existe na
representação congelada — não é a receita que produz números de campeonato.

A literatura de anti-spoofing com front-end SSL converge em dois pontos que
aquela montagem não atende:

1. **o front-end é ajustado**, não congelado (Tak et al., "Automatic speaker
   verification spoofing and deepfake detection using wav2vec 2.0 and data
   augmentation", Odyssey 2022; Wang & Yamagishi, "Investigating
   self-supervised front ends for speech spoofing countermeasures", Odyssey
   2022 — nos dois, destravar o front-end é o fator isolado mais decisivo);
2. **a sequência temporal chega ao back-end**. Artefato de síntese é local
   (transiente, descontinuidade de fase); colapsar o enunciado inteiro num
   vetor antes de qualquer modelagem discriminativa joga fora exatamente o
   sinal procurado.

Este módulo cobre o ponto 2 no caminho PyTorch. É um porte FIEL do
`app/domain/models/architectures/ssl_utils.py::build_ssl_aasist_backend`
(Keras), que já implementa a topologia e é usado pelas variantes
`wavlm_aasist`/`hubert_aasist` do escopo estendido. Portar em vez de
reinventar mantém as duas famílias comparáveis: mesma topologia, mesmas
temperaturas, mesmo readout.

Referência da topologia: Jung et al., "AASIST: Audio Anti-Spoofing using
Integrated Spectro-Temporal Graph Attention Networks", ICASSP 2022.

Torch é importado no topo — ao contrário de `ssl_head.py`, este módulo só é
usado pelo runner PyTorch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn_over_nodes(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """BatchNorm por CANAL num tensor (B, N, C).

    O `layers.BatchNormalization` do Keras normaliza o último eixo; o
    `nn.BatchNorm1d` do torch espera (B, C, L). A transposição existe só para
    igualar a estatística — é por canal, agregando batch e nós nos dois casos.
    """
    return bn(x.transpose(1, 2)).transpose(1, 2)


class SSLGraphAttention(nn.Module):
    """GAT fiel a RawGAT-ST/AASIST — produto par-a-par, tanh e temperatura.

    Não é o GAT aditivo de Velickovic. O mapa de atenção sai do produto
    elemento a elemento entre pares de nós::

        A_ij = softmax_i( w^T · tanh(W_att (h_i ⊙ h_j)) / τ )
        h'   = SELU( BN( W_com (A · h) + W_sem h ) )

    O softmax normaliza sobre o eixo -2 e a agregação soma o último — como no
    código de referência dos autores e no porte Keras deste repositório.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temperature: float = 1.0,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        self.att_proj = nn.Linear(int(in_features), self.out_features)
        self.att_weight = nn.Parameter(torch.empty(self.out_features, 1))
        nn.init.xavier_uniform_(self.att_weight)
        self.proj_with_att = nn.Linear(int(in_features), self.out_features)
        self.proj_without_att = nn.Linear(int(in_features), self.out_features)
        self.bn = nn.BatchNorm1d(self.out_features)
        self.input_drop = nn.Dropout(float(dropout_rate))

    def _att_map(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, 1, C) * (B, 1, N, C) -> (B, N, N, C)
        pairwise = x.unsqueeze(2) * x.unsqueeze(1)
        att = torch.tanh(self.att_proj(pairwise))  # (B, N, N, out)
        att = att @ self.att_weight  # (B, N, N, 1)
        att = att / self.temperature
        return torch.softmax(att, dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        att_map = self._att_map(x).squeeze(-1)  # (B, N, N)
        out = self.proj_with_att(att_map @ x) + self.proj_without_att(x)
        return F.selu(_bn_over_nodes(self.bn, out))


class SSLHtrgGraphAttention(nn.Module):
    """HS-GAL heterogênea do AASIST (§2.3): três parâmetros por tipo de aresta.

    O que dá nome à camada são conjuntos DISTINTOS de parâmetros de atenção por
    tipo de aresta — tipo1↔tipo1 (`w11`), tipo2↔tipo2 (`w22`) e o par cruzado
    (`w12`, compartilhado nas duas direções). Atenção homogênea com embedding
    de tipo não reproduz isso.

    O master node agrega o grafo inteiro por atenção própria e é devolvido para
    alimentar a HS-GAL seguinte. Sem master de entrada, inicializa como a média
    dos nós — como no código de referência.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 32,
        temperature: float = 100.0,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        in_features = int(in_features)

        self.proj_type1 = nn.Linear(in_features, in_features)
        self.proj_type2 = nn.Linear(in_features, in_features)
        self.att_proj = nn.Linear(in_features, self.out_features)
        self.att_projM = nn.Linear(in_features, self.out_features)

        def _att_vec() -> nn.Parameter:
            p = nn.Parameter(torch.empty(self.out_features, 1))
            nn.init.xavier_uniform_(p)
            return p

        self.att_weight11 = _att_vec()
        self.att_weight22 = _att_vec()
        self.att_weight12 = _att_vec()
        self.att_weightM = _att_vec()

        self.proj_with_att = nn.Linear(in_features, self.out_features)
        self.proj_without_att = nn.Linear(in_features, self.out_features)
        self.proj_with_attM = nn.Linear(in_features, self.out_features)
        self.proj_without_attM = nn.Linear(in_features, self.out_features)
        self.bn = nn.BatchNorm1d(self.out_features)
        self.input_drop = nn.Dropout(float(dropout_rate))

    def _att_map(self, x: torch.Tensor, n1: int) -> torch.Tensor:
        pairwise = x.unsqueeze(2) * x.unsqueeze(1)  # (B, N, N, C)
        att = torch.tanh(self.att_proj(pairwise))  # (B, N, N, out)

        a11 = att[:, :n1, :n1, :] @ self.att_weight11
        a12 = att[:, :n1, n1:, :] @ self.att_weight12
        a21 = att[:, n1:, :n1, :] @ self.att_weight12
        a22 = att[:, n1:, n1:, :] @ self.att_weight22

        top = torch.cat([a11, a12], dim=2)  # (B, n1, N, 1)
        bottom = torch.cat([a21, a22], dim=2)  # (B, n2, N, 1)
        att_map = torch.cat([top, bottom], dim=1) / self.temperature
        return torch.softmax(att_map, dim=-2)

    def _update_master(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        att = torch.tanh(self.att_projM(x * master))  # (B, N, out)
        att = att @ self.att_weightM  # (B, N, 1)
        att = torch.softmax(att / self.temperature, dim=-2)
        pooled = att.transpose(1, 2) @ x  # (B, 1, C)
        return self.proj_with_attM(pooled) + self.proj_without_attM(master)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        master: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n1 = int(x1.shape[1])
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = x.mean(dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._att_map(x, n1).squeeze(-1)  # (B, N, N)
        master = self._update_master(x, master)

        out = self.proj_with_att(att_map @ x) + self.proj_without_att(x)
        out = F.selu(_bn_over_nodes(self.bn, out))
        return out[:, :n1, :], out[:, n1:, :], master


class GraphPool(nn.Module):
    """Top-k de nós com gate sigmoide (Graph U-Nets, Gao & Ji 2019)."""

    def __init__(self, in_features: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = float(ratio)
        self.score_proj = nn.Parameter(torch.empty(int(in_features), 1))
        nn.init.xavier_uniform_(self.score_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_nodes = x.shape[1]
        k = max(int(n_nodes * self.ratio), 1)
        scores = (x @ self.score_proj).squeeze(-1)  # (B, N)
        _, idx = torch.topk(scores, k=k, dim=1, sorted=False)
        gathered = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        gate = torch.sigmoid(torch.gather(scores, 1, idx)).unsqueeze(-1)
        return gathered * gate


class GraphReadout(nn.Module):
    """Readout do AASIST: máximo ⊕ média ponderada por atenção."""

    def __init__(self, in_features: int):
        super().__init__()
        self.att_w = nn.Parameter(torch.empty(int(in_features), 1))
        nn.init.xavier_uniform_(self.att_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_max = x.max(dim=1).values
        alpha = torch.softmax(x @ self.att_w, dim=1)  # (B, N, 1)
        h_att = (x * alpha).sum(dim=1)
        return torch.cat([h_max, h_att], dim=-1)


class SSLAASISTBackend(nn.Module):
    """Sequência SSL (B, T, H) -> logits (B, 2), via grafo espectro-temporal.

    Espelha `ssl_utils.build_ssl_aasist_backend`: projeção 1x1 para `proj_dim`,
    LayerNorm, reamostragem temporal para `t_fixed`, dois GATs (espectral com
    os canais como nós, temporal com os frames como nós), uma HS-GAL, pooling
    top-k em cada ramo e readout concatenado.

    A reamostragem temporal FIXA existe porque o ramo espectral usa o tempo
    como dimensão de features dos nós, e o peso do GAT exige feature-dim
    estática — a sequência SSL tem T variável com o comprimento do clipe.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 128,
        t_fixed: int = 64,
        graph_dim: int = 32,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.t_fixed = int(t_fixed)
        self.proj = nn.Conv1d(int(input_dim), int(proj_dim), kernel_size=1)
        self.proj_ln = nn.LayerNorm(int(proj_dim))

        # Espectral: canais viram nós (N=proj_dim, features=t_fixed).
        self.gat_spec = SSLGraphAttention(
            self.t_fixed, graph_dim, temperature=2.0, dropout_rate=dropout_rate
        )
        # Temporal: frames viram nós (N=t_fixed, features=proj_dim).
        self.gat_temp = SSLGraphAttention(
            int(proj_dim), graph_dim, temperature=2.0, dropout_rate=dropout_rate
        )
        self.hsgal = SSLHtrgGraphAttention(
            graph_dim, graph_dim, temperature=100.0, dropout_rate=dropout_rate
        )
        self.pool_spec = GraphPool(graph_dim, ratio=0.5)
        self.pool_temp = GraphPool(graph_dim, ratio=0.5)
        self.readout_spec = GraphReadout(graph_dim)
        self.readout_temp = GraphReadout(graph_dim)

        # DIVERGÊNCIA DELIBERADA do porte Keras (`ssl_utils.py`), que descarta
        # o master node da HS-GAL: `spectral, temporal, _master = ...`. No
        # AASIST original o master ENTRA no readout —
        # `cat([T_max, T_avg, S_max, S_avg, master])` — e é ele que carrega a
        # visão global do grafo. Descartado, os quatro tensores do ramo do
        # master (att_projM, att_weightM, proj_with_attM, proj_without_attM)
        # ficam sem gradiente: parâmetros calculados a cada passo e jogados
        # fora. Um teste trava que todo parâmetro do back-end recebe gradiente.
        self.head = nn.Sequential(
            nn.LayerNorm(5 * graph_dim),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(5 * graph_dim, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, H) -> (B, T, proj_dim)
        h = self.proj(x.transpose(1, 2)).transpose(1, 2)
        h = self.proj_ln(h)
        # Reamostragem linear do eixo temporal (equivale ao tf.image.resize
        # bilinear com half_pixel_centers do porte Keras).
        h = F.interpolate(
            h.transpose(1, 2),
            size=self.t_fixed,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        spectral = self.gat_spec(h.transpose(1, 2))  # (B, proj_dim, g)
        temporal = self.gat_temp(h)  # (B, t_fixed, g)
        spectral, temporal, master = self.hsgal(spectral, temporal)

        s = self.readout_spec(self.pool_spec(spectral))
        t = self.readout_temp(self.pool_temp(temporal))
        return self.head(torch.cat([s, t, master.squeeze(1)], dim=-1))


class WeightedLayerSum(nn.Module):
    """Soma ponderada (softmax) das hidden-states, PRESERVANDO o tempo.

    A receita SUPERB agrega as camadas depois do pooling temporal, porque a
    tarefa downstream ali recebe um vetor. Aqui a agregação acontece ANTES —
    sobre as sequências — para que o back-end de grafo receba (B, T, H). É a
    mesma combinação convexa de camadas; o que muda é não descartar o tempo.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(int(num_layers)))

    def forward(self, hidden_states) -> torch.Tensor:
        stacked = torch.stack(list(hidden_states), dim=0)  # (L, B, T, H)
        weights = torch.softmax(self.layer_logits, dim=0)
        return torch.einsum("l,lbth->bth", weights, stacked)


class SSLAASISTModel(nn.Module):
    """Front-end SSL + soma ponderada de camadas + back-end de grafo AASIST.

    O backbone entra pronto (HuggingFace `WavLMModel`/`HubertModel`). Quem
    decide o que é treinável é `configure_finetuning`, não esta classe.
    """

    def __init__(
        self,
        backbone,
        num_layers: int,
        hidden_size: int,
        proj_dim: int = 128,
        t_fixed: int = 64,
        graph_dim: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.layer_sum = WeightedLayerSum(num_layers)
        self.backend = SSLAASISTBackend(
            input_dim=hidden_size,
            proj_dim=proj_dim,
            t_fixed=t_fixed,
            graph_dim=graph_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        out = self.backbone(waveform, output_hidden_states=True)
        return self.backend(self.layer_sum(out.hidden_states))


def configure_finetuning(
    backbone, freeze_backbone: bool, freeze_feature_encoder: bool = True
) -> dict:
    """Define o que treina no backbone e DEVOLVE o que realmente foi feito.

    Até 2026-08-09 o runner ligava `requires_grad` e `backbone.train()` para
    `--no-freeze-backbone`, mas os embeddings eram calculados sob `no_grad()` e
    o otimizador só recebia a cabeça: o artefato saía com
    `backbone_trainable: true` num run integralmente congelado. Esta função
    existe para que a declaração venha da CONTAGEM de parâmetros treináveis, e
    não de uma flag.

    O extrator convolucional fica congelado mesmo no fine-tuning — prática
    padrão dos modelos wav2vec2/WavLM/HuBERT (`freeze_feature_encoder` da
    HuggingFace) e o que Tak et al. (2022) fazem: aquelas camadas aprendem
    filtros de baixo nível já estáveis e destravá-las desestabiliza o treino.
    """
    for param in backbone.parameters():
        param.requires_grad = not freeze_backbone

    frozen_feature_encoder = False
    if not freeze_backbone and freeze_feature_encoder:
        fn = getattr(backbone, "freeze_feature_encoder", None)
        if callable(fn):
            fn()
            frozen_feature_encoder = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    backbone.train(not freeze_backbone)
    return {
        "requested_freeze_backbone": bool(freeze_backbone),
        "feature_encoder_frozen": frozen_feature_encoder,
        "backbone_trainable_params": int(trainable),
        "backbone_total_params": int(total),
        # É ISTO que o artefato deve declarar: derivado da contagem real.
        "backbone_trainable": bool(trainable > 0),
    }
