"""Back-end AASIST em PyTorch e o fine-tuning do front-end SSL — ABLAÇÃO.

ESCOPO (revisto em 2026-08-11). Este caminho **não faz parte do escopo oficial
do benchmark**. Ele existiu por dois dias como as entradas "WavLM AASIST" e
"HuBERT AASIST", sob a premissa de que destravar o front-end era o estado da
arte. A premissa estava desatualizada: os sistemas de topo do ASVspoof 5 (2024)
usam SSL **congelado**, e o resultado de referência da receita ajustada (Tak et
al., Odyssey 2022 — 0,82% de EER no ASVspoof21 LA) usa wav2vec 2.0 XLS-R
(~300M), não WavLM/HuBERT base (94,5M). Combinar esses backbones com o grafo
seria abordagem NOVA, não benchmark de configuração documentada.

O código fica, testado, como ablação disponível. Estes testes travam duas
propriedades que ele precisa ter caso volte a ser usado:

1. o front-end é de fato AJUSTADO quando pedido — a flag que prometia isso não
   fazia nada, e o artefato declarava `backbone_trainable: true` mentindo;
2. a sequência temporal chega ao back-end em vez de virar um vetor.

O ponto 2 tem um teste que o mede diretamente
(`test_backend_e_sensivel_a_ordem_temporal`): embaralhar os frames muda a
saída. O pooling global, por construção, devolvia o mesmo vetor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.domain.models.architectures.torch_ssl_aasist import (  # noqa: E402
    GraphPool,
    GraphReadout,
    SSLAASISTBackend,
    SSLGraphAttention,
    SSLHtrgGraphAttention,
    WeightedLayerSum,
    configure_finetuning,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─── Back-end de grafo ─────────────────────────────────────────────────────


def test_backend_aceita_sequencia_e_devolve_dois_logits():
    backend = SSLAASISTBackend(input_dim=768)
    out = backend(torch.randn(3, 149, 768))

    assert out.shape == (3, 2)


@pytest.mark.parametrize("t_frames", [37, 149, 301])
def test_backend_absorve_comprimento_temporal_variavel(t_frames):
    """A reamostragem fixa existe porque o GAT espectral exige feature-dim
    estática — o T do backbone varia com o comprimento do clipe."""
    backend = SSLAASISTBackend(input_dim=768, t_fixed=64)

    assert backend(torch.randn(2, t_frames, 768)).shape == (2, 2)


def test_backend_e_sensivel_a_ordem_temporal():
    """O teste que justifica a mudança de back-end.

    Média e desvio por dimensão são INVARIANTES a permutação dos frames: o
    pooling global antigo devolvia exatamente o mesmo vetor para um enunciado e
    para o mesmo enunciado embaralhado no tempo. Artefato de síntese é local,
    então essa invariância descartava justamente o sinal procurado.
    """
    torch.manual_seed(0)
    backend = SSLAASISTBackend(input_dim=64).eval()
    x = torch.randn(1, 96, 64)
    x_embaralhado = x[:, torch.randperm(96), :]

    # Controle: o pooling global NÃO distingue os dois.
    pooled = torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=-1)
    pooled_shuf = torch.cat(
        [x_embaralhado.mean(1), x_embaralhado.std(1, unbiased=False)], dim=-1
    )
    assert torch.allclose(pooled, pooled_shuf, atol=1e-5)

    with torch.no_grad():
        assert not torch.allclose(backend(x), backend(x_embaralhado), atol=1e-4)


def test_gat_usa_temperatura_e_softmax_sobre_o_eixo_certo():
    """Temperatura alta achata a atenção em direção à média uniforme."""
    torch.manual_seed(0)
    x = torch.randn(2, 10, 16)
    frio = SSLGraphAttention(16, 8, temperature=0.5, dropout_rate=0.0).eval()
    quente = SSLGraphAttention(16, 8, temperature=1000.0, dropout_rate=0.0).eval()
    quente.load_state_dict(frio.state_dict())

    with torch.no_grad():
        a_frio = frio._att_map(x).squeeze(-1)
        a_quente = quente._att_map(x).squeeze(-1)

    # O softmax roda no eixo -2 do tensor (B, N_i, N_j, 1) — ou seja, sobre
    # N_j, que vira o último eixo depois do squeeze. É o que a agregação
    # `att_map @ x` exige: cada nó de saída i é combinação convexa dos j.
    assert torch.allclose(a_frio.sum(dim=-1), torch.ones(2, 10), atol=1e-5)
    uniforme = torch.full((2, 10, 10), 0.1)
    assert (a_quente - uniforme).abs().max() < (a_frio - uniforme).abs().max()


def test_hsgal_tem_parametros_distintos_por_tipo_de_aresta():
    """É o que dá nome à camada (AASIST §2.3); atenção homogênea não serve."""
    hsgal = SSLHtrgGraphAttention(in_features=16, out_features=8)

    assert not torch.equal(hsgal.att_weight11, hsgal.att_weight22)
    assert not torch.equal(hsgal.att_weight11, hsgal.att_weight12)
    assert not torch.equal(hsgal.att_weightM, hsgal.att_weight12)


def test_hsgal_preserva_a_contagem_de_nos_de_cada_ramo():
    hsgal = SSLHtrgGraphAttention(in_features=16, out_features=8).eval()
    x1 = torch.randn(2, 12, 16)
    x2 = torch.randn(2, 5, 16)

    with torch.no_grad():
        o1, o2, master = hsgal(x1, x2)

    assert o1.shape == (2, 12, 8)
    assert o2.shape == (2, 5, 8)
    assert master.shape == (2, 1, 8)


def test_graph_pool_reduz_pela_razao_e_aplica_gate():
    pool = GraphPool(in_features=8, ratio=0.5)
    out = pool(torch.randn(2, 10, 8))

    assert out.shape == (2, 5, 8)


def test_readout_concatena_maximo_e_atencao():
    readout = GraphReadout(in_features=8)
    out = readout(torch.randn(2, 7, 8))

    assert out.shape == (2, 16)


def test_soma_ponderada_de_camadas_preserva_o_tempo():
    """A agregação de camadas acontece ANTES do pooling, sobre sequências."""
    ls = WeightedLayerSum(num_layers=13)
    out = ls([torch.randn(2, 50, 768) for _ in range(13)])

    assert out.shape == (2, 50, 768)
    pesos = torch.softmax(ls.layer_logits, dim=0)
    assert pytest.approx(1.0, abs=1e-6) == float(pesos.sum())
    # Inicialização uniforme: nenhuma camada é privilegiada de partida.
    assert torch.allclose(pesos, torch.full((13,), 1 / 13), atol=1e-6)


def test_gradiente_atravessa_o_back_end_inteiro():
    """Nenhum parâmetro pode ficar órfão.

    O porte Keras descarta o master node da HS-GAL, e com isso os quatro
    tensores do ramo do master ficam sem gradiente — calculados a cada passo e
    jogados fora. Aqui o master entra no readout, como no AASIST original.
    """
    backend = SSLAASISTBackend(input_dim=32)
    x = torch.randn(2, 40, 32, requires_grad=True)

    backend(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    sem_grad = [n for n, p in backend.named_parameters() if p.grad is None]
    assert sem_grad == []


def test_master_node_entra_no_readout():
    """Se o master fosse descartado, zerá-lo não mudaria a saída."""
    torch.manual_seed(0)
    backend = SSLAASISTBackend(input_dim=32, graph_dim=8).eval()
    x = torch.randn(2, 40, 32)

    with torch.no_grad():
        antes = backend(x).clone()
        backend.hsgal.proj_with_attM.weight.zero_()
        backend.hsgal.proj_without_attM.weight.zero_()
        backend.hsgal.proj_with_attM.bias.zero_()
        backend.hsgal.proj_without_attM.bias.zero_()
        depois = backend(x)

    assert not torch.allclose(antes, depois, atol=1e-6)


# ─── Treinabilidade do backbone ────────────────────────────────────────────


class _FakeBackbone(torch.nn.Module):
    """Stub com a superfície que `configure_finetuning` usa."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Linear(4, 4)
        self.encoder = torch.nn.Linear(4, 4)
        self.freeze_called = False

    def freeze_feature_encoder(self):
        self.freeze_called = True
        for p in self.feature_extractor.parameters():
            p.requires_grad = False


def test_congelado_declara_zero_parametros_treinaveis():
    info = configure_finetuning(_FakeBackbone(), freeze_backbone=True)

    assert info["backbone_trainable"] is False
    assert info["backbone_trainable_params"] == 0
    assert info["feature_encoder_frozen"] is False


def test_destravado_declara_o_que_realmente_treina():
    """A declaração vem da CONTAGEM, não da flag.

    Era exatamente aqui que o runner mentia: `--no-freeze-backbone` ligava
    `requires_grad`, mas os embeddings saíam de um `no_grad()` e o otimizador
    só recebia a cabeça — o artefato declarava `backbone_trainable: true` num
    run integralmente congelado.
    """
    backbone = _FakeBackbone()

    info = configure_finetuning(backbone, freeze_backbone=False)

    assert info["backbone_trainable"] is True
    assert info["backbone_trainable_params"] > 0
    # Extrator convolucional congelado mesmo no fine-tuning (prática padrão
    # wav2vec2/WavLM/HuBERT, seguida também por Tak et al.).
    assert info["feature_encoder_frozen"] is True
    assert backbone.freeze_called is True
    assert info["backbone_trainable_params"] < info["backbone_total_params"]


def test_fine_tuning_de_verdade_muda_os_pesos_do_backbone():
    """Um passo de treino tem de alterar o backbone, não só o back-end."""
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, x):
            return self.lin(x)

    backbone = Tiny()
    backend = SSLAASISTBackend(input_dim=16, proj_dim=16, t_fixed=8, graph_dim=8)
    antes = backbone.lin.weight.detach().clone()

    opt = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": 1e-2},
            {"params": backend.parameters(), "lr": 1e-2},
        ]
    )
    logits = backend(backbone(torch.randn(4, 20, 16)))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    opt.step()

    assert not torch.allclose(antes, backbone.lin.weight)


# ─── Guarda de CLI ─────────────────────────────────────────────────────────


def test_runner_recusa_destravar_backbone_no_caminho_de_cache():
    """A combinação que produzia o artefato mentiroso agora é erro de entrada.

    `--backend mlp` pré-calcula embeddings sob `no_grad()`: destravar o
    backbone ali não treina nada. Em vez de aceitar e declarar errado, o
    parser recusa.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(
                _PROJECT_ROOT
                / "scripts"
                / "benchmark"
                / "run_wavlm_original_benchmark.py"
            ),
            "--architecture",
            "wavlm",
            "--dataset",
            "inexistente.npz",
            "--out",
            "inexistente",
            "--backend",
            "mlp",
            "--no-freeze-backbone",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(_PROJECT_ROOT),
    )

    assert proc.returncode != 0
    assert "--no-freeze-backbone exige --backend aasist" in proc.stderr


def test_runner_recusa_grafo_aasist_com_backbone_congelado():
    """A combinação SIMÉTRICA mente do outro lado.

    `--backend aasist` renomeia a entrada para "WavLM AASIST"/"HuBERT AASIST",
    e o manifesto oficial declara essas duas com o variante
    `pytorch_finetuned_aasist_graph`. Como `--freeze-backbone` é o DEFAULT do
    parser (serve às entradas `Original`), esquecer `--no-freeze-backbone`
    produziria um artefato rotulado como fine-tuning tendo treinado só o
    back-end — e não existe entrada de manifesto para um AASIST congelado.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(
                _PROJECT_ROOT
                / "scripts"
                / "benchmark"
                / "run_wavlm_original_benchmark.py"
            ),
            "--architecture",
            "wavlm",
            "--dataset",
            "inexistente.npz",
            "--out",
            "inexistente",
            "--backend",
            "aasist",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(_PROJECT_ROOT),
    )

    assert proc.returncode != 0
    assert "--backend aasist exige --no-freeze-backbone" in proc.stderr


def test_manifesto_declara_as_duas_variantes_com_runner_ssl():
    from benchmarks.config import (
        ALL_TCC_ARCHITECTURES,
        SSL_DOCKER_ARCHITECTURES,
        SSL_FINETUNED_ARCHITECTURES,
    )

    # VAZIA desde 2026-08-11: o escopo oficial só tem SSL congelado (ver
    # benchmarks/config.py). A derivação segue no lugar para que
    # reintroduzir uma entrada `:ssl_finetuned` volte a acionar as flags.
    assert SSL_FINETUNED_ARCHITECTURES == []
    # Se voltarem, não podem cair na lista que `benchmarks.runner` treina
    # pelo caminho Keras.
    for arch in SSL_FINETUNED_ARCHITECTURES:
        assert arch in SSL_DOCKER_ARCHITECTURES
        assert arch not in ALL_TCC_ARCHITECTURES
