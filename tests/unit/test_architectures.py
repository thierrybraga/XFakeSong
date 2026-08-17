"""Registry de arquiteturas: contratos de registro, construção e parâmetros.

SUJEITO: `architectures/registry.py` e `factory.py` — que toda arquitetura
declarada exista, aceite seus `default_params`, e que os três lugares onde um
hiperparâmetro pode viver (registry, `create_model`, `planning`) não divirjam.

O bloco final (sincronia das três fontes) veio em 2026-08-17 de
`test_resume_guards_and_artifacts.py`, que agrupava por data de correção.
"""

import importlib
import inspect

import numpy as np
import pytest

from app.domain.models.architectures.factory import ArchitectureFactoryRegistry
from app.domain.models.architectures.registry import (
    _TRAINING_ONLY_PARAM_KEYS,
    ArchitectureRegistry,
)


def test_registry_initialization():
    registry = ArchitectureRegistry()
    available_architectures = registry.list_architectures()

    assert "AASIST" in available_architectures
    assert "RawGAT-ST" in available_architectures


def test_get_architecture_info():
    registry = ArchitectureRegistry()
    info = registry.get_architecture("AASIST")

    assert info is not None
    assert info.name == "AASIST"
    assert "default" in info.supported_variants


def test_create_factory():
    registry = ArchitectureFactoryRegistry()
    factory = registry.get_factory("AASIST")

    assert factory is not None
    assert hasattr(factory, "create_model")


# A construção real do AASIST (SincConv sobre áudio bruto (48000, 1)) é
# exercida em tests/integration/test_architectures_build.py, que roda com
# TensorFlow de verdade. Aqui existia um `test_create_model_aasist` marcado
# com `@pytest.mark.skip` incondicional pelo mesmo motivo — nunca executava e
# duplicava a cobertura da integração.


# ─── Guardas contra "config morto" e listas divergentes ────────────────────
# Estes testes cobrem classes de bug que já ocorreram mais de uma vez no
# projeto: parâmetros declarados no registry que nunca chegavam ao modelo, e
# listas de arquiteturas mantidas à mão em lugares diferentes.

def _architecture_cases():
    registry = ArchitectureRegistry()
    return [(name, info) for name, info in registry.get_all_architectures().items()]


@pytest.mark.parametrize(
    "name,info", _architecture_cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_default_params_are_accepted_by_builder(name, info):
    """Todo `default_params` deve ser consumível pelo `create_model` da arquitetura.

    Sem esta guarda, uma chave de hiperparâmetro pode ficar anos no registry
    sem NENHUM efeito (o builder a engole via **kwargs) — foi o caso de
    hidden_dim/num_layers no AASIST, de cinco chaves no Sonic Sleuth e das
    specs antigas do Ensemble.
    """
    module = importlib.import_module(info.module_path)
    create_model = getattr(module, info.function_name)
    signature = inspect.signature(create_model)
    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )

    model_keys = set(info.default_params) - set(_TRAINING_ONLY_PARAM_KEYS)

    if not accepts_var_keyword:
        # Sem **kwargs, `create_model` precisa nomear cada chave — o contrário
        # dispararia TypeError na criação.
        unknown = sorted(model_keys - set(signature.parameters))
        assert not unknown, (
            f"{name}: chaves de default_params que o create_model não aceita "
            f"(config morto): {unknown}"
        )
        return

    # Com **kwargs, `create_model` engole qualquer chave em silêncio. A chave
    # precisa então ser nomeada por ALGUM builder do módulo — se não aparece em
    # nenhuma assinatura, é config morto garantido.
    known_params = set()
    for _, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ != module.__name__:
            continue
        known_params.update(inspect.signature(func).parameters)

    unknown = sorted(model_keys - known_params)
    assert not unknown, (
        f"{name}: chaves de default_params que nenhum builder de "
        f"{info.module_path} nomeia (config morto): {unknown}"
    )


def test_settings_architecture_list_matches_registry():
    """`TrainingConfig.available_architectures` espelha o registry.

    `app/core/` não pode importar `app/domain/`, então a lista é estática — e
    justamente por isso precisa de uma guarda: estava com 7 das 12
    arquiteturas neurais registradas.
    """
    from app.core.config.settings import TrainingConfig

    registry = ArchitectureRegistry()
    assert set(TrainingConfig().available_architectures) == set(
        registry.list_architectures_snake()
    )


def test_registry_and_factory_agree_on_input_shapes():
    """Registry e factory precisam responder o mesmo sobre um mesmo contrato.

    O `validate_input_shape` do registry rejeitava áudio bruto 1-D `(T,)`
    enquanto a factory aceitava a mesma forma.
    """
    registry = ArchitectureRegistry()
    factory_registry = ArchitectureFactoryRegistry()

    cases = [
        ("AASIST", (48000,), True),
        ("AASIST", (48000, 1), True),
        ("AASIST", (100, 80), False),
        ("Conformer", (100, 80), True),
        ("Conformer", (100, 13), False),
    ]
    for name, shape, expected in cases:
        assert registry.validate_input_shape(name, shape) is expected, (name, shape)
        assert (
            factory_registry.validate_compatibility(name, shape) is expected
        ), (name, shape)


def test_conformer_has_single_paper_configuration():
    """Conformer consolidado numa configuração única (Conformer-M do paper).

    Havia duas variantes com os nomes INVERTIDOS: 'conformer' com 8 blocos e
    'conformer_lite' com 16 — a "lite" era ~2× maior que a completa.
    """
    from app.domain.models.architectures import conformer

    params = conformer._CONFORMER_M_PARAMS
    assert params["num_blocks"] == 16
    assert params["d_model"] == 256
    assert params["num_heads"] == 4
    assert params["d_ff"] == 4 * params["d_model"]
    assert params["conv_kernel_size"] == 31
    # 'conformer_lite' sobrevive apenas como alias da mesma configuração.
    assert {"conformer", "conformer_lite", "conformer_m"} <= conformer._CONFORMER_ALIASES

    registry = ArchitectureRegistry()
    variants = registry.get_architecture("Conformer").supported_variants
    assert set(variants) <= conformer._CONFORMER_ALIASES | {"default"}


def test_sonic_sleuth_paper_variant_is_registered_and_faithful():
    """A configuração literal da Figura 3 do artigo é alcançável por variante."""
    from app.domain.models.architectures import sonic_sleuth

    registry = ArchitectureRegistry()
    variants = registry.get_architecture("Sonic Sleuth").supported_variants
    assert "sonic_sleuth_paper" in variants

    cfg = sonic_sleuth._SONIC_SLEUTH_PAPER_CONFIG
    assert cfg["num_conv_blocks"] == 3          # 32 → 64 → 128
    assert cfg["use_residual"] is False
    assert cfg["use_se_blocks"] is False
    assert cfg["use_gap_gmp"] is False          # Flatten, como na figura
    assert cfg["classifier_dropout"] == 0.1
    assert cfg["learning_rate"] == 1e-3


def test_ast_pretrained_never_degrades_silently():
    """`pretrained=True` no AST carrega pesos AudioSet OU falha — nunca no-op.

    A flag muda o significado científico do resultado: o AST do paper parte de
    pesos ImageNet→AudioSet. Fora da configuração ViT-Base (a do checkpoint) a
    transferência é impossível e precisa ERRAR, não seguir do zero.
    """
    from app.domain.models.architectures import spectrogram_transformer
    from app.domain.models.architectures.ast_pretrained import (
        ASTPretrainedUnavailable,
    )

    with pytest.raises(ASTPretrainedUnavailable, match="ViT-Base"):
        spectrogram_transformer.create_model(
            (64, 64), num_classes=2, pretrained=True,
            embed_dim=32, num_blocks=1, num_heads=2, ff_dim=64,
        )


def test_benchmark_plan_enables_ast_pretrained_weights():
    """O plano do benchmark parte de pesos pré-treinados, como o artigo.

    Treinar o ViT-Base do zero sobre este dataset é o regime documentado de
    colapso; o registry mantém False só para não exigir rede em testes/CI.
    """
    from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS

    assert NEURAL_BENCHMARK_HPARAMS["spectrogramtransformer"]["pretrained"] is True


def test_ast_small_variant_is_registered():
    """Variante dimensionada para treino do zero (sem pesos pré-treinados)."""
    from app.domain.models.architectures import spectrogram_transformer

    registry = ArchitectureRegistry()
    variants = registry.get_architecture("SpectrogramTransformer").supported_variants
    assert "spectrogram_transformer_small" in variants

    small = spectrogram_transformer._AST_SMALL_PARAMS
    base_embed = 768  # ViT-Base do paper
    assert small["embed_dim"] < base_embed
    assert small["ff_dim"] == 4 * small["embed_dim"]
    assert small["embed_dim"] % small["num_heads"] == 0


def test_output_head_convention_is_uniform():
    """num_classes>=2 → softmax de N unidades em TODAS as variantes.

    O `ensemble_adaptive` usava `1 if num_classes <= 2`, ou seja, emitia uma
    sigmoid enquanto o resto do projeto emitia softmax de 2 — duas convenções
    de saída convivendo.
    """
    from app.domain.models.architectures import ensemble

    shape = (16000, 1)
    for variant in ("ensemble", "ensemble_adaptive"):
        model = ensemble.create_model(shape, num_classes=2, architecture=variant)
        assert model.output_shape[-1] == 2, (variant, model.output_shape)


def test_factory_and_registry_share_num_classes_default():
    """As duas portas de entrada precisam do MESMO default de num_classes."""
    import inspect

    from app.domain.models.architectures import factory
    from app.domain.models.architectures import registry as registry_mod

    factory_default = inspect.signature(
        factory.create_model_by_name
    ).parameters["num_classes"].default
    registry_default = inspect.signature(
        registry_mod.create_model_by_name
    ).parameters["num_classes"].default
    assert factory_default == registry_default == 2


# ─── Conformidade com os artigos ──────────────────────────────────────────

def test_graph_attention_follows_paper_formulation():
    """RawGAT-ST/AASIST usam produto par-a-par + tanh + TEMPERATURA.

    Não é o GAT aditivo de Velickovic. A temperatura (2.0 nos GATs, 100.0 nas
    HS-GAL) controla o quanto a atenção se aproxima de uma média uniforme — é
    um hiperparâmetro explícito do artigo, e sem ele a camada não é a do paper.
    """
    import tensorflow as tf

    from app.domain.models.architectures.layers import AASISTGraphAttentionLayer

    x = tf.random.normal((4, 12, 32))
    low = AASISTGraphAttentionLayer(16, temperature=1.0)
    high = AASISTGraphAttentionLayer(16, temperature=100.0)
    low.build(x.shape)
    high.build(x.shape)
    high.att_proj.set_weights(low.att_proj.get_weights())
    high.att_weight.assign(low.att_weight)

    att_low = tf.squeeze(low._derive_att_map(x), -1)
    att_high = tf.squeeze(high._derive_att_map(x), -1)

    # Temperatura alta ⇒ atenção mais uniforme.
    assert float(tf.math.reduce_std(att_high)) < float(tf.math.reduce_std(att_low))
    # Combinação convexa no eixo agregado pelo matmul.
    assert np.allclose(tf.reduce_sum(att_low, axis=2).numpy(), 1.0, atol=1e-4)


def test_hsgal_has_three_edge_type_parameter_sets():
    """A HS-GAL do AASIST (§2.3) tem parâmetros por TIPO DE ARESTA.

    Uma atenção homogênea com *type embeddings* — como havia antes — não
    reproduz a contribuição que dá nome à camada.
    """
    import tensorflow as tf

    from app.domain.models.architectures.layers import (
        AASISTHtrgGraphAttentionLayer,
    )

    layer = AASISTHtrgGraphAttentionLayer(out_features=16, temperature=100.0)
    out1, out2, master = layer(
        [tf.random.normal((2, 5, 24)), tf.random.normal((2, 7, 24))]
    )

    weight_names = {w.name for w in layer.weights}
    for expected in ("att_weight11", "att_weight22", "att_weight12", "att_weightM"):
        assert any(expected in name for name in weight_names), expected

    # Tipos preservados na saída + master node de 1 nó.
    assert out1.shape[1] == 5 and out2.shape[1] == 7
    assert master.shape[1] == 1


def test_conformer_relative_positions_are_descending():
    """Transformer-XL/Conformer indexam R por DISTÂNCIA relativa (decrescente)."""
    import tensorflow as tf

    from app.domain.models.architectures.conformer import (
        RelativePositionalEncoding,
    )

    layer = RelativePositionalEncoding(d_model=8, max_len=64)
    layer.build((None, None, 8))
    pos = layer(tf.zeros((1, 5, 8))).numpy()[0]
    table = layer.pe.numpy()[0]

    assert np.allclose(pos[0], table[4]), "1ª posição = maior distância"
    assert np.allclose(pos[-1], table[0]), "última posição = distância 0"


def test_cct_tokenizer_matches_paper():
    """O tokenizer do CCT é [Conv+ReLU+MaxPool]×2 — sem Squeeze-and-Excitation."""
    from app.domain.models.architectures import hybrid_cnn_transformer as cct

    model = cct.create_model((100, 80), num_classes=2, transformer_layers=1)
    kinds = [type(sub).__name__ for sub in model.get_layer("cct_tokenizer").conv_layers]
    assert kinds == ["Conv2D", "MaxPooling2D", "Conv2D", "MaxPooling2D"], kinds


def test_ast_input_contract_matches_paper_regime():
    """O AST precisa de tokens suficientes para o ViT-Base fazer sentido.

    Com o contrato antigo (100×80) sobravam 63 tokens contra os 1212 do artigo,
    alimentando 85M parâmetros — o regime que degradava o treino até chute.
    """
    registry = ArchitectureRegistry()
    req = registry.get_architecture("SpectrogramTransformer").input_requirements

    assert req["feature_dim"] == 128, "o AST usa 128 bandas mel"
    assert req["min_sequence_length"] == 300, "3 s com hop de 10 ms = 300 quadros"

    tokens = ((req["min_sequence_length"] - 16) // 10 + 1) * (
        (req["feature_dim"] - 16) // 10 + 1
    )
    assert tokens > 300, tokens


def test_ast_applies_paper_input_normalization():
    """AST normaliza a entrada para média 0 e desvio 0,5 (§2.1)."""
    import tensorflow as tf

    from app.domain.models.architectures.layers import ASTInputNormalization

    out = ASTInputNormalization()(tf.random.normal((3, 40, 16, 1)) * 4.0 - 7.0)
    assert abs(float(np.mean(out.numpy()))) < 1e-3
    assert abs(float(np.std(out.numpy())) - 0.5) < 1e-2


def test_rawnet2_declares_both_paper_variants():
    """RawNet2 de verificação de locutor ≠ baseline anti-spoofing do ASVspoof."""
    from app.domain.models.architectures import rawnet2

    registry = ArchitectureRegistry()
    variants = registry.get_architecture("RawNet2").supported_variants
    assert "rawnet2_antispoofing" in variants

    asv = rawnet2._RAWNET2_ANTISPOOFING_PARAMS
    assert asv["sinc_filters"] == 20          # o baseline usa 20, não 128
    assert asv["res_filters"] == [20, 20, 128, 128, 128, 128]
    assert asv["gru_layers"] == 3


def test_am_softmax_margin_is_applied_in_the_loss():
    """A margem CosFace precisa entrar pela loss — na camada ela é inerte.

    No grafo funcional os rótulos nunca chegam ao `call` da `AMSoftmaxLayer`,
    então `margin=` lá não tem efeito: a cabeça 'am_softmax' era AM-Softmax
    só no nome.
    """
    import tensorflow as tf

    from app.domain.models.architectures.layers import AMSoftmaxCrossEntropy

    logits = tf.constant([[2.0, -1.0], [-1.5, 3.0]])
    y_true = tf.constant([0, 1])

    com_margem = float(
        tf.reduce_mean(AMSoftmaxCrossEntropy(scale=15.0, margin=0.35)(y_true, logits))
    )
    sem_margem = float(
        tf.reduce_mean(AMSoftmaxCrossEntropy(scale=15.0, margin=0.0)(y_true, logits))
    )
    assert com_margem > sem_margem


# ─── sincronia entre as TRÊS fontes de hiperparâmetro ──────────────────────
#
# Movidos em 2026-08-17 de `test_resume_guards_and_artifacts.py`, que agrupava
# por DATA de correção. O sujeito é o contrato do registry/arquitetura, que é o
# deste arquivo — e fica ao lado de `test_default_params_are_accepted_by_builder`,
# a guarda irmã que checa a CHAVE enquanto estas checam o VALOR.


@pytest.mark.parametrize(
    "arch_name,module_name,plan_key",
    [("RawGAT-ST", "rawgat_st", "rawgatst"), ("AASIST", "aasist", "aasist")],
)
def test_hparams_batem_nas_tres_fontes(arch_name, module_name, plan_key):
    """CLAUDE.md exige registry + create_model + planning em sincronia.

    Um default divergente no `create_model` faz o caminho do app/Gradio treinar
    com uma receita diferente da que o benchmark documenta.
    """
    from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS

    info = ArchitectureRegistry().get_architecture(arch_name)
    plano = NEURAL_BENCHMARK_HPARAMS[plan_key]
    modulo = importlib.import_module(info.module_path)
    assinatura = inspect.signature(getattr(modulo, info.function_name)).parameters

    comparaveis = [k for k in info.default_params if k in plano and k in assinatura]
    assert comparaveis, f"{arch_name}: nada em comum para comparar"

    for chave in comparaveis:
        registry_v = info.default_params[chave]
        plano_v = plano[chave]
        builder_v = assinatura[chave].default
        assert registry_v == pytest.approx(plano_v), (
            f"{arch_name}.{chave}: registry={registry_v} != planning={plano_v}"
        )
        assert registry_v == pytest.approx(builder_v), (
            f"{arch_name}.{chave}: registry={registry_v} != create_model={builder_v}"
        )


def test_global_clipnorm_do_rawgat_chega_ao_construtor():
    """Era literal 0.7 no compile enquanto o registry declarava 0.5.

    Não basta existir nas três fontes: o runner precisa PROMOVER a chave para
    `parameters`, senão ela não chega ao `create_model` e volta a ser config
    morto.
    """
    import ast
    from pathlib import Path

    from app.domain.models.architectures import rawgat_st

    assinatura = inspect.signature(rawgat_st.create_model).parameters
    assert "global_clipnorm" in assinatura

    runner = Path(__file__).resolve().parents[2] / "benchmarks/runner.py"
    fonte = runner.read_text(encoding="utf-8")
    inicio = fonte.find('elif compact in {"aasist", "rawgatst"}')
    assert inicio > 0, "ramo de promoção do rawgatst não encontrado"
    assert '"global_clipnorm"' in fonte[inicio : inicio + 1200], (
        "global_clipnorm não está no whitelist de promoção do runner"
    )
    # Ler o AST garante que o arquivo segue parseável.
    ast.parse(fonte)
