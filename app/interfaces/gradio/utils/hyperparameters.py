import gradio as gr
from typing import Dict, Any, Tuple
import logging
from app.core.db_setup import get_flask_app
from app.domain.models.architecture_config import ArchitectureConfig
from app.extensions import db

# Global state for hyperparameters (cache local da sessão)
hp_state: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger("hyperparameters")

def optimize_default(
    architecture: str,
    batch_v: float,
    epochs_v: float,
    lr_v: float,
    dropout_v: float,
    l2_v: float,
    val_v: float,
    d_model_v: float,
    num_heads_v: float,
    num_blocks_v: float,
    patch_v: str,
    ff_dim_v: float,
    filters_v: str,
    kernel_sizes_v: str,
    att_heads_v: float,
    hid_dim_v: float,
    n_layers_v: float,
    conv_kernel_v: str,
    hid_base_v: float,
    n_layers_base_v: float,
    base_filters_v: float,
    res_blocks_v: float,
    trans_layers_v: float,
    att_heads_h_v: float,
    conv_filters_raw_v: str,
    gru_units_v: float,
    dense_units_v: float
) -> Tuple:
    """
    Otimiza e salva os hiperparâmetros padrão para uma arquitetura no BANCO DE DADOS.
    Retorna atualizações para os componentes da interface Gradio.
    """
    try:
        flask_app = get_flask_app()
        
        with flask_app.app_context():
            # 1. Carregar configuração atual do Banco
            arch_config = ArchitectureConfig.query.filter_by(
                architecture_name=architecture, 
                variant_name="default"
            ).first()
            
            current_hp = arch_config.parameters if arch_config else {}
            
            merged = dict(current_hp)

            # 2. Atualizar com valores fornecidos (apenas se válidos)
            if isinstance(batch_v, (int, float)):
                merged["batch_size"] = int(batch_v)
            if isinstance(epochs_v, (int, float)):
                merged["epochs"] = int(epochs_v)
            if isinstance(lr_v, (int, float)):
                merged["learning_rate"] = float(lr_v)
            if isinstance(dropout_v, (int, float)):
                merged["dropout_rate"] = float(dropout_v)
            if isinstance(l2_v, (int, float)):
                merged["l2_reg_strength"] = float(l2_v)
            if isinstance(val_v, (int, float)):
                merged["validation_split"] = float(val_v)

            arch_l = (architecture or "").lower()

            # Parâmetros específicos por arquitetura
            if "spectrogramtransformer" in arch_l or "spectrogram" in arch_l or "transformer" in arch_l:
                if isinstance(d_model_v, (int, float)):
                    merged["d_model"] = int(d_model_v)
                if isinstance(num_heads_v, (int, float)):
                    merged["num_heads"] = int(num_heads_v)
                if isinstance(num_blocks_v, (int, float)):
                    merged["num_blocks"] = int(num_blocks_v)
                if isinstance(ff_dim_v, (int, float)):
                    merged["ff_dim"] = int(ff_dim_v)
                try:
                    if isinstance(patch_v, str) and "x" in patch_v:
                        parts = patch_v.split("x")
                        merged["patch_size"] = (int(parts[0]), int(parts[1]))
                except Exception:
                    pass

            if "multiscalecnn" in arch_l or "multiscale" in arch_l:
                if isinstance(filters_v, str) and filters_v:
                    try:
                        merged["filters"] = [int(x) for x in filters_v.split(",")]
                    except Exception:
                        pass
                if isinstance(kernel_sizes_v, str) and kernel_sizes_v:
                    try:
                        merged["kernel_sizes"] = [int(x) for x in kernel_sizes_v.split(",")]
                    except Exception:
                        pass

            if arch_l == "conformer":
                if isinstance(att_heads_v, (int, float)):
                    merged["attention_heads"] = int(att_heads_v)
                if isinstance(hid_dim_v, (int, float)):
                    merged["hidden_dim"] = int(hid_dim_v)
                if isinstance(n_layers_v, (int, float)):
                    merged["num_layers"] = int(n_layers_v)
                if isinstance(conv_kernel_v, str) and conv_kernel_v:
                    try:
                        merged["conv_kernel_size"] = int(conv_kernel_v)
                    except Exception:
                        pass

            if "hybrid" in arch_l:
                if isinstance(base_filters_v, (int, float)):
                    merged["base_filters"] = int(base_filters_v)
                if isinstance(res_blocks_v, (int, float)):
                    merged["num_residual_blocks"] = int(res_blocks_v)
                if isinstance(trans_layers_v, (int, float)):
                    merged["num_transformer_layers"] = int(trans_layers_v)
                if isinstance(att_heads_h_v, (int, float)):
                    merged["attention_heads"] = int(att_heads_h_v)

            if arch_l == "aasist" or "rawgat" in arch_l:
                if isinstance(hid_base_v, (int, float)):
                    merged["hidden_dim"] = int(hid_base_v)
                if isinstance(n_layers_base_v, (int, float)):
                    merged["num_layers"] = int(n_layers_base_v)

            if "rawnet2" in arch_l:
                if isinstance(conv_filters_raw_v, str) and conv_filters_raw_v:
                    try:
                        merged["conv_filters"] = [int(x) for x in conv_filters_raw_v.split(",")]
                    except Exception:
                        pass
                if isinstance(gru_units_v, (int, float)):
                    merged["gru_units"] = int(gru_units_v)
                if isinstance(dense_units_v, (int, float)):
                    merged["dense_units"] = int(dense_units_v)

            # 3. Salvar configurações atualizadas no BANCO
            if not arch_config:
                arch_config = ArchitectureConfig(
                    architecture_name=architecture,
                    variant_name="default",
                    parameters=merged,
                    description=f"Configuração padrão para {architecture}"
                )
                db.session.add(arch_config)
            else:
                arch_config.parameters = merged
                # Forçar atualização de flag de modificação se necessário
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(arch_config, "parameters")
            
            db.session.commit()
            logger.info(f"Hiperparâmetros para {architecture} salvos no banco.")

            # Atualizar estado global
            hp_state[architecture] = merged

            return (
                merged,
                gr.update(value=int(merged.get("batch_size", batch_v))),
                gr.update(value=int(merged.get("epochs", epochs_v))),
                gr.update(value=float(merged.get("learning_rate", lr_v))),
                gr.update(value=float(merged.get("dropout_rate", dropout_v))),
                gr.update(value=float(merged.get("l2_reg_strength", l2_v))),
                gr.update(value=float(merged.get("validation_split", val_v))),
                gr.update(
                    value=int(
                        merged.get(
                            "d_model", d_model_v)), visible=(
                        "spectrogram" in arch_l or "transformer" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "num_heads", num_heads_v)), visible=(
                        "spectrogram" in arch_l or "transformer" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "num_blocks", num_blocks_v)), visible=(
                        "spectrogram" in arch_l or "transformer" in arch_l)),
                gr.update(
                    value=(
                        f"{merged.get('patch_size')[0]}x{merged.get('patch_size')[1]}" if isinstance(
                            merged.get('patch_size'), (tuple, list)) else patch_v), visible=(
                        "spectrogram" in arch_l or "transformer" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "ff_dim", ff_dim_v)), visible=(
                        "spectrogram" in arch_l or "transformer" in arch_l)),
                gr.update(
                    value=(
                        ",".join(
                            str(x) for x in merged.get(
                                "filters", [])) or filters_v), visible=(
                        "multiscale" in arch_l)),
                gr.update(
                    value=(
                        ",".join(
                            str(x) for x in merged.get(
                                "kernel_sizes", [])) or kernel_sizes_v), visible=(
                        "multiscale" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "attention_heads", att_heads_v)), visible=(
                        arch_l == "conformer")),
                gr.update(
                    value=int(
                        merged.get(
                            "hidden_dim", hid_dim_v)), visible=(
                        arch_l == "conformer")),
                gr.update(
                    value=int(
                        merged.get(
                            "num_layers", n_layers_v)), visible=(
                        arch_l == "conformer")),
                gr.update(
                    value=str(
                        merged.get(
                            "conv_kernel_size", conv_kernel_v)), visible=(
                        arch_l == "conformer")),
                gr.update(
                    value=int(
                        merged.get(
                            "hidden_dim", hid_base_v)), visible=(
                        arch_l == "aasist" or "rawgat" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "num_layers", n_layers_base_v)), visible=(
                        arch_l == "aasist" or "rawgat" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "base_filters", base_filters_v)), visible=(
                        "hybrid" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "num_residual_blocks", res_blocks_v)), visible=(
                        "hybrid" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "num_transformer_layers", trans_layers_v)), visible=(
                        "hybrid" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "attention_heads", att_heads_h_v)), visible=(
                        "hybrid" in arch_l)),
                gr.update(
                    value=(
                        ",".join(
                            str(x) for x in merged.get(
                                "conv_filters", [])) or conv_filters_raw_v), visible=(
                        "rawnet2" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "gru_units", gru_units_v)), visible=(
                        "rawnet2" in arch_l)),
                gr.update(
                    value=int(
                        merged.get(
                            "dense_units", dense_units_v)), visible=(
                        "rawnet2" in arch_l))
            )
    except Exception as e:
        logger.error(f"Erro ao salvar hiperparâmetros: {e}")
        # Retorna erro e updates vazios (sem alteração)
        return ({"error": str(e)},) + (gr.update(),) * 26


def load_defaults(architecture: str) -> Tuple:
    """
    Carrega os hiperparâmetros padrão para uma arquitetura do BANCO DE DADOS.
    Não salva nem sobrescreve, apenas lê.
    """
    try:
        flask_app = get_flask_app()
        
        hp = {}
        with flask_app.app_context():
            # Tenta carregar do banco
            arch_config = ArchitectureConfig.query.filter_by(
                architecture_name=architecture, 
                variant_name="default"
            ).first()
            
            if arch_config:
                hp = arch_config.parameters
            else:
                hp = {}

        hp_state[architecture] = hp
        merged = dict(hp)

        # Helper para pegar valor ou default seguro
        def get_val(key, default):
            return merged.get(key, default)

        arch_l = (architecture or "").lower()

        return (
            merged,
            gr.update(value=int(get_val("batch_size", 32))),
            gr.update(value=int(get_val("epochs", 10))),
            gr.update(value=float(get_val("learning_rate", 0.001))),
            gr.update(value=float(get_val("dropout_rate", 0.3))),
            gr.update(value=float(get_val("l2_reg_strength", 0.0001))),
            gr.update(value=float(get_val("validation_split", 0.2))),
            gr.update(value=int(get_val("d_model", 64)), visible=(
                "spectrogram" in arch_l or "transformer" in arch_l)),
            gr.update(value=int(get_val("num_heads", 4)), visible=(
                "spectrogram" in arch_l or "transformer" in arch_l)),
            gr.update(value=int(get_val("num_blocks", 2)), visible=(
                "spectrogram" in arch_l or "transformer" in arch_l)),
            gr.update(
                value=(
                    f"{get_val('patch_size', (4, 4))[0]}x{get_val('patch_size', (4, 4))[1]}" if isinstance(
                        get_val('patch_size', (4, 4)), (tuple, list)) and len(
                        get_val('patch_size', (4, 4))) >= 2 else str(
                        get_val('patch_size', "4x4"))), visible=(
                    "spectrogram" in arch_l or "transformer" in arch_l)),
            gr.update(value=int(get_val("ff_dim", 128)), visible=(
                "spectrogram" in arch_l or "transformer" in arch_l)),
            gr.update(
                value=(
                    ",".join(
                        str(x) for x in get_val(
                            "filters", [
                                32, 64, 128]))), visible=(
                    "multiscale" in arch_l)),
            gr.update(
                value=(
                    ",".join(
                        str(x) for x in get_val(
                            "kernel_sizes", [
                                3, 3, 3]))), visible=(
                    "multiscale" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "attention_heads", 4)), visible=(
                    arch_l == "conformer")),
            gr.update(
                value=int(
                    get_val(
                        "hidden_dim", 144)), visible=(
                    arch_l == "conformer")),
            gr.update(
                value=int(
                    get_val(
                        "num_layers", 4)), visible=(
                    arch_l == "conformer")),
            gr.update(
                value=str(
                    get_val(
                        "conv_kernel_size", 31)), visible=(
                    arch_l == "conformer")),
            gr.update(
                value=int(
                    get_val(
                        "hidden_dim", 64)), visible=(
                    arch_l == "aasist" or "rawgat" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "num_layers", 2)), visible=(
                    arch_l == "aasist" or "rawgat" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "base_filters", 32)), visible=(
                    "hybrid" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "num_residual_blocks", 2)), visible=(
                    "hybrid" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "num_transformer_layers", 2)), visible=(
                    "hybrid" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "attention_heads", 4)), visible=(
                    "hybrid" in arch_l)),
            gr.update(
                value=(
                    ",".join(
                        str(x) for x in get_val(
                            "conv_filters", [
                                128, 128]))), visible=(
                    "rawnet2" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "gru_units", 1024)), visible=(
                    "rawnet2" in arch_l)),
            gr.update(
                value=int(
                    get_val(
                        "dense_units", 1024)), visible=(
                    "rawnet2" in arch_l))
        )
    except Exception as e:
        logger.error(f"Erro ao carregar hiperparâmetros: {e}")
        return ({"error": str(e)},) + (gr.update(),) * 26
