#!/usr/bin/env python3
"""
update_tcc_results.py — Atualiza automaticamente as tabelas e números do TCC
com os resultados reais gerados pelo pipeline de treinamento.

Lê:
  results/training_metrics.json   (gerado por train_advanced.py)
  results/robustness_results.json (gerado por robustness_test.py)

Escreve:
  results/TCC_XFakeSong_Final.tex (atualiza in-place, faz backup .bak)

As seções autogenéradas são delimitadas por:
  %% AUTOGEN_BEGIN:<tag>
  %% AUTOGEN_END:<tag>

Uso:
  python scripts/update_tcc_results.py
  python scripts/update_tcc_results.py --dry-run   # mostra diff sem salvar
  python scripts/update_tcc_results.py --check     # exit code 1 se divergência
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_JSON = BASE_DIR / "results" / "training_metrics.json"
ROB_JSON   = BASE_DIR / "results" / "robustness_results.json"
TEX_PATH   = BASE_DIR / "results" / "TCC_XFakeSong_Final.tex"

# ---------------------------------------------------------------------------
# Ordem e nomes de exibição
# ---------------------------------------------------------------------------
ARCH_ORDER = [
    "multiscale_cnn",
    "ensemble_adaptive",
    "efficientnet_lstm",
    "aasist",
    "rawnet2",
]

ARCH_DISPLAY = {
    "multiscale_cnn":    "MultiscaleCNN (Res2Net-50)",
    "ensemble_adaptive": "Ensemble Adaptativo",
    "efficientnet_lstm": "EfficientNet-LSTM",
    "aasist":            "AASIST",
    "rawnet2":           "RawNet2",
}
# Nome curto para tabelas mais estreitas
ARCH_SHORT = {
    "multiscale_cnn":    "MultiscaleCNN",
    "ensemble_adaptive": "Ensemble Adapt.",
    "efficientnet_lstm": "Effic.-LSTM",
    "aasist":            "AASIST",
    "rawnet2":           "RawNet2",
}
# Causa de não-convergência (usada na tabela de apêndice)
ARCH_CAUSE = {
    "efficientnet_lstm": "Seq.\\ PCM longa",
    "aasist":            "SincConv+GAT, CPU",
    "rawnet2":           "GRU sobre PCM, CPU",
    "multiscale_cnn":    "---",
    "ensemble_adaptive": "---",
}


# ---------------------------------------------------------------------------
# Formatação BR
# ---------------------------------------------------------------------------
def fp(v: float, d: int = 2) -> str:
    """Percentagem BR: 97.7833 → '97,78\\%'"""
    return f"{v:.{d}f}\\%".replace(".", ",")

def fd(v: float, d: int = 3) -> str:
    """Decimal BR: 0.997 → '0,997'"""
    return f"{v:.{d}f}".replace(".", ",")

def fl(v: float) -> str:
    """Latência 1 casa: 133.7 → '133,7'"""
    return f"{v:.1f}".replace(".", ",")

def fm(v: float) -> str:
    """Memória 1 casa: 186.8 → '186,8'"""
    return f"{v:.1f}".replace(".", ",")

def fi(v: int) -> str:
    """Inteiro com ponto mil BR: 48965581 → '48.965.581'"""
    return f"{v:,}".replace(",", ".")

def color_sim(c: bool) -> str:
    return r"\textcolor{successgreen}{Sim}" if c else r"\textcolor{dangerred}{N\~{a}o}"

def color_ok(c: bool) -> str:
    return "OK" if c else "n/a"


# ---------------------------------------------------------------------------
# Classificação de convergência
# ---------------------------------------------------------------------------
def is_converged(r: dict) -> bool:
    """Modelo convergiu se accuracy > 55% e não foi skipped/error."""
    if not r:
        return False
    if r.get("skipped") or r.get("error"):
        return False
    return r.get("test_accuracy_pct", 50.0) > 55.0


def get_arch(archs: dict, key: str) -> dict:
    """Retorna dados da arquitetura, garantindo defaults para modelos pulados."""
    r = archs.get(key, {})
    if not r or r.get("skipped") or r.get("error"):
        # Valores padrão para modelos não treinados (acaso)
        return {
            "test_accuracy_pct": 50.00,
            "test_eer_pct":      50.00,
            "test_auc_roc":      0.500,
            "latency_10s_ms":    r.get("latency_10s_ms", 0.0) if r else 0.0,
            "epochs_run":        r.get("epochs_run", 0) if r else 0,
            "params":            r.get("params", 0) if r else 0,
            "memory_mb":         r.get("memory_mb", 0.0) if r else 0.0,
            "best_val_accuracy_pct": 50.0,
            "hyperparams": r.get("hyperparams", {}) if r else {},
            "_skipped": True,
        }
    return r


# ---------------------------------------------------------------------------
# Geradores de conteúdo LaTeX
# ---------------------------------------------------------------------------

def gen_tab_resultados(archs: dict) -> str:
    """Tabela 3 — Desempenho por arquitetura."""
    rows = []
    for key in ARCH_ORDER:
        r   = get_arch(archs, key)
        name = ARCH_DISPLAY[key]
        conv = is_converged(archs.get(key, {}))
        acc  = fp(r["test_accuracy_pct"])
        eer  = fp(r["test_eer_pct"])
        auc  = fd(r["test_auc_roc"])
        lat  = fl(r["latency_10s_ms"]) if r["latency_10s_ms"] else "---"
        ep   = str(r["epochs_run"])
        cv   = color_sim(conv)
        if conv:
            rows.append(
                f"\\textbf{{{name}}}\n"
                f"  & \\textbf{{{acc}}} & \\textbf{{{eer}}} & \\textbf{{{auc}}}\n"
                f"  & {lat} & {ep} & {cv} \\\\"
            )
        else:
            rows.append(
                f"{name}\n"
                f"  & {acc} & {eer} & {auc}\n"
                f"  & {lat} & {ep} & {cv} \\\\"
            )
    return "\n".join(rows) + "\n"


def gen_tab_complexidade(archs: dict) -> str:
    """Tabela 4 — Complexidade computacional."""
    # Ordenar por params decrescente (mesma lógica do TCC original)
    order_by_params = sorted(
        ARCH_ORDER,
        key=lambda k: get_arch(archs, k).get("params", 0),
        reverse=True,
    )
    rows = []
    for key in order_by_params:
        r    = get_arch(archs, key)
        name = ARCH_SHORT[key]
        params = fi(r["params"]) if r["params"] else "---"
        mem    = fm(r["memory_mb"]) if r["memory_mb"] else "---"
        lat    = fl(r["latency_10s_ms"]) if r["latency_10s_ms"] else "---"
        conv   = is_converged(archs.get(key, {}))
        cv     = color_sim(conv)
        rows.append(f"{name} & {params} & {mem} & {lat} & {cv} \\\\")
    return "\n".join(rows) + "\n"


def gen_tab_robustez(rob_results: dict) -> str:
    """Tabela 5 — Robustez AWGN (acc + EER por SNR)."""
    snr_keys = ["snr_30db", "snr_20db", "snr_10db"]
    rows = []
    for key in ARCH_ORDER:
        r = rob_results.get(key)
        if not r:
            continue
        clean = r.get("clean", {})
        name  = ARCH_SHORT[key]
        row   = f"{name}\n"
        row  += f"  & {fp(clean.get('accuracy_pct', 50))} & {fp(clean.get('eer_pct', 50))}\n"
        for snr in snr_keys:
            s = r.get(snr, {})
            row += f"  & {fp(s.get('accuracy_pct', 50))} & {fp(s.get('eer_pct', 50))}\n"
        row  = row.rstrip("\n") + " \\\\"
        rows.append(row)
    return "\n".join(rows) + "\n"


def gen_tab_comparacao_our(archs: dict) -> str:
    """Tabela 6 — linhas 'este trab.' (apenas modelos convergidos)."""
    rows = []
    for key in ARCH_ORDER:
        r    = archs.get(key, {})
        if not is_converged(r):
            continue
        name_short = ARCH_SHORT[key]
        eer  = fd(r["test_eer_pct"], 2)   # tabela usa valor sem símbolo %
        auc  = fd(r["test_auc_roc"])
        device = r.get("device", "CPU")
        rows.append(
            f"\\textbf{{{name_short} (este trab.)}}\n"
            f"  & \\textbf{{{eer}}} & \\textbf{{{auc}}}\n"
            f"  & BRSpeech-DF ({device}) & --- \\\\"
        )
    return "\n".join(rows) + "\n"


def gen_tab_hyperparams(archs: dict) -> str:
    """Apêndice A2 — hiperparâmetros dos modelos convergidos."""
    conv_archs = [k for k in ARCH_ORDER if is_converged(archs.get(k, {}))]
    if not conv_archs:
        return "% Nenhum modelo convergiu\n"

    # Cabeçalho dinâmico
    header_cols = " & ".join(f"\\textbf{{{ARCH_SHORT[k]}}}" for k in conv_archs)
    # Valores por parâmetro
    def val(key: str, field: str, fmt=str):
        r = get_arch(archs, key)
        hp = r.get("hyperparams", {})
        v = hp.get(field) if field in hp else r.get(field)
        if v is None:
            return "---"
        result = fmt(v)
        # Garantir separador decimal BR para floats
        if isinstance(v, float):
            result = str(result).replace(".", ",")
        return result

    def col(field, fmt=str):
        return " & ".join(val(k, field, fmt) for k in conv_archs)

    rows = [
        f"Learning Rate (inicial)        & {col('lr')} \\\\",
        f"Batch Size                     & {col('batch_size')} \\\\",
        r"L2 Regulariza\c{c}\~{a}o      & " + col("l2") + " \\\\",
        r"\'{E}pocas m\'{a}x.\ " + f"          & {col('epochs_max')} \\\\",
        r"Early Stopping (paci\^{e}ncia) & " + col("patience") + " \\\\",
        f"LR Patience                    & {col('lr_patience')} \\\\",
        r"\'{E}pocas executadas          & " + col("epochs_run") + " \\\\",
        r"Melhor val.\ acur\'{a}cia      & " + " & ".join(
            fp(get_arch(archs, k).get("best_val_accuracy_pct", 0)) for k in conv_archs
        ) + " \\\\",
        r"Par\^{a}metros                 & " + " & ".join(
            fi(get_arch(archs, k).get("params", 0)) for k in conv_archs
        ) + " \\\\",
        r"Mem\'{o}ria modelo             & " + " & ".join(
            fm(get_arch(archs, k).get("memory_mb", 0)) + " MB" for k in conv_archs
        ) + " \\\\",
    ]
    return "\n".join(rows) + "\n"


def gen_tab_nao_convergiu(archs: dict) -> str:
    """Apêndice A3 — arquiteturas que não convergiram."""
    rows = []
    for key in ARCH_ORDER:
        r    = archs.get(key, {})
        conv = is_converged(r)
        if conv:
            continue
        rd   = get_arch(archs, key)
        name = ARCH_DISPLAY[key]
        params = fi(rd["params"]) if rd["params"] else "---"
        mem    = fm(rd["memory_mb"]) if rd["memory_mb"] else "---"
        lat    = fl(rd["latency_10s_ms"]) if rd["latency_10s_ms"] else "---"
        ep     = str(rd["epochs_run"])
        cause  = ARCH_CAUSE.get(key, "---")
        rows.append(f"{name} & {params} & {mem} & {lat} & {ep} & {cause} \\\\")
    if not rows:
        return "% Todos os modelos convergiram\n"
    return "\n".join(rows) + "\n"


def gen_tab_consolidado(archs: dict) -> str:
    """Apêndice A4 — resultados consolidados de todos os modelos."""
    rows = []
    for key in ARCH_ORDER:
        r    = get_arch(archs, key)
        conv = is_converged(archs.get(key, {}))
        name = ARCH_SHORT[key]
        acc  = fp(r["test_accuracy_pct"])
        eer  = fp(r["test_eer_pct"])
        auc  = fd(r["test_auc_roc"])
        lat  = fl(r["latency_10s_ms"]) + "ms" if r["latency_10s_ms"] else "---"
        mem  = fm(r["memory_mb"]) if r["memory_mb"] else "---"
        rows.append(f"{name} & {acc} & {eer} & {auc} & {lat} & {mem} & {color_ok(conv)} \\\\")
    return "\n".join(rows) + "\n"


def gen_tab_robustez_completo(rob_results: dict) -> str:
    """Apêndice A5 — robustez completa com AUC-ROC."""
    snr_keys_labels = [
        ("snr_30db", "30"),
        ("snr_20db", "20"),
        ("snr_10db", "10"),
    ]
    # Only archs with robustness data
    archs_with_rob = [k for k in ARCH_ORDER if k in rob_results]
    if not archs_with_rob:
        return "% Sem dados de robustez\n"

    def ncols():
        return len(archs_with_rob) + 1  # arch name + 4 SNR cols but table is fixed 5

    # Accuracy rows
    acc_rows  = [r"\multicolumn{5}{l}{\textit{Acur\'{a}cia (\%)}} \\"]
    eer_rows  = [r"\midrule", r"\multicolumn{5}{l}{\textit{EER (\%)}} \\"]
    auc_rows  = [r"\midrule", r"\multicolumn{5}{l}{\textit{AUC-ROC}} \\"]

    for key in archs_with_rob:
        r    = rob_results[key]
        name = ARCH_SHORT[key]
        clean = r.get("clean", {})

        acc_row = f"{name} & {fd(clean.get('accuracy_pct', 50), 2)}"
        eer_row = f"{name} & {fd(clean.get('eer_pct', 50), 2)}"
        auc_row = f"{name} & {fd(clean.get('auc_roc', 0.5))}"

        for snr_key, _ in snr_keys_labels:
            s = r.get(snr_key, {})
            acc_row += f" & {fd(s.get('accuracy_pct', 50), 2)}"
            eer_row += f" & {fd(s.get('eer_pct', 50), 2)}"
            auc_row += f" & {fd(s.get('auc_roc', 0.5))}"

        acc_rows.append(acc_row + " \\\\")
        eer_rows.append(eer_row + " \\\\")
        auc_rows.append(auc_row + " \\\\")

    return "\n".join(acc_rows + eer_rows + auc_rows) + "\n"


# ---------------------------------------------------------------------------
# Substituição de marcadores
# ---------------------------------------------------------------------------

def replace_autogen(tex: str, tag: str, new_content: str) -> str:
    """Substitui o conteúdo entre %% AUTOGEN_BEGIN:tag e %% AUTOGEN_END:tag.

    Usa callable replacement para evitar que backslashes no conteúdo LaTeX
    sejam interpretados como sequências de escape pelo módulo re.
    """
    pattern = (
        r"(%% AUTOGEN_BEGIN:" + re.escape(tag) + r"\n)"
        r"(.*?)"
        r"(%% AUTOGEN_END:" + re.escape(tag) + r")"
    )
    begin_marker = f"%% AUTOGEN_BEGIN:{tag}\n"
    end_marker   = f"%% AUTOGEN_END:{tag}"

    def replacer(m):
        return begin_marker + new_content + end_marker

    new_tex, n = re.subn(pattern, replacer, tex, flags=re.DOTALL)
    if n == 0:
        print(f"  AVISO: marcador AUTOGEN_BEGIN:{tag} não encontrado no .tex")
    return new_tex


# ---------------------------------------------------------------------------
# Atualização de números no texto narrativo
# ---------------------------------------------------------------------------

def _best_conv(archs: dict) -> dict:
    """Retorna o melhor modelo convergido (maior accuracy)."""
    best = None
    for k in ARCH_ORDER:
        r = archs.get(k, {})
        if is_converged(r):
            if best is None or r["test_accuracy_pct"] > best["test_accuracy_pct"]:
                best = r
    return best or {}


def update_narrative(tex: str, archs: dict, rob: dict) -> str:
    """Substitui números-chave no texto narrativo.

    Estratégia conservadora: usa padrões específicos para não afetar
    trechos que não sejam referências diretas a métricas dos nossos modelos.
    """
    mc  = get_arch(archs, "multiscale_cnn")
    ens = get_arch(archs, "ensemble_adaptive")
    mc_conv  = is_converged(archs.get("multiscale_cnn", {}))
    ens_conv = is_converged(archs.get("ensemble_adaptive", {}))

    # Mapeamento: padrão exato → novo valor (apenas se o modelo convergiu)
    subs = []

    if mc_conv:
        acc = f"{mc['test_accuracy_pct']:.2f}".replace(".", ",")
        eer = f"{mc['test_eer_pct']:.2f}".replace(".", ",")
        auc = f"{mc['test_auc_roc']:.3f}".replace(".", ",")
        lat = f"{mc['latency_10s_ms']:.1f}".replace(".", ",")
        mem = f"{mc['memory_mb']:.1f}".replace(".", ",")

        subs += [
            # Abstract / conclusions references to MultiscaleCNN numbers
            (r"97,78\\% de acur[eé]ncia e AUC-ROC de 0,997",
             f"{acc}\\% de acurácia e AUC-ROC de {auc}"),
            (r"97,78\\% de acur",            f"{acc}\\% de acur"),
            (r"AUC-ROC de 0,997",            f"AUC-ROC de {auc}"),
            (r"EER de 5,56\\%",              f"EER de {eer}\\%"),
            (r"5,56\\% \\(EER\\)",           f"{eer}\\% (EER)"),
            (r"lat[eê]ncia.*?133,7",         f"latência de {lat}"),
            (r"186,8\\\\,MB",               f"{mem}\\,MB"),
        ]

    if ens_conv:
        acc_e = f"{ens['test_accuracy_pct']:.2f}".replace(".", ",")
        eer_e = f"{ens['test_eer_pct']:.2f}".replace(".", ",")
        auc_e = f"{ens['test_auc_roc']:.3f}".replace(".", ",")
        lat_e = f"{ens['latency_10s_ms']:.1f}".replace(".", ",")
        mem_e = f"{ens['memory_mb']:.1f}".replace(".", ",")

        subs += [
            (r"91,11\\%",   f"{acc_e}\\%"),
            (r"13,33\\%",   f"{eer_e}\\%"),
            (r"0,975",      f"{auc_e}"),
            (r"48,1\\\\,ms", f"{lat_e}\\,ms"),
            (r"8,9\\\\,MB",  f"{mem_e}\\,MB"),
        ]

    # Dataset size — atualizar se temos dados de treino
    train_total = archs.get("_meta", {})
    # (pego diretamente do training JSON via chamador)

    for pattern, repl in subs:
        try:
            tex = re.sub(pattern, repl, tex)
        except re.error:
            pass  # padrão inválido — ignorar

    return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Atualiza tabelas do TCC com resultados reais dos JSONs"
    )
    parser.add_argument("--dry-run",  action="store_true",
                        help="Mostra o que seria alterado sem salvar")
    parser.add_argument("--check",    action="store_true",
                        help="Retorna exit code 1 se o .tex diverge dos JSONs")
    parser.add_argument("--tex",      type=str, default=str(TEX_PATH),
                        help=f"Caminho do .tex (default: {TEX_PATH})")
    parser.add_argument("--train-json", type=str, default=str(TRAIN_JSON))
    parser.add_argument("--rob-json",   type=str, default=str(ROB_JSON))
    args = parser.parse_args()

    tex_path   = Path(args.tex)
    train_path = Path(args.train_json)
    rob_path   = Path(args.rob_json)

    # Verificar arquivos
    missing = [p for p in [tex_path, train_path] if not p.exists()]
    if missing:
        print("Arquivos não encontrados:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # Carregar JSONs
    print(f"Carregando {train_path.name}...")
    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    archs = train_data.get("architectures", {})

    rob_results = {}
    if rob_path.exists():
        print(f"Carregando {rob_path.name}...")
        with open(rob_path, encoding="utf-8") as f:
            rob_data = json.load(f)
        rob_results = rob_data.get("results", {})
    else:
        print(f"  {rob_path.name} não encontrado — tabelas de robustez não serão atualizadas")

    # Carregar .tex
    print(f"Carregando {tex_path.name}...")
    tex_original = tex_path.read_text(encoding="utf-8")
    tex = tex_original

    # Resumo do que foi treinado
    print("\nStatus dos modelos:")
    n_conv = 0
    for key in ARCH_ORDER:
        r    = archs.get(key, {})
        conv = is_converged(r)
        if conv:
            n_conv += 1
        status = "OK" if conv else ("PULADO" if r.get("skipped") else ("ERRO" if r.get("error") else "NÃO CONVERGIU"))
        acc = r.get("test_accuracy_pct", "---")
        print(f"  {key:<22} {status:<15} acc={acc}")

    # Atualizar seções marcadas
    print("\nAtualizando tabelas:")
    updates = {
        "tab_resultados":     gen_tab_resultados(archs),
        "tab_complexidade":   gen_tab_complexidade(archs),
        "tab_comparacao_our": gen_tab_comparacao_our(archs),
        "tab_hyperparams":    gen_tab_hyperparams(archs),
        "tab_nao_convergiu":  gen_tab_nao_convergiu(archs),
        "tab_consolidado":    gen_tab_consolidado(archs),
    }
    if rob_results:
        updates["tab_robustez"]         = gen_tab_robustez(rob_results)
        updates["tab_robustez_completo"] = gen_tab_robustez_completo(rob_results)

    for tag, new_content in updates.items():
        tex_new = replace_autogen(tex, tag, new_content)
        changed = tex_new != tex
        print(f"  {tag:<30} {'ATUALIZADO' if changed else 'sem alteração'}")
        tex = tex_new

    # Atualizar números narrativos
    print("\nAtualizando números no texto narrativo...")
    tex = update_narrative(tex, archs, rob_results)

    # Atualizar tamanho do dataset no abstract/intro/conclusões
    n_train = train_data.get("dataset", {}).get("train", 0)
    n_val   = train_data.get("dataset", {}).get("val", 0)
    n_test  = train_data.get("dataset", {}).get("test", 0)
    n_total = n_train + n_val + n_test
    if n_total > 0 and n_total != 600:
        n_total_br  = f"{n_total:,}".replace(",", ".")
        n_train_br  = f"{n_train:,}".replace(",", ".")
        # Substituir referências ao total do dataset (cuidado: apenas o numero isolado)
        tex = re.sub(r"\b600 amostras\b",  f"{n_total_br} amostras", tex)
        tex = re.sub(r"\b600 exemplos\b",  f"{n_total_br} exemplos", tex)
        tex = re.sub(r"\b420 amostras\b",  f"{n_train_br} amostras", tex)
        print(f"  Dataset total: 600 → {n_total_br} amostras")

    # Resultado
    if args.check:
        if tex != tex_original:
            print("\nDivergência detectada: o .tex está desatualizado em relação aos JSONs.")
            sys.exit(1)
        else:
            print("\nOK: o .tex está em sincronia com os JSONs.")
            sys.exit(0)

    if args.dry_run:
        # Mostrar diff simplificado
        orig_lines = tex_original.splitlines()
        new_lines  = tex.splitlines()
        import difflib
        diff = list(difflib.unified_diff(orig_lines, new_lines, lineterm="",
                                         fromfile="original", tofile="atualizado"))
        if diff:
            print("\n--- DIFF (primeiras 80 linhas) ---")
            print("\n".join(diff[:80]))
            if len(diff) > 80:
                print(f"... ({len(diff) - 80} linhas adicionais omitidas)")
        else:
            print("\nNenhuma alteração detectada.")
        return

    # Backup + salvar
    if tex != tex_original:
        bak = tex_path.with_suffix(".tex.bak")
        shutil.copy2(tex_path, bak)
        print(f"\nBackup salvo em: {bak.name}")
        tex_path.write_text(tex, encoding="utf-8")
        print(f"Salvo em: {tex_path}")
    else:
        print("\nNenhuma alteração — arquivo não foi modificado.")

    # Resumo final
    print("\n" + "="*60)
    print(f"RESUMO: {n_conv}/{len(ARCH_ORDER)} arquiteturas convergiram")
    conv_list  = [k for k in ARCH_ORDER if is_converged(archs.get(k, {}))]
    noconv_list = [k for k in ARCH_ORDER if not is_converged(archs.get(k, {}))]
    if conv_list:
        print("  Convergidos: " + ", ".join(conv_list))
    if noconv_list:
        print("  Não convergidos: " + ", ".join(noconv_list))
    print(f"\nDataset: {n_total} amostras "
          f"(train={n_train} | val={n_val} | test={n_test})")
    print("="*60)


if __name__ == "__main__":
    main()
