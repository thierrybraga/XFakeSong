"""Comparação PAREADA entre dois modelos no mesmo conjunto de teste.

MOTIVAÇÃO 2026-08-09: o benchmark reporta IC 95% por modelo (bootstrap de
clusters em ``benchmarks/evaluate.py``), mas IC individual não decide
diferença. No ``clean_benchmark_15k`` o Conformer (EER 0,43% [0,14; 1,00]) e o
Hybrid CNN-Transformer (0,43% [0,00; 0,74]) têm intervalos que se sobrepõem
quase por inteiro — e a regra "ICs se sobrepõem, logo não há diferença" é
falsa para amostras pareadas: os dois modelos veem EXATAMENTE as mesmas 1.382
amostras, então o que importa é a distribuição da DIFERENÇA, não a de cada um.

Dois testes, porque medem coisas diferentes:

``mcnemar``
    Sobre as decisões duras no limiar fixo do protocolo. Responde "estes dois
    modelos erram nas mesmas amostras?". Usa a forma exata (binomial), válida
    para qualquer contagem de discordâncias — a aproximação qui-quadrado é
    ruim justamente quando o total discordante é pequeno, que é o caso entre
    os modelos do topo.

``paired_bootstrap``
    Sobre a diferença de EER/AUC, reamostrando os MESMOS índices para os dois
    modelos. Preserva a correlação entre eles, então o IC da diferença é bem
    mais estreito que a distância entre os ICs individuais.

Ambos reamostram/agrupam por CLUSTER quando os IDs estão disponíveis (a mesma
unidade dos IC por modelo — frase/locutor). Sem eles, caem para amostra e
declaram isso em ``unit``: amostras da mesma frase não são independentes, e
tratar como se fossem produz p-valor otimista.

Sem correção de multiplicidade embutida: use ``holm_adjust`` sobre o conjunto
de comparações que você de fato vai reportar.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _p_floor(n_bootstrap: int) -> float:
    """Menor p-valor que ``n_bootstrap`` reamostragens conseguem expressar.

    O p bilateral é ``2 * (k + 1) / (n + 1)``; com ``k = 0`` o mínimo é
    ``2 / (n + 1)``. Reportar esse piso importa porque ele se propaga: com 55
    comparações e 1.000 reamostragens, Holm multiplica 0,002 por 55 e nenhum
    par consegue ficar abaixo de 0,05 — nem os cujo IC da diferença exclui zero
    com folga. Sem o piso declarado, "não significativo" se confunde com
    "resolução insuficiente".
    """
    return 2.0 / (int(n_bootstrap) + 1)


def _round_p(value: float) -> float:
    """Arredonda p-valor por ALGARISMOS significativos, não casas decimais.

    ``round(2.2e-11, 6)`` devolve ``0.0``: um p pequeno o bastante desaparece, e
    o ajustado de Holm chega a sair MENOR que o bruto — o oposto do que a
    correção faz. Seis significativos preservam a ordem de grandeza e mantêm o
    JSON legível.
    """
    return float(f"{float(value):.6g}")


def _binom_two_sided_p(k: int, n: int) -> float:
    """p-valor exato bilateral de ``Binomial(n, 0.5)`` observando ``k``.

    Soma as caudas por probabilidade (método de Sterne simplificado: todos os
    resultados no mínimo tão improváveis quanto o observado). Para p=0.5 a
    distribuição é simétrica, então equivale a dobrar a cauda menor — com o
    cuidado de nunca ultrapassar 1.
    """
    if n <= 0:
        return 1.0
    k = min(int(k), int(n) - int(k)) if n else 0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def mcnemar_test(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    *,
    cluster_ids: Optional[Sequence[Any]] = None,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> Dict[str, Any]:
    """McNemar sobre acertos/erros de dois modelos nas mesmas amostras.

    ``only_a_correct`` = amostras que A acerta e B erra; ``only_b_correct`` = o
    inverso. Concordâncias não entram — é o ponto do teste.

    Sem ``cluster_ids``: binomial exata sobre as discordâncias.

    Com ``cluster_ids``: as contagens continuam por AMOSTRA (é o que a
    estatística de McNemar mede), mas o p-valor vem de um **bootstrap de
    clusters** da diferença ``only_a - only_b``. Amostras da mesma frase não
    são independentes, e a binomial exata sobre elas dá p otimista.

    CORREÇÃO 2026-08-09: a primeira versão agregava por MAIORIA dentro do
    cluster antes de contar. Com ~7,5 amostras por frase e acurácia alta, a
    maioria quase nunca vira — comparando HuBERT sob duas janelas, cuja EER
    difere em 3,9 pp, a agregação zerava as 183 discordâncias e devolvia
    ``p = 1``, enquanto o bootstrap pareado nos scores acusava p < 0,001. Um
    teste que não distingue "sem diferença" de "sem poder" é pior que nenhum.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    a = (np.asarray(pred_a).ravel().astype(int) == y_true).astype(int)
    b = (np.asarray(pred_b).ravel().astype(int) == y_true).astype(int)
    if len(a) != len(y_true) or len(b) != len(y_true):
        raise ValueError("predições desalinhadas com y_true")

    a_only = (a == 1) & (b == 0)
    b_only = (a == 0) & (b == 1)
    only_a = int(np.sum(a_only))
    only_b = int(np.sum(b_only))
    discordant = only_a + only_b

    out: Dict[str, Any] = {
        "test": "mcnemar_exact",
        "unit": "sample",
        "n_units": int(len(a)),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "discordant": discordant,
        "p_value": _round_p(_binom_two_sided_p(only_a, discordant)),
    }
    if cluster_ids is None:
        return out

    clusters = np.asarray(cluster_ids).astype(str).ravel()
    if len(clusters) != len(y_true):
        raise ValueError("cluster_ids desalinhado com y_true")
    groups = np.unique(clusters)
    if len(groups) < 2:
        out["cluster_warning"] = "menos de 2 clusters: p-valor segue por amostra"
        return out

    index_by_group = [np.flatnonzero(clusters == g) for g in groups]
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_bootstrap), dtype="float64")
    for i in range(int(n_bootstrap)):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([index_by_group[j] for j in chosen])
        diffs[i] = float(np.sum(a_only[idx]) - np.sum(b_only[idx]))

    tail = min(
        (np.sum(diffs <= 0) + 1) / (len(diffs) + 1),
        (np.sum(diffs >= 0) + 1) / (len(diffs) + 1),
    )
    p_value = min(1.0, 2.0 * tail)
    floor = _p_floor(len(diffs))
    out.update(
        test="mcnemar_cluster_bootstrap",
        unit="cluster",
        n_units=int(len(groups)),
        n_samples=int(len(a)),
        p_value=_round_p(p_value),
        p_value_sample_exact=_round_p(_binom_two_sided_p(only_a, discordant)),
        bootstrap_samples=int(len(diffs)),
        p_value_floor=_round_p(floor),
        p_value_at_floor=bool(p_value <= floor + 1e-12),
    )
    return out


def _eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """EER pelo cruzamento de FPR e FNR na curva ROC."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def paired_bootstrap_test(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    metric: str = "eer",
    cluster_ids: Optional[Sequence[Any]] = None,
    n_bootstrap: int = 1000,
    seed: int = 12345,
) -> Dict[str, Any]:
    """IC 95% e p-valor da DIFERENÇA (A − B) de EER ou AUC.

    A cada reamostragem os MESMOS índices alimentam os dois modelos — é isso
    que torna o teste pareado e o IC da diferença estreito. O p-valor é o
    percentil bilateral da fração de reamostragens que cruzam o zero.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel().astype(int)
    sa = np.asarray(scores_a, dtype="float64").ravel()
    sb = np.asarray(scores_b, dtype="float64").ravel()
    if len(sa) != len(y_true) or len(sb) != len(y_true):
        raise ValueError("scores desalinhados com y_true")
    if metric not in {"eer", "auc_roc"}:
        raise ValueError(f"métrica não suportada: {metric!r}")

    def compute(y: np.ndarray, s: np.ndarray) -> float:
        return _eer(y, s) if metric == "eer" else float(roc_auc_score(y, s))

    observed = compute(y_true, sa) - compute(y_true, sb)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    clusters = None
    groups: Optional[np.ndarray] = None
    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids).astype(str).ravel()
        if len(clusters) != n:
            raise ValueError("cluster_ids desalinhado com y_true")
        groups = np.unique(clusters)
        if len(groups) < 2:
            clusters, groups = None, None

    diffs: List[float] = []
    for _ in range(int(n_bootstrap)):
        if clusters is None:
            idx = rng.integers(0, n, n)
        else:
            chosen = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([np.flatnonzero(clusters == g) for g in chosen])
        yb = y_true[idx]
        if yb.min() == yb.max():  # reamostra sem as duas classes: descarta
            continue
        try:
            diffs.append(compute(yb, sa[idx]) - compute(yb, sb[idx]))
        except Exception:  # noqa: BLE001 — reamostra degenerada
            continue

    out: Dict[str, Any] = {
        "test": "paired_bootstrap",
        "metric": metric,
        "unit": "cluster" if clusters is not None else "sample",
        "observed_difference": round(float(observed), 6),
        "bootstrap_samples": len(diffs),
    }
    if clusters is not None and groups is not None:
        out["n_clusters"] = int(len(groups))
    if not diffs:
        out["status"] = "indeterminado: nenhuma reamostragem válida"
        return out
    values = np.asarray(diffs, dtype="float64")
    lo, hi = np.percentile(values, [2.5, 97.5])
    # p bilateral: 2x a menor cauda em torno de zero, limitado a 1. O +1 no
    # numerador e no denominador evita p=0 exato, que o número de reamostragens
    # não sustenta.
    n_b = len(values)
    tail = min(
        (np.sum(values <= 0) + 1) / (n_b + 1),
        (np.sum(values >= 0) + 1) / (n_b + 1),
    )
    p_value = min(1.0, 2.0 * tail)
    floor = _p_floor(n_b)
    out.update(
        difference_ci95_low=round(float(lo), 6),
        difference_ci95_high=round(float(hi), 6),
        p_value=_round_p(p_value),
        p_value_floor=_round_p(floor),
        p_value_at_floor=bool(p_value <= floor + 1e-12),
        # O IC da diferença NÃO tem o piso do p-valor: ele continua informativo
        # quando o p satura. Se os dois discordarem, é a resolução que faltou.
        significant_at_95=bool(lo > 0 or hi < 0),
    )
    return out


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Correção de Holm-Bonferroni, preservando a ordem de entrada.

    Comparar 11 modelos par a par são 55 testes; sem ajuste, ~3 saem
    "significativos" a 5% só por acaso. Holm é uniformemente mais poderoso que
    Bonferroni e não exige independência entre os testes — o que importa aqui,
    já que as comparações compartilham o mesmo conjunto de teste.
    """
    values = [float(p) for p in p_values]
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * values[idx]
        running = max(running, min(1.0, candidate))
        adjusted[idx] = running
    return adjusted


def compare_models(
    y_true: Sequence[int],
    models: Dict[str, Dict[str, Any]],
    *,
    threshold: float = 0.5,
    cluster_ids: Optional[Sequence[Any]] = None,
    n_bootstrap: int = 5000,
    metric: str = "eer",
    seed: int = 12345,
) -> Dict[str, Any]:
    """Matriz de comparações par a par entre todos os modelos informados.

    ``models`` mapeia nome -> ``{"scores": [...]}``; as decisões duras saem de
    ``scores >= threshold``, o mesmo limiar fixo do protocolo. O resultado traz
    os p-valores brutos e os ajustados por Holm sobre o conjunto TODO de pares.
    """
    names = list(models)
    y = np.asarray(y_true).ravel().astype(int)
    pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa = np.asarray(models[a]["scores"], dtype="float64").ravel()
            sb = np.asarray(models[b]["scores"], dtype="float64").ravel()
            entry: Dict[str, Any] = {"model_a": a, "model_b": b}
            try:
                entry["mcnemar"] = mcnemar_test(
                    y,
                    (sa >= threshold).astype(int),
                    (sb >= threshold).astype(int),
                    cluster_ids=cluster_ids,
                    # O MESMO n do bootstrap pareado: sem repassar, o McNemar
                    # ficava no default (2.000) e seu piso 2/2001 x 55 = 0,055
                    # travava todo p ajustado logo acima de 0,05, enquanto o
                    # bootstrap pareado já resolvia. Os dois testes precisam ter
                    # a mesma resolução para que discordância entre eles
                    # signifique algo sobre os dados.
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                entry["paired_bootstrap"] = paired_bootstrap_test(
                    y,
                    sa,
                    sb,
                    metric=metric,
                    cluster_ids=cluster_ids,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 — um par ruim não derruba
                entry["status"] = "error"
                entry["error"] = str(exc)
            pairs.append(entry)

    for key in ("mcnemar", "paired_bootstrap"):
        indexed = [
            (p, p[key]["p_value"])
            for p in pairs
            if isinstance(p.get(key), dict) and p[key].get("p_value") is not None
        ]
        for (pair, _), adj in zip(indexed, holm_adjust([v for _, v in indexed])):
            pair[key]["p_value_holm"] = _round_p(adj)

    protocol: Dict[str, Any] = {
        "decision_threshold": threshold,
        "metric": metric,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "unit": "cluster" if cluster_ids is not None else "sample",
        "multiplicity_correction": "holm",
        "n_comparisons": len(pairs),
    }
    # Holm multiplica o p pelo número de comparações, então o piso do bootstrap
    # também é multiplicado. Se o produto passar de 0,05, NENHUM par consegue
    # ficar significativo — nem os que o IC da diferença separa com folga — e
    # "indistinguíveis" viraria uma conclusão da resolução, não dos dados.
    floor = _p_floor(int(n_bootstrap))
    piso_holm = floor * max(1, len(pairs))
    protocol["p_value_floor"] = _round_p(floor)
    protocol["min_resolvable_holm_p"] = _round_p(piso_holm)
    if piso_holm >= 0.05 and pairs:
        needed = int(math.ceil(2.0 * len(pairs) / 0.01)) + 1
        protocol["warning"] = (
            f"resolução insuficiente: com {int(n_bootstrap)} reamostragens e "
            f"{len(pairs)} comparações, o menor p ajustado por Holm é "
            f"{piso_holm:.3g} — acima de 0,05. Nenhum par pode sair "
            f"significativo por p, INDEPENDENTE dos dados; use o IC da "
            f"diferença ou repita com n_bootstrap >= {needed}."
        )
    return {
        "protocol": protocol,
        "models": names,
        "pairs": pairs,
    }
