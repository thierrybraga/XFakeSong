#!/usr/bin/env python3
"""Auditoria forense de vazamento de domínio no dataset do benchmark (P0.0).

Testa a hipótese central do diagnóstico: as acurácias ~100% (Sonic Sleuth, etc.)
vêm de um ATALHO de domínio/gerador, não de detecção genuína de falsificação.

Duas evidências são produzidas:

1. **Atalho intra-fonte (brspeech).** Treina uma regressão logística sobre ~10
   features GLOBAIS e baratas (RMS, offset DC, ZCR, silêncio nas bordas,
   centróide/largura espectral, energia em alta frequência, fator de crista)
   restritas ao brspeech — a única fonte com real E fake. Se features que NÃO
   capturam artefato fino de spoofing separam real/fake com acurácia alta, o
   sinal discriminante é um artefato sistemático (vocoder/pré-processamento),
   não a falsificação em si. Reporta acurácia/AUC combinada e por feature.

2. **Atalho por fonte pura.** Quantifica a fração do conjunto de teste que é
   trivialmente separável só pela identidade da fonte (cvpt=100% real,
   fkvoice=100% fake aparecem em treino E teste no split estratificado).

Sem GPU. Usa o MESMO X pré-processado que os modelos consomem (audita o que o
benchmark realmente viu, incluindo qualquer artefato de pré-processamento).

Exemplo:
    python scripts/audit_dataset_leakage.py \
        --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
        --source brspeech --max-per-class 1500 \
        --out results/audit_dataset_leakage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SR = 16000
EPS = 1e-9

FEATURE_NAMES = [
    "rms",
    "dc_offset",
    "abs_mean",
    "std",
    "max_abs",
    "crest_factor",
    "zcr",
    "lead_silence_ratio",
    "trail_silence_ratio",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "hf_energy_ratio_4k",
]


def _derive_source(path: str) -> str:
    base = str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()
    m = re.match(r"([a-z]+)", base)
    return m.group(1) if m else "unknown"


def _label_from_path(path: str) -> int | None:
    p = str(path).replace("\\", "/").lower()
    if "/fake/" in p:
        return 1
    if "/real/" in p:
        return 0
    return None


def _extract_features_chunk(x: np.ndarray) -> np.ndarray:
    """Features globais por amostra. x: (B, T) float32 -> (B, n_feats)."""
    x = np.asarray(x, dtype="float32")
    if x.ndim == 3:  # (B, T, 1)
        x = x[..., 0]
    B, T = x.shape

    rms = np.sqrt(np.mean(x ** 2, axis=1) + EPS)
    dc = np.mean(x, axis=1)
    abs_mean = np.mean(np.abs(x), axis=1)
    std = np.std(x, axis=1)
    max_abs = np.max(np.abs(x), axis=1)
    crest = max_abs / (rms + EPS)

    sign = np.signbit(x)
    zcr = np.mean(sign[:, 1:] != sign[:, :-1], axis=1)

    # Silêncio nas bordas (limiar relativo ao pico de cada amostra).
    thr = (0.02 * max_abs).reshape(-1, 1) + EPS
    active = np.abs(x) > thr
    lead = np.argmax(active, axis=1) / float(T)
    trail = np.argmax(active[:, ::-1], axis=1) / float(T)
    # Amostras totalmente silenciosas (nenhum ativo) -> borda = 1.0
    no_active = ~active.any(axis=1)
    lead[no_active] = 1.0
    trail[no_active] = 1.0

    # Espectro (rfft sobre o batch).
    mag = np.abs(np.fft.rfft(x, axis=1)).astype("float32")
    freqs = np.fft.rfftfreq(T, d=1.0 / SR).astype("float32")  # (F,)
    mag_sum = np.sum(mag, axis=1) + EPS
    centroid = np.sum(mag * freqs[None, :], axis=1) / mag_sum
    bandwidth = np.sqrt(
        np.sum(mag * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / mag_sum
    )
    hf_mask = freqs >= 4000.0
    hf_ratio = np.sum(mag[:, hf_mask], axis=1) / mag_sum

    feats = np.stack(
        [rms, dc, abs_mean, std, max_abs, crest, zcr, lead, trail,
         centroid, bandwidth, hf_ratio],
        axis=1,
    ).astype("float32")
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_features(X: np.ndarray, idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Extrai features das linhas `idx` de `X` (mmap) em chunks, baixa memória."""
    out: List[np.ndarray] = []
    idx = np.asarray(idx)
    for start in range(0, len(idx), chunk):
        rows = idx[start:start + chunk]
        # mmap fancy-index materializa só o chunk.
        batch = np.asarray(X[rows])
        out.append(_extract_features_chunk(batch))
    return np.concatenate(out, axis=0) if out else np.empty((0, len(FEATURE_NAMES)))


def _eer(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def _balanced_subsample(
    idx: np.ndarray, y: np.ndarray, max_per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: List[np.ndarray] = []
    for c in (0, 1):
        ci = idx[y[idx] == c]
        if max_per_class and len(ci) > max_per_class:
            ci = rng.choice(ci, size=max_per_class, replace=False)
        keep.append(ci)
    out = np.concatenate(keep)
    return np.sort(out)


def _load_split_meta(meta: dict) -> Dict[str, dict]:
    return (meta or {}).get("splits", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help=".npz com X_train/.../metadata_json")
    ap.add_argument("--source", default="brspeech",
                    help="fonte a auditar (única com real+fake). Default: brspeech")
    ap.add_argument("--max-per-class", type=int, default=1500,
                    help="subamostra por classe (memória/velocidade). 0 = tudo")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/audit_dataset_leakage",
                    help="pasta de saída do relatório (json + md)")
    args = ap.parse_args()

    p = Path(args.dataset)
    if not p.exists():
        ap.error(f"dataset não encontrado: {p}")

    print(f"[audit] carregando {p} (mmap)…")
    d = np.load(p, mmap_mode="r", allow_pickle=False)
    meta = {}
    if "metadata_json" in d.files:
        raw = d["metadata_json"]
        raw = raw.item() if hasattr(raw, "item") else raw
        meta = json.loads(str(raw))
    splits_meta = _load_split_meta(meta)

    # Mapeia cada split -> (X, y, sources, labels_from_path)
    split_keys = [("train", "X_train", "y_train"),
                  ("val", "X_val", "y_val"),
                  ("test", "X_test", "y_test")]

    per_split = {}
    align_warnings = []
    for name, xk, yk in split_keys:
        if xk not in d.files:
            continue
        y = np.asarray(d[yk]).ravel().astype("int64")
        paths = (splits_meta.get(name, {}) or {}).get("paths", [])
        if len(paths) != len(y):
            align_warnings.append(
                f"{name}: n_paths={len(paths)} != n_y={len(y)} (sem proveniência confiável)"
            )
            sources = np.array(["unknown"] * len(y), dtype=object)
        else:
            sources = np.array([_derive_source(pp) for pp in paths], dtype=object)
            # Sanidade: rótulo do path deve bater com y.
            lab_path = np.array([_label_from_path(pp) for pp in paths])
            valid = np.array([v is not None for v in lab_path])
            if valid.any():
                mism = np.mean(lab_path[valid].astype(int) != y[valid])
                if mism > 0.01:
                    align_warnings.append(
                        f"{name}: {mism:.1%} dos rótulos do path divergem de y "
                        "(possível desalinhamento X↔paths)"
                    )
        per_split[name] = {"X": d[xk], "y": y, "src": sources}

    # ---- Matriz fonte x classe (todos os splits) ----
    src_class: Dict[Tuple[str, int], int] = {}
    for name, s in per_split.items():
        for src, lab in zip(s["src"], s["y"]):
            src_class[(str(src), int(lab))] = src_class.get((str(src), int(lab)), 0) + 1
    all_sources = sorted({k[0] for k in src_class})

    # ---- Fração trivialmente separável por fonte no TESTE ----
    test = per_split.get("test")
    trivial_frac = None
    trivial_detail = {}
    if test is not None:
        ts_src = test["src"].astype(str)
        ts_y = test["y"]
        n_test = len(ts_y)
        pure_real_src = [s for s in all_sources
                         if src_class.get((s, 1), 0) == 0 and src_class.get((s, 0), 0) > 0]
        pure_fake_src = [s for s in all_sources
                         if src_class.get((s, 0), 0) == 0 and src_class.get((s, 1), 0) > 0]
        trivial = np.isin(ts_src, pure_real_src) | np.isin(ts_src, pure_fake_src)
        trivial_frac = float(np.mean(trivial)) if n_test else 0.0
        trivial_detail = {
            "n_test": int(n_test),
            "pure_real_sources": pure_real_src,
            "pure_fake_sources": pure_fake_src,
            "n_trivial": int(np.sum(trivial)),
            "trivial_fraction": trivial_frac,
        }

    # ---- Atalho intra-fonte: fit no train, avalia no test (mesma fonte) ----
    src = args.source.lower()
    tr = per_split.get("train")
    te = per_split.get("test")
    intra = {"source": src, "status": "skipped"}
    if tr is not None and te is not None:
        tr_idx = np.where(tr["src"].astype(str) == src)[0]
        te_idx = np.where(te["src"].astype(str) == src)[0]
        # precisa das duas classes em ambos
        ok = (len(np.unique(tr["y"][tr_idx])) == 2
              and len(np.unique(te["y"][te_idx])) == 2)
        if not ok or len(tr_idx) == 0 or len(te_idx) == 0:
            intra = {"source": src, "status": "insufficient",
                     "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}
        else:
            tr_sel = _balanced_subsample(tr_idx, tr["y"], args.max_per_class, args.seed)
            te_sel = _balanced_subsample(te_idx, te["y"], args.max_per_class, args.seed)
            print(f"[audit] extraindo features — train={len(tr_sel)} test={len(te_sel)} "
                  f"(fonte={src})…")
            Xtr = _extract_features(tr["X"], tr_sel, args.chunk)
            Xte = _extract_features(te["X"], te_sel, args.chunk)
            ytr, yte = tr["y"][tr_sel], te["y"][te_sel]

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            proba = clf.predict_proba(Xte_s)[:, 1]
            acc = float(accuracy_score(yte, proba >= 0.5))
            auc = float(roc_auc_score(yte, proba))
            eer = _eer(yte, proba)

            # Univariado: AUC/acc de cada feature isolada.
            per_feat = []
            for j, fname in enumerate(FEATURE_NAMES):
                cj = LogisticRegression(max_iter=1000, class_weight="balanced")
                cj.fit(Xtr_s[:, [j]], ytr)
                pj = cj.predict_proba(Xte_s[:, [j]])[:, 1]
                a = float(accuracy_score(yte, pj >= 0.5))
                u = float(roc_auc_score(yte, pj))
                per_feat.append({"feature": fname, "test_accuracy": a, "test_auc": u})
            per_feat.sort(key=lambda r: r["test_auc"], reverse=True)

            intra = {
                "source": src,
                "status": "ok",
                "n_train": int(len(tr_sel)),
                "n_test": int(len(te_sel)),
                "combined": {"test_accuracy": acc, "test_auc": auc, "test_eer": eer},
                "per_feature": per_feat,
            }

    report = {
        "dataset": str(p),
        "source_class_matrix": {
            s: {"real": src_class.get((s, 0), 0), "fake": src_class.get((s, 1), 0)}
            for s in all_sources
        },
        "test_source_shortcut": trivial_detail,
        "intra_source_shortcut": intra,
        "alignment_warnings": align_warnings,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leakage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(out_dir / "leakage_report.md", report)

    # ---- Resumo no terminal ----
    print("\n" + "=" * 64)
    print("MATRIZ FONTE x CLASSE")
    for s in all_sources:
        r = src_class.get((s, 0), 0); f = src_class.get((s, 1), 0)
        tag = "  <- pura" if (r == 0) ^ (f == 0) else ""
        print(f"  {s:10s} real={r:6d} fake={f:6d}{tag}")
    if trivial_detail:
        print(f"\nTESTE trivialmente separável por fonte: "
              f"{trivial_detail['n_trivial']}/{trivial_detail['n_test']} "
              f"({trivial_detail['trivial_fraction']*100:.1f}%)")
        print(f"  fontes puras real={trivial_detail['pure_real_sources']} "
              f"fake={trivial_detail['pure_fake_sources']}")
    if intra.get("status") == "ok":
        c = intra["combined"]
        print(f"\nATALHO INTRA-FONTE ({src}) — features globais triviais:")
        print(f"  acurácia teste = {c['test_accuracy']*100:.2f}%  "
              f"AUC = {c['test_auc']:.4f}  EER = {c['test_eer']*100:.2f}%")
        verdict = ("ATALHO CONFIRMADO" if c["test_accuracy"] >= 0.90
                   else "atalho parcial" if c["test_accuracy"] >= 0.75
                   else "sem atalho global óbvio")
        print(f"  veredito: {verdict}")
        print("  top features (AUC):")
        for r in intra["per_feature"][:5]:
            print(f"    {r['feature']:22s} acc={r['test_accuracy']*100:5.1f}% "
                  f"auc={r['test_auc']:.3f}")
    else:
        print(f"\nATALHO INTRA-FONTE: {intra.get('status')} ({intra})")
    if align_warnings:
        print("\nAVISOS:")
        for w in align_warnings:
            print(f"  ! {w}")
    print("=" * 64)
    print(f"\nRelatório: {out_dir / 'leakage_report.json'}")
    print(f"           {out_dir / 'leakage_report.md'}\n")
    return 0


def _write_markdown(path: Path, rep: dict) -> None:
    lines = ["# Auditoria de vazamento — dataset do benchmark", ""]
    lines.append(f"Dataset: `{rep['dataset']}`\n")
    lines.append("## Matriz fonte × classe\n")
    lines.append("| Fonte | real | fake | observação |")
    lines.append("|---|---:|---:|---|")
    for s, v in rep["source_class_matrix"].items():
        pure = "fonte pura (atalho)" if (v["real"] == 0) ^ (v["fake"] == 0) else ""
        lines.append(f"| {s} | {v['real']} | {v['fake']} | {pure} |")
    ts = rep.get("test_source_shortcut") or {}
    if ts:
        lines.append("\n## Atalho por fonte pura (teste)\n")
        lines.append(f"- Trivialmente separável só pela fonte: "
                     f"**{ts['n_trivial']}/{ts['n_test']} "
                     f"({ts['trivial_fraction']*100:.1f}%)**")
        lines.append(f"- Fontes puras real: `{ts['pure_real_sources']}` · "
                     f"fake: `{ts['pure_fake_sources']}`")
    intra = rep.get("intra_source_shortcut") or {}
    if intra.get("status") == "ok":
        c = intra["combined"]
        lines.append(f"\n## Atalho intra-fonte ({intra['source']})\n")
        lines.append(f"Regressão logística sobre features globais triviais "
                     f"(n_train={intra['n_train']}, n_test={intra['n_test']}):\n")
        lines.append(f"- **Acurácia teste: {c['test_accuracy']*100:.2f}%** · "
                     f"AUC {c['test_auc']:.4f} · EER {c['test_eer']*100:.2f}%")
        lines.append("\n| Feature | acc teste | AUC |")
        lines.append("|---|---:|---:|")
        for r in intra["per_feature"]:
            lines.append(f"| {r['feature']} | {r['test_accuracy']*100:.1f}% | "
                         f"{r['test_auc']:.3f} |")
    if rep.get("alignment_warnings"):
        lines.append("\n## Avisos\n")
        for w in rep["alignment_warnings"]:
            lines.append(f"- ⚠ {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
