#!/usr/bin/env python3
"""Recupera o histórico de épocas truncado por retomada, a partir do ``run.log``.

MOTIVAÇÃO. ``model.fit()`` devolve apenas as épocas da execução CORRENTE. Com
``BackupAndRestore``, uma retomada na época 84 produz um histórico de 17
entradas para um treino de 100 — foi o que aconteceu no ``clean_benchmark_15k``
com RawNet2 (17/100) e RawGAT-ST (91/100). Os dois treinaram as 100 épocas: está
nos ``run.log``, linha a linha, com as quatro séries.

O callback ``PersistentEpochHistory`` (``trainer.py``) fecha isso para execuções
NOVAS. Este script trata os artefatos que já existem, e o faz porque a
truncagem não é cosmética:

- ``update_tcc_latex.py`` RECUSA gerar as tabelas do artigo enquanto
  ``epochs`` divergir do orçamento declarado — e está certo em recusar;
- ``training_stability`` é calculado sobre a série gravada, então o veredito
  desses dois descreve um fragmento. "RawNet2: stable" saiu de 17 épocas.

NÃO É FABRICAÇÃO, e a diferença importa. O ``backfill_artifact_metadata.py``
DERIVA campos do que já está no artefato e por isso recusou este caso ("parsing
de log, não derivação"). Aqui a fonte é o log da MESMA execução, no MESMO
diretório, e a recuperação só é aceita depois de provar que log e artefato
descrevem o mesmo treino: **a cauda da série reconstruída tem de bater, valor a
valor, com o histórico que o artefato gravou**. Divergiu, aborta.

Simulação por padrão; ``--write`` aplica e guarda ``<arquivo>.pre-recovery.bak``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _analisar_estabilidade(historico: Dict[str, List[float]], orcamento: int):
    """Importa a análise de estabilidade SÓ quando há histórico a recalcular.

    `benchmarks.stability` puxa o pacote `benchmarks`, que importa o runner e
    com ele numpy/TensorFlow. Os reparos de metadado deste script (realinhar
    `epochs_executed`, corrigir a tolerância declarada) não recalculam nada e
    precisam rodar onde só existe a biblioteca padrão — em um checkout sem o
    ambiente de treino instalado, que é onde artefatos costumam ser auditados.
    """
    from benchmarks.stability import analyze_training_stability

    return analyze_training_stability(historico, epochs_budget=orcamento)

#: Uma linha de fim de época do `ModelTrainer`. `val_loss`/`val_accuracy` vêm
#: DEPOIS de `loss`/`accuracy`, e o `(?<![_a-z])` impede que ` loss=` case com
#: o sufixo de `val_loss=`.
_EPOCH_LINE = re.compile(
    r"epoch=(?P<epoch>\d+)/(?P<total>\d+)\s+epoch_s=[\d.]+"
    r".*?(?<![_a-z])loss=(?P<loss>[-\d.eE+naN]+)"
    r"\s+accuracy=(?P<acc>[-\d.eE+naN]+)"
    r"\s+val_loss=(?P<val_loss>[-\d.eE+naN]+)"
    r"\s+val_accuracy=(?P<val_acc>[-\d.eE+naN]+)"
)

_SERIES = ("loss", "accuracy", "val_loss", "val_accuracy")


def parse_run_log(path: Path) -> Dict[str, List[float]]:
    """Reconstrói as quatro séries indexadas pela época ABSOLUTA.

    Uma época repetida no log (a retomada reexecuta a época em que caiu) fica
    com a ÚLTIMA ocorrência — é a que produziu o estado que seguiu adiante.
    """
    por_epoca: Dict[int, Dict[str, float]] = {}
    total: Optional[int] = None
    for linha in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _EPOCH_LINE.search(linha)
        if not m:
            continue
        total = int(m.group("total"))
        por_epoca[int(m.group("epoch"))] = {
            "loss": float(m.group("loss")),
            "accuracy": float(m.group("acc")),
            "val_loss": float(m.group("val_loss")),
            "val_accuracy": float(m.group("val_acc")),
        }
    if not por_epoca:
        raise ValueError(f"nenhuma linha de época reconhecida em {path}")

    ultima = max(por_epoca)
    faltando = [e for e in range(1, ultima + 1) if e not in por_epoca]
    if faltando:
        raise ValueError(
            f"buracos na série reconstruída de {path}: épocas {faltando[:10]}"
            f"{'...' if len(faltando) > 10 else ''}"
        )
    return {
        "_total": total,
        **{k: [por_epoca[e][k] for e in range(1, ultima + 1)] for k in _SERIES},
    }


#: Tolerância do casamento de cauda.
#:
#: O `run.log` imprime com ~6 algarismos significativos (`1.17211`), enquanto o
#: `results.json` guarda a precisão do float (`1.172114`) — uma diferença
#: relativa de ~3e-6 que uma tolerância de 1e-6 rejeita. 1e-4 acomoda o
#: arredondamento do log e continua ordens de grandeza mais apertada do que
#: qualquer diferença entre DUAS EXECUÇÕES distintas, que divergem já no
#: primeiro algarismo significativo — que é o que este teste existe para pegar.
_TOL_CAUDA = 1e-4


def _divergencia_da_cauda(
    recuperado: Dict[str, List[float]], gravado: Dict[str, Any]
) -> Optional[str]:
    """``None`` se a série gravada é o sufixo da reconstruída; senão, o motivo.

    É a prova de que o log e o artefato descrevem o mesmo treino. Sem ela, um
    log de outra execução no mesmo diretório passaria despercebido.
    """
    n = len(gravado.get("val_loss") or [])
    if not n:
        return "histórico gravado vazio"
    for chave in _SERIES:
        atual = [float(v) for v in (gravado.get(chave) or [])]
        if len(atual) != n:
            return f"série '{chave}' gravada tem {len(atual)} pontos, esperado {n}"
        cauda = recuperado[chave][-n:]
        if len(cauda) != n:
            return f"série '{chave}' do log é curta demais para a cauda de {n}"
        for i, (a, b) in enumerate(zip(atual, cauda)):
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isclose(a, b, rel_tol=_TOL_CAUDA, abs_tol=1e-9):
                epoca = len(recuperado[chave]) - n + i + 1
                return (
                    f"'{chave}' diverge na época {epoca}: artefato={a!r} "
                    f"log={b!r}"
                )
    return None


def _reconciliar_epochs_executed(arch: Dict[str, Any], epocas: int) -> bool:
    """Alinha ``training_config.epochs_executed`` com a série de fato gravada.

    O campo é escrito pelo runner com o tamanho do histórico que o ``fit()``
    daquela EXECUÇÃO devolveu. Numa retomada esse número é o do fragmento (17
    para o RawNet2, 91 para o RawGAT-ST), e recuperar o histórico sem tocá-lo
    deixa o artefato afirmando duas coisas incompatíveis: ``epochs=100`` ao
    lado de ``epochs_executed=17``. Quem lê o segundo conclui que o orçamento
    não foi cumprido — exatamente o oposto do que o log mostra.
    """
    tc = arch.get("training_config")
    if not isinstance(tc, dict):
        return False
    # Só CORRIGE o que já existe. O runner SSL nunca escreve esta chave — a
    # ausência dela lá é o formato daquele runner, não uma lacuna, e `epochs`
    # já declara o orçamento cumprido. Criá-la aqui inventaria um campo que
    # nenhuma execução produziu.
    if "epochs_executed" not in tc or tc["epochs_executed"] == epocas:
        return False
    tc["epochs_executed"] = epocas
    return True


def _corrigir_tolerancia_declarada(arch: Dict[str, Any]) -> bool:
    """Realinha a tolerância CITADA no artefato com a que o teste usa.

    As primeiras recuperações gravaram "rel_tol=1e-6" numa época em que a
    constante já era 1e-4. O número é a prova de que log e artefato batem —
    publicá-lo errado enfraquece justamente o que ele existe para sustentar.
    """
    rec = arch.get("history_recovery")
    if not isinstance(rec, dict) or "verification" not in rec:
        return False
    correta = (
        f"a série gravada é sufixo exato da reconstruída (rel_tol={_TOL_CAUDA:g})"
    )
    if rec["verification"] == correta:
        return False
    rec["verification"] = correta
    return True


def recuperar(run_dir: Path, *, write: bool) -> int:
    tocados = 0
    for results_json in sorted(run_dir.glob("*/results.json")):
        dados = json.loads(results_json.read_text(encoding="utf-8"))
        mudou = False
        for nome, arch in (dados.get("architectures") or {}).items():
            hist = arch.get("history") or {}
            gravadas = len(hist.get("val_loss") or [])
            orcamento = int(arch.get("config", {}).get("epochs") or 0) or int(
                (dados.get("config") or {}).get("epochs") or 0
            )
            if not gravadas or not orcamento or gravadas >= orcamento:
                # Histórico íntegro, mas `epochs_executed` (ou a tolerância
                # citada) pode ter sobrado de uma recuperação anterior a esta
                # correção.
                reparos = []
                if gravadas and _reconciliar_epochs_executed(arch, gravadas):
                    reparos.append(f"epochs_executed -> {gravadas}")
                if _corrigir_tolerancia_declarada(arch):
                    reparos.append(f"rel_tol declarada -> {_TOL_CAUDA:g}")
                if reparos:
                    print(f"   ~ {nome:<24} {'; '.join(reparos)}")
                    mudou = True
                    tocados += 1
                else:
                    print(
                        f"   = {nome:<24} {gravadas}/{orcamento} épocas — "
                        f"nada a fazer"
                    )
                continue

            log = results_json.parent / "run.log"
            if not log.is_file():
                print(f"   ! {nome:<24} {gravadas}/{orcamento} — sem run.log")
                continue
            try:
                rec = parse_run_log(log)
            except ValueError as exc:
                print(f"   ! {nome:<24} {exc}")
                continue

            completo = len(rec["val_loss"])
            if completo <= gravadas:
                print(
                    f"   ! {nome:<24} log tem {completo} épocas, artefato tem "
                    f"{gravadas} — nada a recuperar"
                )
                continue
            motivo = _divergencia_da_cauda(rec, hist)
            if motivo:
                print(
                    f"   ! {nome:<24} ABORTADO: log e artefato podem ser de "
                    f"execuções diferentes — {motivo}"
                )
                continue

            print(
                f"   + {nome:<24} {gravadas} -> {completo} épocas "
                f"(orçamento {orcamento})"
            )
            if not write:
                tocados += 1
                continue

            arch["history"] = {k: rec[k] for k in _SERIES}
            arch["epochs"] = completo
            _reconciliar_epochs_executed(arch, completo)
            arch["training_stability"] = _analisar_estabilidade(
                arch["history"], orcamento
            )
            arch["history_recovery"] = {
                "source": str(log.relative_to(ROOT)) if log.is_relative_to(ROOT)
                else str(log),
                "script": "scripts/reporting/recover_truncated_history.py",
                "epochs_before": gravadas,
                "epochs_after": completo,
                "verification": (
                    f"a série gravada é sufixo exato da reconstruída "
                    f"(rel_tol={_TOL_CAUDA:g})"
                ),
                "note": (
                    "histórico truncado por retomada do BackupAndRestore; o "
                    "treino executou o orçamento inteiro (ver run.log). "
                    "training_stability recalculado sobre a série completa."
                ),
            }
            mudou = True
            tocados += 1

        if mudou and write:
            bak = results_json.with_suffix(".json.pre-recovery.bak")
            if not bak.exists():
                shutil.copy2(results_json, bak)
            results_json.write_text(
                json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"     gravado: {results_json.name} (backup em {bak.name})")
    return tocados


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="diretório do run")
    parser.add_argument("--write", action="store_true", help="aplica as mudanças")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"run não encontrado: {run_dir}")

    modo = "APLICANDO" if args.write else "SIMULACAO (use --write para aplicar)"
    print(f"== Recuperacao de historico truncado — {modo}")
    print(f"   run: {run_dir}")
    n = recuperar(run_dir, write=args.write)
    if not args.write:
        print(f"\n   {n} arquitetura(s) recuperavel(is). Repita com --write.")
    else:
        print(f"\n   {n} arquitetura(s) recuperada(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
