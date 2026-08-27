#!/usr/bin/env bash
# Retreino do RawGAT-ST: correcao do cronograma de decaimento.
#
# ENQUADRAMENTO (decidido em 2026-08-17). Este NAO e um retreino em busca de
# numero melhor -- e a correcao de um defeito que o proprio TCC declara na
# secao de limitacoes. A distincao importa: o `test-lock` do dataset afirma que
# a particao de teste "nao foi usada para selecionar arquitetura,
# hiperparametros ou correcoes", e essa afirmacao so sobrevive se a mudanca for
# a reparacao de um defeito documentado ANTES de olhar o resultado.
#
# !! LEIA ANTES DE RODAR (2026-08-17, revisao) !!
#
# A PREMISSA DESTE SCRIPT NAO SE SUSTENTA NOS DADOS. Ele foi escrito supondo
# que as 34 epocas finais com LR congelado eram desperdicio. A trajetoria de
# validacao do run publicado diz o contrario -- elas sao a MELHOR fase:
#
#     ep   1- 17  media val_acc 0,7037   (a epoca selecionada esta aqui)
#     ep  18- 40  media val_acc 0,8149
#     ep  41- 66  media val_acc 0,8476
#     ep  67-100  media val_acc 0,8608   <- LR congelado, e o PICO (0,8997) esta na 88
#
# O piso do cosseno nao atrapalhou: e annealing funcionando. Corrigir
# `decay_steps` faria o LR decair mais devagar e NUNCA alcancar o regime que
# produziu o melhor resultado -- pode piorar, e nao ha evidencia de que melhore.
#
# O unico custo MEDIDO deste modelo e a selecao da epoca: 17 (val_acc 0,8468)
# contra o pico na 88 (0,8997) = 5,29 p.p. Isso se resolve com
# `--checkpoint-monitor val_eer`, nao com decay_steps -- mas trocar o criterio
# em UM modelo quebra a comparabilidade com os outros dez da tabela.
#
# Mantido no repositorio como registro da hipotese e da sua refutacao. Se for
# rodar mesmo assim, trate o resultado como exploratorio, fora da tabela.

# O QUE MUDA, E SO ISSO
#   decay_steps  100.000 -> 152.100
#
# 152.100 = ceil(24.324/16) x 100 epocas = o orcamento REAL de passos. Com
# 100.000 o cosseno atingia o piso na epoca 66 e as 34 finais rodavam com a
# taxa de aprendizado congelada.
#
# O QUE NAO MUDA, DE PROPOSITO
#   - dropout 0,35 e L2 1e-3: o ajuste para 0,50/3e-3 foi REFUTADO pelo
#     fatorial de 17/08 (com dropout 0,50 a validacao fica em 0,5000 exato por
#     25 epocas enquanto o treino chega a 95,4%) e revertido no codigo;
#   - selecao do checkpoint por `val_loss`: trocar para `val_eer` recuperaria
#     5,29 p.p. NESTE modelo, mas os outros dez da tabela foram selecionados
#     por `val_loss`. Trocar em um so quebraria a comparabilidade que e o
#     ponto do protocolo. Se quiser a ablacao do criterio, rode um segundo
#     run com `--checkpoint-monitor val_eer` e reporte SEPARADO da tabela.
#
# CUSTO: ~27 h em RTX 3060 (medido: 26,85 h no run publicado).
#
# ARMADILHA JA PISADA: rodar `--plan-only` no HOST Windows resolve o perfil
# para cpu (sem TensorFlow instalado) e o plano sai com batch_size=4 em vez de
# 16 -- o que tornaria `decay_steps` errado de novo. Valide o plano DENTRO do
# container, como este script faz.
set -euo pipefail

IMAGEM="${XFAKE_IMAGE:-xfakesong/benchmark:nvidia}"
DATASET="${XFAKE_DATASET:-data/datasets/benchmark_dataset_15k.npz}"
SAIDA="${XFAKE_OUT:-data/results/rawgat_retune_decay}"
NOME="${XFAKE_CONTAINER:-xfake_rawgat_retune}"

ARGS=(
  python scripts/benchmark/run_benchmark.py
  --model RawGAT-ST
  --dataset "$DATASET"
  --out "$SAIDA"
  --device-profile gpu
  --epochs 100
  --batch-size 32
  --seed 42
  --snr 30 20 10 5
  --train-aug-snr 30 20 10
  --train-noise-copies 1
  --waveform-noise-batch-size 64
  --latency-runs 30
  --experiment-scope official
  --waveform-train-augmentation
  --no-api
  --fail-on-source-shortcut
)

echo "==> 1/2 validando o plano DENTRO do container (batch e decay_steps)"
docker run --rm --gpus all -v "$(pwd)":/app -w /app "$IMAGEM" \
  "${ARGS[@]}" --out "${SAIDA}_planonly" --plan-only >/dev/null

docker run --rm -v "$(pwd)":/app -w /app "$IMAGEM" python - <<'PY'
import json, math, sys
d = json.load(open("data/results/rawgat_retune_decay_planonly/benchmark_plan.json",
                   encoding="utf-8"))
c = d["architectures"]["RawGAT-ST"]["training_config"]
b, ep = c["batch_size"], c["epochs"]
passos = math.ceil(24324 / b) * ep
cobertura = c["decay_steps"] / passos
print(f"    batch={b} epocas={ep} passos={passos} decay={c['decay_steps']} "
      f"cobertura={cobertura:.3f}")
print(f"    dropout={c['dropout_rate']} l2={c['l2_reg_strength']}")
erros = []
if b != 16:
    erros.append(f"batch_size={b}, esperado 16 (perfil de dispositivo errado?)")
if abs(cobertura - 1.0) > 0.02:
    erros.append(f"cobertura do decay={cobertura:.3f}, esperado ~1,000")
if c["dropout_rate"] != 0.35 or c["l2_reg_strength"] != 0.001:
    erros.append("dropout/L2 divergem do artefato publicado (0,35 / 1e-3)")
if erros:
    print("    ABORTA:", "; ".join(erros)); sys.exit(1)
print("    plano OK")
PY

echo "==> 2/2 iniciando o treino (~27 h). Acompanhe com: docker logs -f $NOME"
docker run -d --name "$NOME" --gpus all -v "$(pwd)":/app -w /app \
  --memory "${DOCKER_TRAIN_MEMORY_LIMIT:-36G}" --restart on-failure:3 \
  "$IMAGEM" "${ARGS[@]}"
