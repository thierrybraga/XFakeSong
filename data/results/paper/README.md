# Material academico (TCC)

`main.tex` e a fonte canonica. Tudo o mais nesta pasta e **gerado** e nao deve
ser editado a mao nem versionado como verdade.

## Estado em 2026-07-31

Os artefatos gerados foram **removidos**: `consolidated/`, `figures/`,
`tabelas_benchmark.tex` (substituido por um placeholder), `main.pdf` e os
subprodutos do LaTeX (`.aux`, `.log`, `.lof`, `.lot`, `.out`, `.toc`).

Motivo: todos vinham de execucoes sobre `benchmark_audio_raw_balanced_15k*` --
o dataset anterior, com atalho de fonte de 87,6% e disjuncao de falante vacua,
ja apagado do disco. Os numeros (EER medio de ~1%, acuracia de ate 99,87%)
medem desempenho *in-domain com atalho disponivel*; nao sao comparaveis com a
literatura nem com o dataset canonico atual
(`data/datasets/benchmark_dataset.npz`, CETUC pareado com clones XTTS-v2).
Mantidos versionados, um `pdflatex main.tex` os reintroduziria no artigo em
silencio.

**Consequencia esperada:** `main.tex` nao compila ate a nova bateria rodar --
faltam as figuras referenciadas. Isso e deliberado.

## Estado em 2026-08-06 -- qual run alimenta o artigo

O run completo do escopo oficial (11 arquiteturas, todas `status: ok`) e:

| Propriedade | Valor |
| --- | --- |
| Run | `data/results/clean_benchmark_15k` |
| Dataset | `data/datasets/benchmark_dataset_15k.npz` (15.000 amostras) |
| SHA-256 | `3775b35fb05155011b21a30ff60632aaa7c9f67f031bd6f7874a5f313363400f` |
| Particao | 12.162 treino / 1.456 val / 1.382 teste |
| Fingerprint do teste | `ab4c3a9fa97982624a21fdecce358c4fa046e6bfe37a01620c0fb561b16dd60a` |
| Conclusao | 2026-08-06 14:41 (RawGAT-ST, ultimo modelo) |

**Nao use `data/datasets/benchmark_dataset.npz` (40k) para regenerar o artigo.**
Ele e a variante canonica do projeto para o *protocolo* de dataset, mas a
bateria sobre ele foi interrompida manualmente durante o RawNet2, com apenas 6
das 11 arquiteturas concluidas (Hybrid CNN-Transformer, Conformer, WavLM
Original, HuBERT Original, RandomForest e SVM). Esta arquivada como registro
historico em `data/results/archive/benchmark_40k_2026-08-02/`. Nao existe run
completo sobre o 40k.

As duas variantes tem **conjuntos de teste diferentes** (fingerprints
distintos), entao seus numeros nao sao comparaveis e nao podem ser misturados
na mesma consolidacao -- nem via `--prefer-last`, nem passando os dois
diretorios como input.

> **Pendencia antes de consolidar (2026-08-06):** Conformer e RawGAT-ST estao
> marcados para retreino (o Conformer colapsou para `loss = ln 2` a partir da
> epoca ~16 e o numero publicado vem do checkpoint da epoca 10; o RawGAT-ST
> ficou abaixo dos baselines classicos em min t-DCF). Consolidar antes disso
> gera tabelas e figuras que serao descartadas.

> **CONCLUIDO em 2026-08-09 — reavaliacao dos SSL.** WavLM/HuBERT Original
> foram reavaliados com a janela corrigida (`--target-samples 48000`, o clipe
> de 3 s inteiro). Nao foi retreino: o backbone e congelado, entao so os
> embeddings e a cabeca foram refeitos (~5 min de GPU cada). Os artefatos
> antigos, da janela de 64.000, estao integralmente preservados em
> `data/results/archive/ssl_janela64000_2026-08-09/`, com o comando que os
> reproduz.
>
> **Os numeros mudaram, e no caso do HuBERT mudaram muito:**
>
> | Modelo | EER 64.000 | EER 48.000 | Veredito pareado |
> | --- | ---: | ---: | --- |
> | WavLM Original | 3,47% | 3,62% | empate (p = 0,27) |
> | HuBERT Original | 2,17% | **5,93%** | 64.000 melhor (p < 0,001) |
>
> A janela de 4 s ajudava o HuBERT de verdade — mas dava aos dois SSL 33% mais
> quadros que as outras nove arquiteturas recebem, alem de inserir uma emenda
> artificial no sinal. O default ficou em 48.000 por comparabilidade. Ver a
> discussao completa em
> [docs/evaluation/retraining-adjustments.md](../../../docs/evaluation/retraining-adjustments.md),
> secao 2026-08-09.

## Regenerar

```bash
# 1. (se necessario) refazer os modelos pendentes SOBRE O MESMO DATASET
python scripts/benchmark/run_models_sequential.py \
  --dataset data/datasets/benchmark_dataset_15k.npz \
  --models Conformer RawGAT-ST \
  --out data/results/clean_benchmark_15k --device-profile gpu --resume

# 1b. (JA FEITO em 2026-08-09) reavaliacao dos SSL com a janela de 48.000.
#     So refaca se mudar dataset ou janela.

# 2. consolidar (o run entra como argumento POSICIONAL, nao como flag)
#    Gera tambem benchmark_significance.json (McNemar exato + bootstrap
#    pareado, ajustados por Holm) — necessario porque os IC 95% de Conformer e
#    Hybrid CNN-Transformer se sobrepoem e nao decidem o topo do ranking.
python scripts/reporting/consolidate_results.py data/results/clean_benchmark_15k \
  --prefer-last --copy-to data/results/paper/figures

python scripts/reporting/update_tcc_latex.py
python scripts/reporting/export_tcc_extra_figures.py   # DET, score distributions
pdflatex main.tex
```

`--prefer-last` so importa quando o mesmo modelo aparece mais de uma vez nos
inputs; com um unico diretorio de run ele e inofensivo. Se o retreino do passo
1 for para um diretorio separado, passe os dois na ordem
`<run_original> <run_do_retreino>` para que o retreino prevaleca -- e **so** se
ambos forem do mesmo dataset.

> **Cuidado com o default do `--dataset`.** Em
> `scripts/benchmark/run_models_sequential.py` ele e
> `data/datasets/benchmark_dataset.npz` (o 40k). Omitir a flag treina contra um
> conjunto de teste diferente do run que alimenta o artigo, e o resultado
> **nao pode** entrar na mesma consolidacao. Passe o `_15k.npz` explicitamente.
