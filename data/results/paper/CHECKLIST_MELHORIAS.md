# Checklist consolidado de melhorias e ajustes — TCC XFakeSong

Consolida tudo o que foi levantado na sessão de 2026-08-14: revisão técnica
(fidelidade aos artefatos), compilação do PDF, revisão linguística e análise
estrutural. Deduplicado e priorizado.

**Legenda:** ✅ aplicado · ⬜ pendente · ⏱ esforço estimado

---

## PARTE I — Pendentes

### P1. Obrigatório antes da defesa
*São os pontos em que a banca derruba a alegação com os dados do próprio documento.*

| # | Item | Onde | ⏱ |
|---|---|---|---|
| 1 | ⬜ **Explicação causal do achado central é refutada pelos dados.** O texto diz que o diagnóstico de estabilidade ordena a robustez; dos 2 modelos `unstable_oscillation`, o Conformer é o **3.º mais robusto** (91,46%) e o RawGAT-ST o último. Reescrever para: nenhuma das variáveis examinadas ordena a robustez; o contraste AASIST × RawGAT-ST é observação em *n*=2 compatível com sobreajuste. | `main.tex:2554-2558` | 30 min |
| 2 | ⬜ **`n_fft` errado — publica os parâmetros de um bug corrigido.** Texto diz 512 amostras (→ 6,2% de sobreposição); o código deriva **1024** (`MIN_STFT_OVERLAP=0.5`, hop 480 → **53,1%**). Corrigir o texto e o docstring do módulo (`benchmark_frontend.py:9`). | `main.tex:1510` | 15 min |
| 3 | ✅ **Testes pareados ausentes sob ruído.** ~~Rodar `benchmarks/significance.py` sobre as predições a 10 e 5 dB.~~ **Executado** (2026-08-16): `consolidated/benchmark_significance_noise.json` cobre as quatro combinações (10 dB e 5 dB × cluster de locutor e de frase). Resultado: **nenhum dos dez pares do topo se separa** por McNemar nem por *bootstrap*. Resta só levar isso ao texto — o ordenamento sob ruído é bloco indistinguível, não *ranking*. | `benchmarks/significance.py` | ✅ |
| 4 | ⬜ **Previsão da §2.3 nunca confrontada.** Afirma que artefatos de fase são "só alcançáveis por arquiteturas que operam sobre a forma de onda"; as três redes raw terminam em 6.º, 7.º e 11.º. Retomar em §5.3 ou enfraquecer para hipótese. | `main.tex:811` / §5.3 | 30 min |
| 5 | ⬜ **Duas convenções opostas de erro sem aviso.** "Rejeitar um *bonafide*" = falso positivo em §4.5, mas $P_{\mathrm{miss}}$ no t-DCF/DET — invertidas, a 15 linhas de distância. Inserir tabela de correspondência em §4.5. | `main.tex` §4.5 | 20 min |
| 6 | ⬜ **AST classificado como SSL.** A 4.ª geração diz "Representação SSL e atenção: WavLM, HuBERT, **AST**". O AST é pré-treinamento **supervisionado** em AudioSet, como o §3.2 afirma corretamente. | `main.tex:864` | 5 min |

### P2. Alto retorno, baixo custo

| # | Item | Onde | ⏱ |
|---|---|---|---|
| 7 | ⬜ **30 subseções fora do sumário.** `\subsubsection*` (estrelado) em 11 arquiteturas, 9 na metodologia e 10 na análise. "Robustez a Ruído: o Achado Central" não aparece no sumário de um documento de 81 páginas. Remover o asterisco ao menos em §3 e §5.3. | 21 ocorrências | 20 min |
| 8 | ⬜ **11 das 24 equações descrevem descritores não usados.** A §"Descritores Adicionais" (~150 linhas) traz forma espectral, CQT, prosódia e delta para atributos "não consumidos pelos 11 modelos". Mover para apêndice — reequilibra a Metodologia de 38,6% para ~33% e resolve os 4 rótulos órfãos. | `main.tex:1548` | 30 min |
| 9 | ⬜ **Notação "±" em IC de Wilson.** "±0,31 p.p. no topo (99,71% → [99,26%; 99,89%])": o intervalo de Wilson não é simétrico (é −0,45/+0,18 em torno de 99,71%). Suprimir o "±" e reportar só o intervalo. | `main.tex:2850` | 5 min |
| 10 | ⬜ **QP2 reescrita *a posteriori*.** "A formulação é deliberadamente aberta quanto aos fatores" foi escrita depois de saber que as hipóteses caíram. Declarar francamente: partiu-se de duas hipóteses, ambas rejeitadas. | `main.tex:629` | 10 min |
| 11 | ⬜ **§1.1 promete mais do que §5.3 entrega.** Três parágrafos estabelecem que a assimetria de custo "orienta toda a leitura dos resultados"; §5.3 conclui que o $t$-DCF\* não reordena nada. Calibrar: a assimetria importa para o *ponto de operação*, não para a escolha do detector. | `main.tex:553` | 15 min |
| 12 | ⬜ **4 rótulos de equação órfãos** (`eq:cqt`, `eq:cqt_q`, `eq:delta`, `eq:zscore`) — resolvidos junto com o item 8, ou citando-os no texto. | — | 5 min |

### P3. Necessário para publicação

| # | Item | ⏱ |
|---|---|---|
| 13 | ⬜ **Reequilibrar as proporções.** Hoje: Metodologia 38,6%, Resultados 22,8%, Conclusões 22,2%. Alvo: ~30% / ~35% / ~10%. As Conclusões reimprimem os números de §5.3 em vez de sintetizá-los. | 4–6 h |
| 14 | ⬜ **§6 "Sistema Desenvolvido" quebra o IMRAD** — está entre Resultados e Conclusões e não traz medição nova. Mover para fim da Metodologia ou apêndice. | 1 h |
| 15 | ⬜ **Sem seção autônoma de trabalhos relacionados.** O posicionamento tem 30 linhas dentro de um capítulo intitulado "Tecnologias de Síntese de Voz". Falta tabela comparativa com os *benchmarks* de anti-*spoofing* existentes. | 3–5 h |
| 16 | ⬜ **Taxonomia bidimensional.** Quatro classificações concorrentes e irreconciliadas. A família "SSL e áudio bruto" funde *backbone* congelado com treino do zero — o eixo de que a conclusão depende. Substituir por representação × regime de treinamento. | 2 h |
| 17 | ⬜ **Validação *cross-generator*.** Sem ela a contribuição permanece "desempenho contra o XTTS-v2", como o próprio trabalho reconhece. É a limitação dominante. | semanas |
| 18 | ⬜ **Reavaliar SVM/RF com partição de teste nova** (limitação viii): o vetor de 183 descritores foi escolhido com conhecimento do teste. | 1 dia |
| 19 | 🟡 **Ablação do AST** (limitação ix) — **metade feita.** Braço (a) `300×128 + pretrained=False` concluído (`data/results/ast_abl_a/`): acurácia **0,9493** e EER **4,34%** contra 0,9971 / 0,14% do publicado — o pré-treinamento AudioSet vale **4,8 p.p.**, isolado. Falta o braço (b) `100×80 + pretrained=True`, que isola a RESOLUÇÃO: `python scripts/benchmark/run_ast_ablation.py --arm b`. Sem ele, resolução e topologia seguem confundidas. | ~1 dia → resta ~½ |

### P4. Formais e ABNT

| # | Item | ⏱ |
|---|---|---|
| 20 | ⬜ **Ficha catalográfica ausente** — NBR 14724 exige no verso da folha de rosto. | 30 min |
| 21 | ⬜ **"Prof.ª Ana Cláudia"** sem sobrenome nem titulação, na folha de rosto e na de aprovação. | 5 min |
| 22 | ⬜ **`\versaofinaltrue`** (linha 91) para o depósito impresso: títulos e *hyperlinks* em preto. | 1 min |
| 23 | ⬜ **`FPR`, `FNR`, `TPR` em itálico matemático** — pela ISO 80000-2, siglas em fórmula deveriam ser verticais, como os subscritos já corrigidos. | 15 min |
| 24 | ⬜ **Anáfora vaga com "Isso"** (linhas ~528 e ~1716) — trocar pelo referente explícito. Estilo, não erro. | 10 min |

### P5. Abertos pela sessão de 2026-08-17

| # | Item | Onde | ⏱ |
|---|---|---|---|
| 25 | ⬜ **Retreino do RawGAT-ST — o único obrigatório do escopo.** Divergência código↔artefato real (`decay_steps` 100.000 no artefato publicado), instabilidade, e 5,29 p.p. perdidos ao selecionar a época por `val_loss` (ép. 17 contra 88). Config: dropout **0,35**, L2 **1e-3**, decay **152.100**, `--checkpoint-monitor val_eer`. Não use 0,50/3e-3 — o fatorial mediu que dropout 0,50 trava a validação no acaso. | `run_rawgat_retune.py` | ~27 h GPU |
| 26 | ⬜ **Incorporar ao texto o fatorial do RawGAT-ST.** Três células medidas (publicado 0,8997; dropout 0,50 → 0,5000 com treino a 95,4%; L2 3e-3 → 0,8984). É evidência direta para a discussão de sobreajuste da §5.3, hoje apoiada em *n*=2. | `main.tex` §5.3 | 45 min |
| 27 | ⬜ **Registrar no texto a divergência de `decay_steps` de CCT e AST.** Não é retreino (ver justificativa abaixo), mas o TCC deve dizer que o código corrigido não regenera bit a bit os dois artefatos publicados. | `main.tex` (limitações) | 20 min |
| 28 | ⬜ **Qualificar a leitura de importância de atributos.** SHAP dá LFCC 55,7% / Temporal 27,1%; permutação dá Temporal 44,7% / LFCC 41,5% — as famílias INVERTEM. A permutação subestima famílias redundantes (120 colunas LFCC correlacionadas contra 11 temporais). O texto hoje apresenta a permutação como fato. | `main.tex` (§ XAI) | 30 min |
| 29 | ⬜ **Renomear `ALL_TCC_ARCHITECTURES`** — o nome promete o conjunto completo e entrega 9 de 11 (exclui WavLM e HuBERT Original). `KERAS_TCC_ARCHITECTURES` elimina a armadilha. | `benchmarks/config.py` | 15 min |
| 30 | ✅ **`decay_steps` do AASIST — nada a fazer.** Registrado porque foi esta entrada (100.000) que a auditoria de 16/08 leu por engano como sendo do RawGAT-ST. Com batch 24 o orçamento real é ceil(24.324/24)×100 = **101.400**: cobertura de 0,99, dentro da tolerância. A entrada está correta. | `planning.py:220` | ✅ |

---

## PARTE II — Já aplicado nesta sessão

### Rigor / fidelidade aos artefatos
- ✅ **Front-end tabular decidido sobre o teste selado** — `warningbox` em §4.3 + limitação (viii). Inclui o resultado real: LFCC ajudou o RandomForest a 5 dB (AUC 0,838→0,852) e **não ajudou o SVM** (0,8494→0,8499).
- ✅ **SHA-256 da partição de teste** — era de um run arquivado; corrigido para `4f9b8630…abab6` + método declarado.
- ✅ **AST não usa o mesmo front-end** — tabela separada (300×128 vs 100×80), `warningbox` e limitação (ix).
- ✅ **Recorte dos modelos raw** — não é "1 s central da janela de 5 s"; é a janela íntegra de 3 s (48000). Reescrito + `infobox`.
- ✅ **Limiar θ≈0,48 para os SSL** — inexistente nos artefatos; removido da legenda.
- ✅ **"p = 1,00 sob ambas as unidades"** — falso na unidade frase (0,37 e 0,10).
- ✅ **"degradação monotônica"** — AASIST e RawGAT-ST sobem a 30 dB; RawGAT ainda inverte entre 10 e 5 dB.
- ✅ **AASIST 5.ª → 7.ª** posição em acurácia limpa.
- ✅ **Top-15 da importância** — 5/7/3, não 6/7/2.
- ✅ **Textos por partição** — 601/196/183, não 599/195/183.
- ✅ **Números de estabilidade sem fonte** — origem declarada + explicada a aparente contradição com a coluna "Queda".
- ✅ **"Corrigível por recalibração"** — o artefato mostra o oposto: limiar calibrado leva o SVM a 57,31%, *abaixo* dos 58,68% do fixo.
- ✅ **RESUMO/ABSTRACT afirmavam o contrário da §5.3** (representação e capacidade como "fatores de primeira ordem" — são as hipóteses rejeitadas).

### Reporte
- ✅ **Coluna de 5 dB** acrescentada ao gerador `update_tcc_latex.py` e ao fragmento — sobrevive à regeneração.
- ✅ **Nova subseção "Dispersão entre Locutores Não Vistos"** — a coluna "Pior locutor" não era mencionada; o SVM cai a 54,03% no pior falante.
- ✅ **Truncamento das 59 importâncias negativas** declarado, com as duas normalizações.
- ✅ **Auditoria de quase-duplicatas espectrais** (0 pares em 272,7 M comparações) na tabela do corpus.
- ✅ **Multicrop declarado como inócuo** (os 3 crops coincidem).

### Compilação (o PDF não era gerado)
- ✅ **Erro fatal no preâmbulo** — `title=#1` sem chaves: vírgula no título da `infobox` quebrava a lista de opções.
- ✅ **8 transbordamentos de margem** (até 174 pt) — identificadores longos em `\texttt{}` → `\path{}` + `\UrlBreaks` estendido.
- ✅ **Tabela do apêndice** com colunas `p{}` em vez de `ll`.

### Língua e tipografia
- ✅ Frase agramatical ("quando há onde a resposta é aplicada"), `onde` não locativo, 1.ª pessoa isolada ("Verificamos"), futuro perifrástico, paralelismo em enumeração, ambiguidade temporal das eleições, anáfora invertida ("não deixa de importar por isso").
- ✅ 13 subscritos multiletra em itálico → `\mathrm`; 19 unidades fora do `siunitx`; estrangeirismos sem itálico; capitulação de título; nomenclatura "WavLM Original".
- ✅ `\num{}` com ponto decimal no ABSTRACT (13 ocorrências, quebrava o `siunitx`).
- ✅ Comandos do apêndice apontavam para `.npz` inexistente; §6 omitida na "Organização do Trabalho"; apêndice de aliases sem remissão.

---

## Estado atual verificado

- PDF: **81 páginas**, 0 erros, 0 referências pendentes, 0 transbordamento >20 pt, 12/12 figuras.
- 45 verificações numéricas contra os artefatos: **todas passam**.
- 0 `\ref` órfão, 0 `\cite` sem `\bibitem`, bibliografia em ordem de citação.

## Ordem sugerida de ataque

1. Itens **1, 2, 6, 9** — quatro edições cirúrgicas, ~1 h, eliminam os erros factuais restantes.
2. Itens **7, 8, 12** — ~1 h, ganho grande de navegabilidade e equilíbrio.
3. Itens **4, 5, 10, 11** — ~1,5 h, fecham os arcos argumentativos abertos.
4. Item **3** — meia jornada, mas é o que dá força estatística ao capítulo central.
5. **P3/P4** conforme o destino: defesa (P4) ou submissão (P3).
