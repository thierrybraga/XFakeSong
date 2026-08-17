# Revisão técnica e acadêmica — `main.tex` (3621 linhas, revisão de 2026-08-14)

Método: cada afirmação numérica do texto foi conferida contra os artefatos da
execução que gerou os resultados (`data/results/paper/consolidated/*.json`,
`data/results/clean_benchmark_15k/*/results.json`, `benchmark_protocol.json`,
`rf_permutation_importance.csv`) e contra o código (`registry.py`,
`benchmarks/runner.py`, `benchmark_frontend.py`, routers da API). O que está
marcado como **verificado** foi recontado; o que está marcado como **conferir**
não pôde ser fechado neste ambiente (não há TeX instalado).

---

## A. Bloqueadores de rigor

Estes afetam a validade de alegações centrais do trabalho, não a redação.

### A1. O conjunto de teste selado foi usado para decidir o front-end tabular

Duas passagens declaram, explicitamente, que famílias de características foram
adicionadas **em resposta a um resultado do conjunto de teste**:

- linha 1337–1339 (RASTA-PLP): *"foi incorporado especificamente para mitigar a
  degradação de SVM e Random Forest sob ruído, discutida na Seção 5"*;
- linha 1346–1349 (LFCC+Δ/ΔΔ): *"Esta família foi acrescentada em resposta ao
  colapso do ponto de operação dos classificadores clássicos sob o SNR não visto
  de 5 dB, documentado na Seção 5"*;
- linha 2329–2330: *"Esse contraste justifica, a posteriori, a extensão do vetor
  de 63 para 183 descritores"*.

E a Seção 4.4.3 (linha 1788–1793) declara o oposto: *"A partição de teste foi
definida e criptograficamente selada […] com a declaração de que **não foi usada
para escolher arquitetura, hiperparâmetros ou correções**"* — declaração que o
`test_lock` repete literalmente em todos os 11 artefatos.

As duas não podem ser verdadeiras ao mesmo tempo. A mudança de 63 → 183
descritores é uma alteração de front-end (o próprio repositório a trata como
contrato novo, `benchmark_tabular_v2`), portanto uma *correção* no sentido da
declaração. Consequência: os números de SVM e Random Forest **não gozam da mesma
garantia de cegueira** que os nove modelos neurais.

**Correção possível (em ordem de preferência):**
1. Mostrar que a decisão foi tomada sobre a partição de **validação** (o colapso
   a 5 dB também é observável nela) e reescrever as duas passagens nesses termos.
2. Se a decisão foi mesmo tomada olhando o teste, declarar isso na Seção de
   limitações: o protocolo é cego para os neurais e não-cego para os clássicos,
   e os resultados tabulares devem ser lidos como limite superior otimista.

Deixar como está é o problema mais sério do documento, justamente porque o
"protocolo auditável" é listado como contribuição.

### A2. O SHA-256 da partição de teste impresso no Apêndice não é o do run

| Onde | Valor |
|---|---|
| Tabela `tab:ambiente_reproducao` (linha 3431) | `ab4c3a9f… dd60a` |
| `benchmark_protocol.json` → `test_lock.test_archive_identity_sha256` | `4f9b8630…abab6` |
| Os 11 `results.json` → `academic_protocol_guard.test_lock` | `4f9b8630…abab6` |

O hash `ab4c3a9f…` aparece **apenas** em
`data/results/archive/classicos_tabular_v1_2026-08-09/randomforest/…/metrics.json`
— uma execução arquivada, anterior. O SHA-256 do *dataset*
(`3775b35f…3400f`) está correto.

Verificado. Como o selo criptográfico é vendido como "conferível, e não
declarativa" (linha 2940–2942), um leitor que tente conferir o único hash
impresso falha. Trocar por `4f9b8630…abab6` e declarar o método
(`sha256(zip_member_name_crc32_uncompressed_size)`).

### A3. O AST não consome a mesma representação das outras redes espectrais

A Tabela `tab:representacoes_entrada` (linha 1313–1315) afirma que Conformer,
CCT, **AST** e Res2Net consomem "Mapa 100 × 80 em dB". Os artefatos dizem outra
coisa:

| Modelo | `input_preparation.prepared_shape` |
|---|---|
| CCT, Res2Net, Conformer | `[100, 80]` |
| **AST** | **`[300, 128]`** |

A Seção 3.2 (linha 1074–1078) descreve corretamente o AST em 300 × 128 (128
bandas, salto 10 ms, janela 25 ms). Ou seja: o texto se contradiz, e a
Equação `eq:melspec` — que define o front-end espectral do trabalho — **não
descreve o front-end do modelo de melhor acurácia**.

Isso importa além da tipografia. O AST recebe 3× a resolução temporal e 1,6× as
bandas dos outros três; a leitura "comparar famílias sob protocolo idêntico"
(linhas 804 e 880) não se sustenta dentro da própria família espectral. Some-se
a isso que só o AST parte de pesos AudioSet transferidos (confirmado em
`spectrogram_transformer.py:388,515`) e o "99,71%" fica com duas vantagens não
controladas em relação aos seus pares diretos.

**Correção:** ajustar a tabela para listar as duas entradas espectrais, e na
Seção 5.3 declarar explicitamente que a comparação AST × {CCT, Conformer,
Res2Net} confunde arquitetura, resolução de entrada e pré-treinamento. O empate
técnico já reportado (AST ≈ CCT ≈ Conformer) fica, aliás, *mais* interessante
com essa ressalva: o AST não se separa dos pares apesar das duas vantagens.

### A4. A descrição do recorte dos modelos de áudio bruto está errada em três pontos

Texto (linhas 1310–1311 e 1427–1430): *"RawNet2/AASIST/RawGAT-ST consomem recorte
central de 1 s (16000 amostras) da janela de 5 s"*.

Contra o `registry.py` (blocos `input_requirements` de `aasist`, `rawgat_st`,
`rawnet2`) e o `input_preparation` dos artefatos:

| Alegação | Realidade |
|---|---|
| recorte de **1 s** (16000) | `target_sequence_length: 48000` — os três consomem a janela **inteira de 3 s**. 16000 é o `min_sequence_length`, não o comprimento consumido |
| recorte **central** | `train_crop_strategy: random`, `eval_crop_strategy: multicrop`, `eval_num_crops: 3`, `eval_score_aggregation: mean_over_crops` |
| janela de **5 s** | a janela canônica é 3 s; o próprio código registra que `max_duration: 5.0` era um bug corrigido (comentário em `registry.py`, bloco do RawNet2) |

Verificado. A frase é usada como justificativa metodológica ("para conter o
custo quadrático da atenção em grafos e o comprimento da sequência da GRU"), ou
seja, explica uma decisão que não foi tomada.

### A5. Contradição sobre o limiar de decisão

- Legenda da Figura `fig:pipeline` (linha 1191–1193): *"para WavLM Original e
  HuBERT Original, é calibrado em validação com ruído (θ ≈ 0,48)"*.
- Seção 4.5 (linha 1857): *"O limiar é o padrão θ = 0,5 para **os 11 modelos,
  sem exceção**"*.

Artefatos: `decision_threshold = 0.5` para os 11 modelos, no conjunto limpo e em
todos os quatro níveis de SNR; `metric_threshold_policy: fixed_comparison`.
Verificado. A legenda da figura está errada — apagar a segunda oração.

---

## B. Erros factuais conferidos contra os artefatos

### B1. "p = 1,00 nos três pares sob ambas as unidades" é falso

Linha 2244–2245. Recontado em `benchmark_significance.json`:

| Par | McNemar p (Holm), unidade **frase** | unidade **locutor** |
|---|---|---|
| AST × CCT | 1,000 | 1,000 |
| CCT × Conformer | **0,368** | 1,000 |
| AST × Conformer | **0,095** | 1,000 |

O próprio parágrafo anterior (linhas 2223–2224) reporta 0,37 e 0,10. A
afirmação vale só para a unidade locutor. Reescrever: *"sob a unidade locutor os
três pares vão a p = 1,00; sob a unidade frase nenhum atinge significância a
5%"* — a conclusão de empate técnico sobrevive, a formulação é que não.

### B2. A degradação sob ruído não é monotônica

Linha 2364: *"degradação suave e monotônica em todo o recorte"*. Pela
`tab:robustez_awgn`:

- AASIST: 94,72 → **95,01** (30 dB) → **95,08** (20 dB) → 91,68 (10 dB)
- RawGAT-ST: 87,55 → **87,63** (30 dB) → 85,38 → 77,21

Dois dos onze **sobem** antes de cair. Trocar por "suave e majoritariamente
monotônica, com exceção de AASIST e RawGAT-ST, que melhoram levemente a 30 dB —
efeito do augmentation casado".

### B3. Posição do AASIST em acurácia limpa: 5.ª ou 7.ª?

Pelo `benchmark_summary.json`, AASIST (94,72%) é **7.º em acurácia** e **5.º em
EER**. O texto usa as duas:

| Linha | Diz | Correto? |
|---|---|---|
| 2262 | "EER de 2,60% e ocupa a 5.ª posição" | ✔ (é EER) |
| 2109 e 2916 | "7.ª posição em acurácia limpa" | ✔ |
| 2376–2377 | "o AASIST sobe da 5.ª para a 2.ª posição" (contexto: acurácia) | ✘ — é da 7.ª |
| 2573 | "a 5.ª melhor acurácia limpa (94,72%)" | ✘ — é a 7.ª |

### B4. Contagem de famílias no top-15 da importância por permutação

Linha 2319–2320: *"6 são temporais, 7 são LFCC e apenas 2 são MFCC"*. Recontado
diretamente em `rf_permutation_importance.csv` (top 15 por `importance_mean`):
**5 temporais, 7 LFCC, 3 MFCC** — em qualquer das duas normalizações (ver C3).

### B5. Textos distintos por partição

`tab:dataset_fontes` (linha 1643): 599 (Tr) / 195 (Vl) / 183 (Te). Artefato
(`dataset.metadata.splits.*.sentences`): **601 / 196 / 183**. O teste confere; o
treino e a validação não. Se as duas contagens medem coisas diferentes (índice
de frase vs. `text_id`), o rótulo da linha precisa dizer qual.

### B6. Os números de estabilidade citados não estão na tabela referenciada

As linhas 2613–2621 citam "desvio de 0,0963", "queda máxima de 0,4890"
(Conformer) e "0,0274 / 0,2232" (RawGAT-ST) remetendo a
`tab:estabilidade_treinamento`. Essa tabela só traz pico / época / final /
queda-pico−final. Pior: para o Conformer a tabela mostra a **menor** queda do
recorte (0,34%), enquanto o texto o chama de *"caso mais grave"*. O leitor não
tem como reconciliar.

Os valores existem em `results.json → training_stability` (o `status`
`unstable_oscillation` de Conformer e RawGAT-ST está confirmado). Ou a tabela
ganha as duas colunas do diagnóstico, ou o texto cita a fonte.

### B7. "Época do checkpoint" vs. "época do pico de validação"

A Seção 4.7 declara seleção do checkpoint por **menor `val_loss` limpa**. A
`tab:estabilidade_treinamento` e o texto reportam épocas cujo `best_val` é
**acurácia** (99,45%, 99,59%, …). São critérios diferentes e podem apontar
épocas diferentes. Ou unificar, ou dizer que a coluna "Época" é o pico de
acurácia de validação (descritivo) e não a época do checkpoint avaliado.

---

## C. Lacunas de reporte

### C1. A coluna "Pior locutor" nunca é analisada — e é o resultado mais alinhado ao objetivo

`tab:robustez_awgn` traz uma última coluna que o texto **não menciona uma única
vez**. Ela mede exatamente a alegação de generalização a locutor não visto que o
protocolo faz:

| Modelo | Pior dos 11 locutores (limpo) |
|---|---|
| AST | 98,41% |
| CCT | 97,58% |
| Conformer | 90,32% |
| AASIST | 88,10% |
| Res2Net | 87,30% |
| WavLM Original | 81,45% |
| RawNet2 | 74,19% |
| Random Forest | 73,39% |
| HuBERT Original | 70,97% |
| RawGAT-ST | 65,08% |
| **SVM** | **54,03%** |

O SVM, que a Seção 5.3 apresenta como "atrativo sob restrição severa de
recursos" com 93,20% de acurácia, cai a **4 pontos do acaso** no pior locutor.
Isso qualifica fortemente a recomendação de implantação e é evidência direta de
variância inter-locutor — que o trabalho tem e não usa. Deve virar um parágrafo
próprio na Seção 5.3 e uma ressalva na conclusão (linha 2901–2904).

### C2. O nível de 5 dB não tem coluna em nenhuma tabela

A Seção `sec:snr5` é chamada de *"O resultado mais informativo do recorte"*, mas
todos os seus números vivem apenas no corpo do texto. A `tab:robustez_awgn` para
em 10 dB, embora a `tab:prep_part` declare 5 dB como nível avaliado. Os valores
existem e conferem (`robustness["5"]`): SVM 58,68% / recall 18,38% / AUC 0,884 /
EER 20,41%; Res2Net 88,13%; Conformer 86,25%; WavLM 85,96%; RF 74,96%.
Acrescentar a coluna ao gerador `update_tcc_latex.py`.

### C3. A normalização da importância por permutação não é declarada

Os percentuais publicados (44,7 temporal / 41,6 LFCC / 8,7 MFCC / 5,0 RASTA) só
reproduzem se as **59 importâncias negativas** (de 183) forem truncadas em zero
antes de normalizar. Sem truncar:

| Família | Truncado (publicado) | Bruto |
|---|---|---|
| Temporal | 44,7% | **51,0%** |
| LFCC | 41,6% | 36,7% |
| MFCC | 8,7% | 6,8% |
| RASTA-PLP | 5,0% | 5,5% |

A leitura da linha 2346 — *"o bloco cepstral responde por 50,3% — ainda metade
da importância"* — vira **43,5%** sob a normalização bruta, o que muda o
argumento. Declarar o tratamento (é uma escolha defensável: importância negativa
é ruído amostral) e, de preferência, reportar as duas.

### C4. "Corrigível por recalibração" não é o que o artefato mostra

Linha 2437–2439. A Seção 4.5 aponta `calibrated_threshold` /
`accuracy_at_calibrated_threshold` como a evidência do ganho disponível. Para o
SVM a 5 dB:

- limiar fixo 0,5 → **58,68%**
- limiar calibrado → **57,31%** (pior)
- limiar-oráculo de EER → 79,59%

Ou seja: o limiar calibrado **não recupera nada**; o ganho só aparece sob um
oráculo ajustado no próprio conjunto de teste. Isso é um achado melhor e mais
honesto do que o atual — o teto de recuperação existe (≈ 80%), mas nenhuma
regra de calibração disponível o alcança sob degradação fora da distribuição.
Reformular nesses termos.

Na mesma seção, *"nenhum outro exibe a dissociação acurácia/AUC observada no
SVM"* é forte demais: a 5 dB o CCT vai a 81,84% com AUC 0,955 (oráculo 87,92%) e
o AST a 84,88% com AUC 0,952 (oráculo 91,39%). A dissociação existe em vários; o
SVM é o caso extremo. Trocar "nenhum outro exibe" por "nenhum outro na mesma
magnitude".

### C5. Assimetria de avaliação não declarada (mas inócua — vale dizer isso)

Os artefatos publicam `eval_crop_strategy: multicrop`, `eval_num_crops: 3`,
`mean_over_crops` para RawNet2/AASIST/RawGAT-ST e `center`/1 crop para os
demais. Isso lê como test-time augmentation só para uma família. O
`runner.py:1950-1965` documenta que a média sobre crops é inócua quando a janela
canônica cobre o sinal inteiro — que é o caso aqui (48000 = 48000, os três
crops coincidem). O texto deveria dizer isso, já que o campo aparece nos
artefatos que o trabalho declara auditáveis.

### C6. A auditoria de quase-duplicatas espectrais existe e não entra na tabela

`docs/data/dataset-protocol.md` registra o bloco F: 0 pares com cosseno ≥ 0,99
atravessando partições em 272,7 milhões de comparações (maior similaridade
observada 0,9838). É evidência **mais forte** que a sobreposição exata de
SHA-256, que só pega duplicata bit a bit. Acrescentar à `tab:dataset_fontes`
como quinta dimensão de auditoria — reforça a contribuição de protocolo sem
custo nenhum.

---

## D. LaTeX, ABNT e forma

### D1. `\num{}` com ponto decimal no ABSTRACT — provável falha de compilação

**Conferir compilando.** O preâmbulo fixa `input-decimal-markers = {,}` (linha
48), o que remove o ponto da lista de marcadores decimais aceitos na entrada.
Há **13** ocorrências de `\num{}` com ponto, todas no ABSTRACT (linhas
349–359): `99.71`, `99.57`, `0.14`, `0.43`, `98.91`, `97.76`, `92.40`, `91.68`,
`91.46`, `88.21`, `77.21`, `58.68`, `0.884`. Em siunitx v3 isso produz
`Invalid number`. Mesmo que compilasse, `output-decimal-marker = {,}` renderiza
"99,71" em texto inglês.

Correção: usar vírgula na entrada e, se quiser ponto na saída do abstract,
envolver o bloco em `{\sisetup{output-decimal-marker={.}, group-separator={,}} … }`.

### D2. Os comandos de reprodução do Apêndice C apontam para um arquivo inexistente

Cinco blocos usam `data/datasets/benchmark_audio_raw_balanced_15k.npz`. Em
`data/datasets/` existem apenas `benchmark_dataset.npz` e
`benchmark_dataset_15k.npz` — e é este último que a própria
`tab:ambiente_reproducao` registra e que o `test_lock` referencia
(`/app/data/datasets/benchmark_dataset_15k.npz`). Os comandos publicados não
executam. Verificar também se `run_tcc_pipeline.py` (existe) é mesmo o caminho
usado, ou se foi `run_clean_benchmark_pipeline.py`.

### D3. "Organização do Trabalho" pula a Seção 6

Linhas 615–623: a enumeração vai da Seção de resultados direto às conclusões e
**omite a Seção "Sistema Desenvolvido"** — que tem 140 linhas e fecha a QP3.

### D4. O apêndice de aliases nunca é referenciado no corpo

`sec:aliases` mapeia AST → `SpectrogramTransformer`, CCT → `Hybrid
CNN-Transformer`, Res2Net → `MultiscaleCNN`. Nenhum `\ref` aponta para ele. O
leitor atravessa três seções vendo nomes da literatura sem saber que o
repositório os registra com outro nome. Como esses três nomes carregam alegação
de fidelidade aos artigos originais, a remissão devia estar na **primeira
menção**, na Seção 3.

Outros `\label` órfãos (sem impacto, mas indicam remissões que se perderam):
`eq:cqt`, `eq:cqt_q`, `eq:delta`, `eq:zscore`, `sec:dataset_protocolo`,
`sec:posicionamento`, `sec:resultados_consolidados`.

### D5. Gênero dos ordinais alterna para o mesmo referente

"5.ª posição" (2262) / "5.º" (2376) / "2.ª" (2378) / "2.º" (2396) / "7.º"
(2253) / "3.º melhor modelo" (2616). Padronizar — feminino para
"posição/colocação", masculino para "lugar" — e definir um
`\newcommand{\ordf}[1]{\num{#1}.\textordfeminine}` para não repetir o erro.

### D6. Contradição sobre o VAD

Seção 4.6 (linha 1984): *"dada a indisponibilidade do ecossistema `torchaudio`
no ambiente de execução, utilizou-se o VAD por energia descrito na Seção 4.2"*.
Mas a `infobox` da Seção 4.2 diz que o VAD está **desativado** no corpus
experimental (`vad: disabled`), e a `tab:ambiente_reproducao` registra PyTorch
2.5.1+cu124 no ambiente. Nenhum VAD foi aplicado aos dados avaliados. Reescrever
o item para: *"o pipeline integra Silero VAD com fallback por energia; nenhum
dos dois foi aplicado ao corpus experimental (ver Seção 4.2)"*.

### D7. Resumo × Abstract fora de correspondência (NBR 6028)

O ABSTRACT traz os EER (0,14% / 0,43%) que o RESUMO omite. O RESUMO usa `\num`
consistentemente; o ABSTRACT mistura `\num` com números crus
("12,162/1,456/1,382"). Espelhar conteúdo e formatação.

### D8. Folha de rosto e de aprovação incompletas

"Orientadora: Prof.ª Ana Cláudia" — sem sobrenome e sem titulação, repetido na
folha de aprovação. Para depósito, nome completo e titulação (Dra.). Falta
também a ficha catalográfica (NBR 14724 a exige no verso da folha de rosto).

### D9. Miudezas

- linha 1075: `\textit{ckeckpoint}` → `checkpoint`.
- linhas 5–7 (cabeçalho do arquivo): lista "WavLM, HuBERT" sem o qualificador
  "Original", que o texto usa para distinguir do porte Keras. Induz a erro quem
  editar o arquivo.
- `\usepackage[utf8]{inputenc}` é redundante desde o LaTeX 2018 (inofensivo).
- "CCT-2/3×2": na notação de Hassani et al., o primeiro número é a quantidade de
  blocos Transformer. O texto descreve **4** blocos de autoatenção, então a
  designação correta seria CCT-4/3×2.

---

## E. O que está certo (e é bastante)

Vale registrar, porque a densidade de erros acima não é representativa do
conjunto.

**Integridade de referências — impecável.** 0 `\ref` sem `\label`, 0 `\cite` sem
`\bibitem`, 0 `\bibitem` não citado (54 chaves, 55 itens; o único "fora de
ordem" é `kingma2015adam`, citado no apêndice, que fisicamente vem depois da
bibliografia — correto no sistema numérico da ABNT).

**Todas as quatro tabelas de benchmark reproduzem os artefatos exatamente.**
Acurácia, EER, t-DCF*, AUC, F1, latência, parâmetros, tamanho e as quatro
colunas de robustez conferem dígito a dígito com `benchmark_summary.json`.

**Os testes pareados conferem integralmente** com `benchmark_significance.json`:
40/55 significativos por frase, 17/55 por locutor, piso de Holm 0,022
(= 55 × 0,00040), RawGAT-ST separado dos dez outros exatamente no piso,
ΔEER(AST × CCT) = +0,29 p.p. com IC95% [−0,30; +0,64] e p ajustado 1,00.

**Aritmética derivada, verificada:** 4 e 6 amostras mal classificadas em 1382;
26,85/0,92 = 29×; 85,5 M/650.204 = 131×; 42,3 h = 67% de 63,34 h; razão
t-DCF*/EER de 1,86 (Conformer) a 2,86 (AST) e a compressão de 82× → 76× no par
AST × RawGAT-ST; 12.162 → 24.324 com uma cópia ruidosa; 15 candidatos/75 ajustes
(SVM) e 108/540 (RF); 11+26+26+120 = 183 descritores, batendo com
`N_TABULAR_FEATURES_V2`; 37 endpoints, batendo com a contagem por router
(9+8+6+5+4+3+2); 56 locutores; mediana de 4,67 s; cabeça SSL de 393.999
parâmetros (≈ 394 mil) e 12+1 = 13 camadas ocultas; `backbone_trainable: false`.

**Qualidade argumentativa acima da média para um TCC** em três pontos
específicos: (i) a qualificação de que o oráculo majoritário é consequência
algébrica do pareamento e não evidência independente (linhas 1795–1810) — é a
leitura correta e raramente feita; (ii) a distinção entre score de ordenação e
probabilidade calibrada nos clássicos, com o custo de 0,75 p.p. de AUC
quantificado (linhas 1904–1917); (iii) o tratamento do t-DCF* como proxy
declarado, com a ressalva de que a formulação de 2019 não é comparável a
campanhas posteriores.

**A honestidade sobre limitações é apropriada** — gerador único, fonte única,
execução única, SSL só congelado, latências não comparáveis entre runtimes. O
item (vii) (artefato do RawGAT-ST anterior ao ajuste de regularização) é um
nível de transparência incomum.

---

## F. Ordem sugerida de correção

1. **A1** (uso do teste para decidir o front-end) — decide como o trabalho se
   apresenta; tudo o mais é subordinado.
2. **A2, A5, B1, B3, B4, B5** — erros pontuais, correção de minutos, alto custo
   se sobreviverem à banca.
3. **A3, A4** — exigem reescrever a Tabela `tab:representacoes_entrada` e o
   parágrafo do espectrograma, e acrescentar uma ressalva na análise.
4. **D1, D2** — reprodutibilidade e compilação; testar `pdflatex` antes.
5. **C1, C2** — ganho líquido: dois resultados fortes que já existem e não estão
   sendo usados.
6. **C3, C4, B2, B6, B7, C5, C6, D3–D9** — refinamento.
