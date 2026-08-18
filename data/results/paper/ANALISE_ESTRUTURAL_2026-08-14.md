# Análise estrutural, lógica e de rigor — `main.tex`

Segunda passagem sobre o TCC, agora com foco em **rigor matemático, lógica
argumentativa, taxonomia e estrutura**, e na pergunta de adequação a
apresentação e a publicação. A primeira passagem (`REVISAO_TECNICA_2026-08-14.md`)
tratou de fidelidade aos artefatos; esta trata da construção do texto.

Tudo o que está marcado como **verificado** foi recontado contra o código ou os
artefatos.

---

## 0. Veredito

**Para apresentação e defesa: adequado, com três reparos obrigatórios.** O
trabalho tem mérito real — protocolo pareado auditável, comparação estatística
pareada com correção de multiplicidade, honestidade sobre limitações. O que
precisa cair antes da banca são os itens B1, C1 e C2 abaixo, porque são pontos
em que um examinador atento derruba a alegação central com os dados do próprio
documento.

**Para publicação (conferência/periódico): não ainda.** Faltam quatro coisas
estruturais, na ordem: (i) validação *cross-generator*, que o próprio trabalho
identifica como a limitação dominante; (ii) reequilíbrio Metodologia/Resultados
(hoje 38,6% contra 22,8%); (iii) seção autônoma de trabalhos relacionados com
comparação sistemática; (iv) reconciliação das três taxonomias concorrentes.
Como está, o texto é um bom relatório técnico — descreve exaustivamente o que
foi feito — mas ainda não é um argumento científico enxuto em torno de uma tese.

---

## A. Rigor matemático

Auditei as 24 equações numeradas (16 em `equation` + 8 geradas pelos dois blocos
`align`). **A formulação matemática está correta em todas** — DCT-II ortonormal,
t-DCF normalizado da formulação de 2019, EER com a convenção de escada, CQT de
Brown, jitter/shimmer locais relativos, regressão de Furui. Verificações
pontuais:

- $Q = (2^{1/B}-1)^{-1}$ com $B=12$ dá $16{,}817$; o texto diz $\approx 16{,}8$. ✔
- $\alpha_c$ é de fato a normalização ortonormal da DCT-II, e a justificativa de
  isometria está correta. ✔
- $C_1, C_2$ conferem com a Eq. (7) de Kinnunen et al. (2018). ✔
- Os IC de Wilson conferem: recomputei para $n=1382$ e obtive
  $[99{,}26\%; 99{,}89\%]$ para o AST e $[85{,}71\%; 89{,}19\%]$ para o
  RawGAT-ST, exatamente os valores publicados. ✔ **verificado**

Há, porém, dois problemas sérios.

### A1. A janela da STFT está errada — e o valor publicado é o de um bug corrigido

Linha 1510: *"STFT [...] com janela de \num{512} amostras (n\_fft)"*.

O código não usa 512. `benchmark_frontend.py::resolve_n_fft` deriva a janela do
salto impondo `MIN_STFT_OVERLAP = 0.5`:

```
hop   = ceil(48000/100)        = 480 amostras (30 ms)
n_fft = 2^ceil(log2(480/0.5))  = 1024 amostras (64 ms)
sobreposição real              = 53,1%
```

Com os 512 que o texto afirma, a sobreposição seria de **6,2%** — e o comentário
do próprio código documenta essa configuração como um **ponto cego que foi
corrigido**, com a medição que motivou a correção. O texto, portanto, publica os
parâmetros da versão defeituosa do *front-end*, não da que produziu os
resultados. **Verificado por execução da função.**

Isso não é detalhe tipográfico: janela e salto determinam a resolução
tempo-frequência de três das quatro redes espectrais, e 6% de sobreposição
implica perda de informação entre quadros que 53% não tem. Corrigir para
$n_{\mathrm{fft}} = 1024$ e declarar a sobreposição resultante.

*(Observação secundária: o docstring do módulo, linha 9, também diz `n_fft=512`.
Está igualmente desatualizado e deve ser corrigido no código.)*

### A2. Duas convenções opostas de nomeação de erro convivem sem aviso

§4.5 declara: *"A classe positiva é spoof"*. Sob essa convenção:

| Evento | Nome em §4.5 / matrizes de confusão | Nome em §4.5.4 (t-DCF) e curva DET |
|---|---|---|
| Rejeitar um *bonafide* | **falso positivo** | $P_{\mathrm{miss}}^{\mathrm{cm}}$ (**miss**) |
| Aceitar um *spoof* | **falso negativo** | $P_{\mathrm{fa}}^{\mathrm{cm}}$ (**false alarm**) |

As duas convenções são **exatamente invertidas**, e o documento alterna entre
elas a poucos parágrafos de distância: §5.3 descreve *"66 amostras bonafide
rejeitadas"* como falsos positivos do AASIST e, ~15 linhas depois, chama o mesmo
tipo de erro de $P_{\mathrm{miss}} = 42{,}3\%$ na discussão da curva DET.

Ambas as definições estão individualmente corretas — a segunda é a convenção
ASV/t-DCF herdada do ASVspoof. O problema é que o texto nunca avisa que houve
troca de referencial. Um examinador que cruzar as duas passagens conclui, com
razão, que há contradição. **Correção:** uma tabela de correspondência em §4.5,
antes da primeira aparição do t-DCF.

---

## B. Rigor estatístico

### B1. A alegação central é feita sem o teste que o próprio trabalho estabeleceu como padrão

O trabalho monta um aparato estatístico exemplar para o conjunto limpo: McNemar
por *cluster*, *bootstrap* pareado, correção de Holm sobre 55 comparações, piso
de resolução declarado, duas unidades de agrupamento. Conclui, corretamente, que
o topo é empate técnico.

**Nada disso é aplicado ao ordenamento sob ruído** — que é o "achado central" do
trabalho. §5.3 apresenta a ordem a \SI{10}{\decibel} como *ranking* simples:
*"lidera o Res2Net (92,40%), seguido de AASIST (91,68%), Conformer (91,46%),
RawNet2 (90,38%) e CCT (90,30%)"*, sem um único intervalo.

Os intervalos existem nos artefatos e nunca foram usados
(`robustness["10"].accuracy_ci95_*`, 1000 reamostragens por *cluster*):

| Modelo | Acc @ 10 dB | IC 95% |
|---|---|---|
| Res2Net | 92,40% | [90,93; 93,78] |
| AASIST | 91,68% | [90,30; 92,89] |
| Conformer | 91,46% | [90,16; 92,82] |
| RawNet2 | 90,38% | [89,00; 91,79] |
| CCT | 90,30% | [88,79; 91,73] |

**Os dez pares entre os cinco primeiros têm intervalos que se sobrepõem.**
**Verificado.**

Sejamos precisos sobre o que isso significa: sobreposição de IC marginais **não
prova** ausência de diferença em dados pareados — um teste pareado tem muito
mais poder, e os modelos são avaliados nas mesmas amostras. A crítica correta é
que **o ônus da prova não foi cumprido**: o trabalho aplicou o teste pareado
onde a diferença era pequena (limpo) e não o aplicou onde faz a sua afirmação
mais forte (ruído). A assimetria enfraquece justamente o capítulo que sustenta
a tese.

**Correção:** rodar McNemar/*bootstrap* pareado a \SI{10}{\decibel} e a
\SI{5}{\decibel} — a infraestrutura já existe em `benchmarks/significance.py`,
basta alimentá-la com as predições ruidosas. É provável que a conclusão
qualitativa sobreviva (RawGAT-ST separa-se de todos; AST e CCT saem do topo),
mas com a força que hoje falta.

### B2. "±0,31 ponto percentual" descreve um intervalo que não é simétrico

§5.3 apresenta o IC de Wilson do AST como *"±0,31 ponto percentual no topo
(99,71% → [99,26%; 99,89%])"*. O intervalo de Wilson **não é centrado na
proporção observada**: em torno de 99,71% ele é $-0{,}45/+0{,}18$. O valor
$0{,}31$ é a semilargura em torno do centro de Wilson (99,58%), não em torno de
99,71%. Escrever "±0,31" ao lado de 99,71% sugere simetria que não existe.
Basta suprimir a notação "±" e reportar o intervalo.

---

## C. Lógica e desenvolvimento argumentativo

### C1. A explicação causal do achado central é refutada pelos dados do próprio trabalho

Linhas 2554–2558: *"Nenhuma das duas variáveis ordena o recorte; **o
diagnóstico de estabilidade de treino, sim**. [...] a diferença entre um
detector robusto e um frágil manifestou-se na regularização, e não na escolha
arquitetural."*

O diagnóstico automático classifica **dois** modelos como
`unstable_oscillation`. Eis onde eles caem na robustez a \SI{10}{\decibel}:

| Modelo | `training_stability.status` | Acc @ 10 dB | Posição |
|---|---|---|---|
| **Conformer** | `unstable_oscillation` | **91,46%** | **3.º** |
| RawGAT-ST | `unstable_oscillation` | 77,21% | 11.º |

**Verificado nos artefatos.** Dos dois modelos instáveis, um é o terceiro mais
robusto do recorte, à frente de seis modelos estáveis. A variável "estabilidade"
não ordena coisa alguma — ela tem duas observações positivas e uma delas é
contraexemplo direto.

O parágrafo é internamente irônico: ele usa o Conformer como evidência para
rejeitar a hipótese de capacidade (*"o maior modelo treinado do zero fica entre
eles em robustez"*) e, no período seguinte, adota uma explicação que o próprio
Conformer refuta.

**Correção honesta**, que continua sendo um resultado interessante: *nenhuma*
das variáveis examinadas — representação de entrada, capacidade, diagnóstico
automático de estabilidade — ordena a robustez. O contraste AASIST × RawGAT-ST é
uma **observação em $n=2$ compatível com sobreajuste**, não evidência de lei
geral. Um achado negativo bem delimitado é publicável; uma explicação causal
refutada pela própria tabela, não.

### C2. A previsão teórica da §2.3 nunca é confrontada com os resultados

§2.3 monta uma taxonomia de quatro artefatos de síntese e faz uma previsão
explícita e falseável:

> *"Como a representação Mel de magnitude descarta a fase, esses artefatos são
> invisíveis às redes espectrais e só alcançáveis por arquiteturas que operam
> sobre a forma de onda --- RawNet2, AASIST e RawGAT-ST"*

Os resultados vão na direção oposta: as três redes de forma de onda ocupam a
6.ª, 7.ª e 11.ª posições em acurácia limpa, atrás de todas as redes espectrais.
Se os artefatos de fase fossem determinantes e invisíveis ao domínio espectral,
essa ordem seria impossível.

**O trabalho nunca volta a essa previsão.** É um arco argumentativo aberto na
fundamentação e abandonado — precisamente o tipo de coisa que uma banca cobra.
Duas saídas legítimas: (a) retomar em §5.3 e concluir que, neste *corpus*, os
artefatos de envoltória dominam os de fase, o que é um resultado; ou (b)
enfraquecer a previsão em §2.3 para uma hipótese, sinalizando onde será testada.

### C3. QP2 é apresentada como aberta *a posteriori*

§1.3: *"A formulação é deliberadamente aberta quanto aos fatores: as hipóteses
inicialmente mais plausíveis [...] são testadas, e não pressupostas"*.

A cláusula foi visivelmente escrita depois de saber que as hipóteses caíram.
Não há problema em testar e rejeitar hipóteses — é o melhor que uma pesquisa
empírica faz. Há problema em reescrever a pergunta para que a rejeição pareça
antecipada. O texto fica mais forte declarando: *"partiu-se das duas hipóteses
mais plausíveis; ambas foram rejeitadas pelos dados, e a Seção 5.3 discute o
que resta"*.

### C4. A montagem da assimetria de custo entrega um resultado nulo

§1.1 dedica três parágrafos a estabelecer que $C_{\mathrm{fa}} = 10 \gg
C_{\mathrm{miss}} = 1$ *"orienta toda a leitura dos resultados adiante"*. §5.3
conclui que o $t$-DCF\* **não reordena nada** (única troca: SVM × RF, que têm
EER idêntico).

O resultado nulo é reportado com honestidade e é informativo. Mas a promessa da
introdução é grande demais para o que se entrega. Calibrar §1.1: a assimetria
importa para escolher o *ponto de operação*, não para escolher o detector — que
é, aliás, exatamente a conclusão a que §5.3 chega.

---

## D. Taxonomia

O documento opera **quatro classificações simultâneas e mutuamente
irreconciliadas**:

| # | Onde | Eixo | Categorias |
|---|---|---|---|
| 1 | Tab. gerações (§2.3) | histórico | 4 gerações |
| 2 | §3 (organização do capítulo) | família | SSL+áudio bruto / espectro-temporal / clássicos |
| 3 | Tab. representações (§4.3) | entrada | raw / mel 100×80 / mel 300×128 / tabular |
| 4 | §2.3 (lista de artefatos) | fenômeno | fase / envoltória / dinâmica / prosódia |

Nenhuma mapeia sobre a outra, e o texto nunca as concilia. Três consequências
concretas:

### D1. Erro taxonômico na tabela de gerações

A 4.ª geração é rotulada *"Representação SSL e atenção"* e lista **WavLM,
HuBERT, AST**. O AST **não é auto-supervisionado** — é ViT com pré-treinamento
**supervisionado** em AudioSet, como o próprio §3.2 afirma corretamente
(*"pesos AudioSet transferidos"*). Classificá-lo como SSL contradiz o resto do
documento e confunde os dois regimes de pré-treinamento cuja diferença é
material para as conclusões.

### D2. A família "SSL e áudio bruto" mistura dois eixos independentes

§3.1 agrupa WavLM/HuBERT (pré-treinados, ***backbone* congelado**, cabeça rasa
treinada) com RawNet2/AASIST/RawGAT-ST (**treinados do zero**, fim a fim). O
único traço comum é o formato da entrada; o regime de treinamento — que é
exatamente o que a conclusão discute ao explicar por que os SSL ficam na metade
inferior — some dentro da categoria.

**Correção:** taxonomia bidimensional (representação de entrada × regime de
treinamento), que separa naturalmente os cinco grupos reais: raw treinado do
zero, raw congelado pré-treinado, espectral do zero, espectral pré-treinado
(AST, sozinho) e tabular. Isso também torna visível, sem precisar de caixa de
aviso, que o AST é o único do seu grupo.

### D3. O nome "Res2Net" carrega alegação de fidelidade que a tabela contradiz

Res2Net (Gao et al., 2021) aparece na 2.ª geração (2016–2019) pela lógica do
texto de §2.3, embora seja de 2021 e esteja na tabela sob "Espectrograma Mel /
CNN, LSTM, LCNN". Menor que D1, mas na mesma família de imprecisão.

---

## E. Estrutura e organização

### E1. O documento está invertido em relação ao seu conteúdo científico

**Verificado por contagem de linhas do corpo (2801 linhas):**

| Seção | Extensão | % do corpo |
|---|---:|---:|
| Introdução | 179 | 6,4% |
| Tecnologias de Síntese | 258 | 9,2% |
| Arquiteturas | 240 | 8,6% |
| **Metodologia** | **1081** | **38,6%** |
| **Experimentos e Resultados** | **638** | **22,8%** |
| Sistema Desenvolvido | 147 | 5,2% |
| **Conclusões** | **621** | **22,2%** |

A Metodologia tem **1,7× a extensão dos Resultados**, e as Conclusões quase
igualam os Resultados. Num relatório técnico isso é aceitável; num trabalho
científico é inversão de prioridade — o leitor atravessa 1081 linhas de
aparato antes do primeiro número.

### E2. Quase metade das equações descreve descritores que o trabalho não usa

A subseção *"Descritores Adicionais Disponíveis no Pipeline"* (§4.3, ~150
linhas) apresenta **11 equações numeradas** — 5 de forma espectral, 2 de CQT, 3
prosódicas, 1 de delta — para descritores que o próprio texto declara *"não
consumidos pelos 11 modelos deste trabalho"*. São ~46% de todo o aparato
matemático do documento dedicado a material não utilizado.

É também a origem dos quatro rótulos órfãos (`eq:cqt`, `eq:cqt_q`, `eq:delta`,
`eq:zscore`): equações numeradas que nenhum texto cita. **Verificado.**

**Correção:** mover integralmente para apêndice. Isso sozinho reequilibra
Metodologia de 38,6% para ~33% e remove a impressão de preenchimento.

### E3. Trinta subseções não aparecem no sumário

O documento usa `\subsubsection*` (com asterisco, portanto não numerada e
**ausente do sumário**) em 30 lugares: **11** descrições de arquitetura, **9**
na Metodologia e **10** na Análise dos Resultados. **Verificado.**

Consequência prática: num documento de 81 páginas, toda a análise dos
resultados aparece no sumário como uma única linha ("Análise dos Resultados"),
escondendo dez subseções — incluindo "Robustez a Ruído: o Achado Central". Uma
banca que queira ir direto ao achado central não o encontra no sumário.

**Correção:** numerar (remover o asterisco) ao menos as dez de §5.3 e as onze
de §3.

### E4. A Seção 6 quebra a ordem IMRAD

"Sistema Desenvolvido" está **entre os Resultados e as Conclusões**. Não
apresenta medição nova; descreve um artefato de engenharia. A justificativa
dada (*"fecha o percurso que a QP3 abre"*) não se sustenta: a QP3 pergunta por
custo computacional, e §6 não traz nenhum.

**Correção:** mover para o fim da Metodologia (é parte do método) ou para
apêndice (é implementação). Numa submissão a periódico, apêndice.

### E5. Não há seção autônoma de trabalhos relacionados

O posicionamento na literatura ocupa 30 linhas (§2.4), dentro de um capítulo
intitulado "Tecnologias de Síntese de Voz" — cujo título, aliás, não cobre nem
§2.3 (detecção) nem §2.4 (posicionamento). Para publicação, é insuficiente:
falta comparação sistemática com os *benchmarks* de anti-*spoofing* existentes,
em tabela, explicitando o que este protocolo faz que os outros não fazem.

### E6. Conclusões repetem os Resultados em vez de sintetizá-los

As 621 linhas de conclusão reapresentam praticamente todos os números já dados
em §5.3, muitas vezes na mesma ordem. Uma conclusão de trabalho científico
deve responder às QPs e declarar o que se aprendeu, remetendo aos números em
vez de reimprimi-los. Alvo razoável: ~250 linhas.

---

## F. Melhorias priorizadas

### Obrigatório antes da defesa
1. **C1** — reescrever a explicação causal do achado central; o Conformer a
   refuta. *(maior risco de arguição)*
2. **A1** — corrigir $n_{\mathrm{fft}}$ de 512 para 1024 e declarar 53% de
   sobreposição; corrigir também o docstring do módulo.
3. **B1** — rodar os testes pareados a 10 e \SI{5}{\decibel}, ou rebaixar
   explicitamente as afirmações de ordenamento sob ruído.
4. **C2** — retomar em §5.3 a previsão de artefatos de fase feita em §2.3.
5. **A2** — tabela de correspondência entre as duas convenções de erro.
6. **D1** — retirar o AST da linha "Representação SSL".

### Alto retorno, baixo custo
7. **E3** — numerar as subseções de §3 e §5.3 (remoção de 21 asteriscos).
8. **E2** — mover os descritores não usados para apêndice.
9. **B2** — remover a notação "±" do IC de Wilson.
10. **C3** — declarar que as hipóteses da QP2 foram rejeitadas, sem a moldura
    de "abertura deliberada".
11. **C4** — calibrar a promessa de §1.1 ao resultado nulo de §5.3.

### Necessário para publicação
12. **E1/E6** — reequilibrar: Metodologia para ~30%, Conclusões para ~10%,
    Resultados para ~35%.
13. **E4** — §6 para apêndice.
14. **E5** — seção autônoma de trabalhos relacionados, com tabela comparativa.
15. **D2** — taxonomia bidimensional (representação × regime de treinamento).
16. **Validação *cross-generator*** — sem ela, a contribuição permanece
    "desempenho contra o XTTS-v2", como o próprio trabalho reconhece.

---

## G. O que está bem feito

Registro para equilíbrio, e porque nenhum destes pontos deve ser mexido:

- **O aparato estatístico do conjunto limpo** é superior ao de boa parte da
  literatura de TCC e de muitos artigos: McNemar por *cluster*, *bootstrap*
  pareado, Holm sobre 55 comparações, duas unidades de agrupamento, piso de
  resolução declarado. Só falta estendê-lo ao ruído (B1).
- **A qualificação do oráculo majoritário** como consequência algébrica do
  pareamento, e não evidência independente, é a leitura correta e rara.
- **A separação entre score de ordenação e probabilidade calibrada**, com o
  custo de 0,75 p.p. de AUC quantificado, demonstra entendimento fino do que
  cada métrica mede.
- **A formulação matemática está correta em todas as 24 equações.**
- **O tratamento das limitações** — gerador único, execução única, SSL apenas
  congelado, t-DCF como *proxy*, latências não comparáveis — é honesto e
  completo.
- **Integridade de referências**: 0 `\ref` órfão, 0 `\cite` sem `\bibitem`,
  bibliografia em ordem de citação conforme o sistema numérico da ABNT.
