# Protocolo de dataset

Dataset canônico do XFakeSong para detecção de deepfake de áudio em português:
**CETUC pareado com clones XTTS-v2**, com disjunção dupla de locutor e frase
entre treino, validação e teste.

Documento único e definitivo. Cada decisão vem acompanhada da medição que a
sustenta — o que passou no acaso e não exigiu nada, o que reprovou e obrigou a
mudar a construção, e o que permanece como limite declarado.

## 1. O artefato

| Propriedade | Valor verificado |
| --- | --- |
| Corpus | `data/datasets/corpus/` |
| Pares (bonafide + clone do mesmo enunciado) | **49.264** |
| Amostras no corpus | **98.528** — 49.264 reais + 49.264 falsas |
| Locutores | **56**, todos nas duas classes (37 F / 19 M) |
| Frases | 1000 slots → **997 grupos de conteúdo**; **zero** divergência de texto entre locutores |
| Duração total | 133,1 h |
| Cobertura por locutor | 218 a 1000 pares (limitada pelo upstream) |
| Áudio | mono, 16 kHz, PCM_16, RMS −26 dBFS |
| Partição | `data/datasets/splits/` |

### O NPZ do benchmark

| Propriedade | Valor |
| --- | --- |
| Arquivo | `data/datasets/benchmark_dataset.npz` |
| Amostras | 40.980 — 20.490 reais + 20.490 falsas |
| Entrada | áudio bruto `(48000, 1)`, mono, 16 kHz, **3 s** |
| Tamanho | 7,99 GB (sem compressão) |
| SHA-256 | `ae3662c9e2cc904c9c2f06bc541f27fe944baea4a527591cdb127a1d00f62ad7` |
| Clusteres (`text_id`) | 997 |
| Selo do teste | `benchmark_dataset.npz.test-lock.json`, criado antes de qualquer treino |

### A partição (semente 42)

| Split | Amostras | Reais | Falsas | Locutores | Frases | Horas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| treino | 33.226 | 16.613 | 16.613 | 34 (23F/11M) | 602 | 45,47 |
| validação | 3.976 | 1.988 | 1.988 | 11 (7F/4M) | 201 | 5,49 |
| teste | 3.778 | 1.889 | 1.889 | 11 (7F/4M) | 197 | 5,17 |

Aproveitamento: **40.980 de 98.528 (41,6%)**. O que sai, sai por construção:

| Motivo | Amostras |
| --- | ---: |
| fora da diagonal (locutor de uma partição lendo frase de outra) | 52.968 |
| mais curtas que a janela de 3 s (o par inteiro; §5.5) | 4.460 |
| áudio duplicado na fonte (o par inteiro; §4.3) | 120 |

O treino tem 602 slots de frase para 600 grupos porque três textos aparecem em
dois slots cada e o grupo os mantém juntos (§4.2).

## 2. As duas fontes

| Papel | Fonte | Revisão fixada | Formato nativo |
| --- | --- | --- | --- |
| bonafide | [`falabrasil/cetuc`](https://huggingface.co/datasets/falabrasil/cetuc) | `6dbc8081d35fff5bdd133d8a12b8a11f91959b01` | 16 kHz, PCM_16, mono |
| spoof | [`unfake/fake_voices`](https://huggingface.co/datasets/unfake/fake_voices) | `541bf396da524f92a6d6594a0e9952210e7d7e7e` | 24 kHz, FLOAT, mono |

O card do Fake Voices declara: os deepfakes foram gerados com **XTTS**
([Casanova et al., 2024](#ref-xtts)) alimentado com **gravações do CETUC**. As
duas fontes são, portanto, o mesmo material de fala em duas condições — bonafide
e clone —, não dois corpora independentes.

Três propriedades decorrem disso:

1. **Identidade de locutor publicada.** O CETUC nomeia cada locutor
   (`Aislam_M001`) e o Fake Voices reusa o mesmo código (`Aislam_M001_Fake`). A
   disjunção é **verificável**, não inferida por embeddings.
2. **Identidade de texto publicada.** Cada WAV do CETUC vem com a transcrição ao
   lado. Verificado nos 56 locutores: **todos leem as MESMAS 1000 frases**
   foneticamente balanceadas, e o texto do índice `i` é idêntico entre eles —
   zero divergência.
3. **Partição de locutores definida pelos autores.** O CETUC vem em
   `data/{train,dev,test}/`, um tar.gz por locutor em exatamente uma partição,
   disponível como alternativa (§4.1).

### 2.1 Alinhamento enunciado a enunciado (verificado, não suposto)

O CETUC indexa de `0000` a `0999`; o Fake Voices, de `1_fake.wav` a
`1000_fake.wav`. O alinhamento `N_fake.wav → {N-1:04d}` foi verificado
estatisticamente sobre os 1000 enunciados de `M001`:

| Offset testado | corr(duração real, duração clone) | corr(tamanho do texto, duração clone) |
| --- | ---: | ---: |
| `N` → `N` | 0,010 | 0,001 |
| **`N` → `N-1`** | **0,507** | **0,675** |
| `N` → `N+1` | −0,010 | −0,032 |

O tamanho do texto só pode correlacionar com a duração do clone se o clone
estiver lendo aquele texto. O offset está determinado.

### 2.2 Os 45 locutores não pareados ficam fora

Dos 101 locutores do CETUC, **56** têm clone. Os 45 restantes **não entram**:
como existiriam só na classe real, bastaria reconhecer o locutor para acertar o
rótulo. Esse é o modo de falha clássico de aprendizado por atalho
([Geirhos et al., 2020](#ref-geirhos); [Torralba & Efros, 2011](#ref-torralba)),
que em detecção de deepfake se manifesta como dependência de pistas de canal e
corpus em vez de artefatos de síntese ([Müller et al., 2022](#ref-muller)).

Eles ficam disponíveis como conjunto real-only para calibração e robustez.

## 3. O desenho: grid completo locutor × frase × classe

O corpus é o produto cartesiano

    56 locutores  ×  1000 frases  ×  {bonafide, spoof}

com a regra de construção: **o par só entra quando as DUAS amostras passam na
validação**. Disso decorre, sem cota nenhuma:

- **balanceamento exato de classe** — por partição *e por locutor*;
- **locutor não prediz a classe** — todo locutor aparece nas duas classes;
- **frase não prediz a classe** — toda frase aparece nas duas classes.

A cobertura do Fake Voices varia por locutor (entre 218 e 1000 clones); o
pareamento estrito absorve a variação descartando o lado sem par, em vez de
deixar o corpus desbalancear.

### 3.1 Nomenclatura e o campo `source`

Os arquivos das duas classes usam o **mesmo prefixo**:
`ptpair_<CÓDIGO>_<NNNN>_<gerador>.wav`.

Isso é deliberado. O exportador deriva `source_ids` do prefixo do nome quando o
manifesto não diz outra coisa, e falha de propósito quando a regra "fonte →
classe majoritária" acerta mais de 55% dos rótulos. Um prefixo por classe
(`cetuc_` real, `xtts_` fake) daria um oráculo de 100%.

O ponto não é contornar a guarda: é que **`source` deve nomear a origem do
material de fala**, e aqui ela é a mesma nas duas classes — mesmos locutores,
mesmas frases, o clone gerado a partir da gravação. O repositório de aquisição
(que difere) fica em `acquisition_repo` por amostra, e o gerador em
`generator_id`. Um auditor pode recalcular o oráculo por repositório (100% por
construção, artefato de empacotamento) e por fonte de fala (50%).

Como a guarda por metadado deixa de ser informativa nesse ponto, ela é
**substituída por medição direta** do sinal (§5).

## 4. Anti-contaminação: disjunção dupla por bloco diagonal

A partição é o bloco diagonal do grid:

    split(amostra) = S   ⟺   locutor(amostra) ∈ S_loc   E   frase(amostra) ∈ S_frase

| Dimensão | Partição |
| --- | --- |
| locutor | 60/20/20 dos 56 pareados, estratificado por sexo |
| frase | 600 / 200 / resto, dos grupos de conteúdo |

Amostras **fora da diagonal** — locutor de treino lendo frase de teste, e as
outras cinco combinações — são **excluídas**. Não existe forma de aproveitá-las
sem violar uma das duas disjunções; o custo em cobertura está declarado no
`split_manifest.json` e em §1.

Splits agrupados são necessários exatamente quando as amostras não são
independentes ([Roberts et al., 2017](#ref-roberts)); ignorar a estrutura de
grupo é uma forma clássica de vazamento ([Kaufman et al., 2012](#ref-kaufman)).
Aqui existem **duas** estruturas de grupo, e as duas são respeitadas.

### 4.1 Estratificar locutores, e por quê

O padrão (`--speaker-strategy stratified`) reparte os 56 em 60/20/20
**estratificando por sexo** — o código do CETUC já traz o sexo (`F049`, `M001`).
Sem estratificar, um sorteio de 11 locutores pode entregar um conjunto de teste
quase todo de um sexo, e a métrica passaria a medir também a diferença entre
vozes graves e agudas.

A partição publicada pelo CETUC está disponível em `--speaker-strategy official`.
Ela é mais defensável por vir dos autores, mas não é o padrão porque foi
desenhada para ASR sobre os 101 locutores: recortada aos 56 que têm clone, vira
**44/5/7**, e cinco locutores na validação não sustentam uma estimativa — um
único locutor atípico move a métrica inteira.

As duas são disjuntas por locutor e ficam registradas em `split_manifest.json`
(`speaker_partition_strategy` e o mapa completo locutor → partição).

### 4.2 A unidade de conteúdo é a componente conexa `índice ↔ texto`

Agrupar pelo `sentence_index` não basta: **medido no corpus**, a lista de 1000
frases do CETUC **repete algumas** — 3 textos aparecem em dois slots diferentes
(997 textos distintos para 1000 slots). Particionar por slot deixaria o mesmo
texto em treino e em teste.

Agrupar só pelo `text_id` também não serve: se um locutor divergir do texto
canônico de um slot, o slot ganharia dois textos e se dividiria entre partições.

A unidade correta é a **componente conexa** do grafo que liga cada slot ao seu
texto e cada texto ao seu slot. Ela cobre os dois casos. O `text_id` é derivado
do hash do texto normalizado (não do índice), o que torna a divergência
detectável em vez de silenciosa.

### 4.3 Gravações duplicadas na fonte

A auditoria de SHA-256 encontrou **gravações repetidas dentro do próprio CETUC**:
o WAV do índice `N` reaparece como índice `N+1` — mesmo locutor, mesma classe,
bytes idênticos, **transcrições diferentes**. Uma das duas transcrições está
necessariamente errada, e não há como saber qual sem reconhecimento de fala.

Os dois enunciados são **excluídos da partição**, com os clones correspondentes
(`excluded_duplicate_audio`). Manter um seria escolher arbitrariamente entre duas
transcrições das quais uma está errada. O corpus em disco preserva os arquivos —
o defeito é da fonte e fica registrado —, mas eles não entram em treino,
validação nem teste.

## 5. Confundidores medidos no sinal (não presumidos)

Assimetrias entre as fontes poderiam separar as classes sem que nenhum artefato
de síntese fosse detectado. Cada uma foi **medida** com AUC de descritor único, e
tratada conforme a medição — não conforme a intuição.

### 5.1 Nível de áudio: o atalho mais grave, corrigido

Com a política de amplitude convencional ("preservar loudness, atenuar apenas
clipping"), o RMS sozinho separava as classes com **AUC 0,9926** — um detector
acertaria quase tudo medindo volume, sem detectar síntese alguma.

Medido em 400 pares e 300 amostras por classe:

| Medida | bonafide (CETUC) | spoof (XTTS) |
| --- | ---: | ---: |
| RMS mediano | −27,44 dB | **−17,24 dB** |
| desvio-padrão do RMS | 4,85 dB | **0,78 dB** |
| pico mediano | −9,80 dB | −1,56 dB |

O desvio de **0,78 dB** revela a causa: o pipeline de geração normalizou o nível
de todo clone. O CETUC, sendo gravação, mantém a variação natural de 4,85 dB. Não
é característica da síntese — é do empacotamento.

**Correção adotada:** normalização de RMS para **−26 dBFS nas duas classes**, com
teto de pico em −1 dBFS, e o ganho aplicado registrado por amostra em
`applied_gain_db` (a operação é reversível):

| Descritor | AUC antes | AUC depois |
| --- | ---: | ---: |
| `rms_db` | 0,9926 | **0,5538** |
| `peak_db` | 0,9169 | 0,7888 |

**Normalizar o arquivo não basta — a janela também precisa.** Com todos os
arquivos em exatamente −26 dBFS, o RMS medido nos 3 s centrais ainda separava as
classes com **AUC 0,7097** (−25,38 dB no bonafide contra −26,10 no clone). A
gravação do CETUC é mais longa e carrega mais silêncio nas pontas, então seu
miolo é mais denso em fala do que a média do arquivo; o clone, mais curto e
uniforme, tem miolo parecido com a média.

Como o modelo recebe a **janela**, é a janela que precisa estar nivelada. A
política está em `extract_window`, usada tanto pelo exportador quanto pela
auditoria — o que a auditoria mede é literalmente o que entra no modelo.

### 5.2 Fator de crista: artefato genuíno, declarado

Depois da normalização resta um descritor **invariante à escala** que separa as
classes: o fator de crista (pico/RMS), com **AUC 0,67** na janela de 3 s —
16,30 dB no bonafide contra 15,18 dB no clone. Nenhuma normalização de nível
altera isso.

A atribuição é sustentada por evidência: se o pipeline de geração tivesse
aplicado um **limitador**, os picos ficariam grudados num teto. Não ficam —
apenas 3,0% dos clones passam de −0,5 dBFS, contra 3,3% do bonafide, e o desvio
do pico é de 0,85 dB. O que foi normalizado foi o **nível**, não o pico. Logo a
menor faixa dinâmica é propriedade da forma de onda gerada — a suavização de
transientes típica de vocoder —, não do tratamento de nível.

É, portanto, um **artefato de síntese legítimo**: o tipo de pista que um detector
deve usar. Fica declarado porque é o teto de um detector trivial neste corpus.

### 5.3 Como a auditoria trata as categorias

| Categoria | Descritores | Critério |
| --- | --- | --- |
| **empacotamento** | `rms_db` | **reprova** acima de AUC 0,60 — nível absoluto não carrega informação sobre síntese |
| **duração** | `duration_sec` | apenas reportada; o critério é a **taxa de repetição**, que deve diferir menos de **0,02** entre as classes (§5.5) |
| **sinal** | crista, pico, ZCR, centroide, rolloff, flatness, banda 7–8 kHz | apenas reportados — são o que um detector legitimamente usa |

Reprovar por descritor de sinal seria exigir que a síntese fosse indetectável;
não reprovar por descritor de empacotamento foi o que quase deixou passar o
atalho de nível de 0,99. Todos são medidos **na janela normalizada** que o modelo
recebe.

### 5.4 Banda e reamostragem (24 kHz → 16 kHz)

O clone é nativo em 24 kHz e o bonafide em 16 kHz. A reamostragem deixa uma
assinatura de filtro anti-aliasing perto de 8 kHz — se ela separasse as classes,
o modelo aprenderia taxa de amostragem.

| Cadeia | AUC banda 7,0–7,9 kHz | AUC banda 6,0–7,0 kHz |
| --- | ---: | ---: |
| **A. real nativo 16k / fake 24k→16k (adotada)** | **0,5014** | 0,5704 |
| B. real 16k→24k→16k / fake 24k→16k | 0,5125 | 0,5712 |
| C. A + passa-baixas 7 kHz nas duas classes | 0,5570 | 0,5713 |

A assinatura **não existe**: 0,5014 é o acaso. O XTTS herda a banda do material
CETUC de 16 kHz que o condicionou, então não há energia acima de 8 kHz para a
reamostragem cortar. As mitigações B e C não melhoram nada e C piora — foram
descartadas, e a cadeia simples é a adotada. O 0,57 em 6–7 kHz é fraco e
plausivelmente artefato de síntese genuíno (perda de detalhe espectral do
vocoder), não de canal: nenhuma mitigação de canal o reduz.

### 5.5 Duração e política de janela

Lendo o MESMO texto, o XTTS fala mais rápido: mediana de 4,41 s contra 4,90 s do
bonafide (AUC de duração 0,64). Isso é prosódia genuína do gerador.

**O problema.** Com janela fixa, quem não alcança a janela é repetido (`tile`) e
quem passa é recortado. Como o clone é sistematicamente mais curto, com janela de
5 s as taxas de repetição ficavam **0,510 no bonafide contra 0,700 no clone** —
diferença de 0,19. A emenda da repetição é um artefato *nosso*, introduzido pela
política de janela, e estava correlacionado com a classe.

**A correção.** A janela é de **3 s** e o par é descartado quando qualquer um dos
dois lados não a alcança (`excluded_shorter_than_window`). Resultado: **nenhuma
amostra é repetida** — toda amostra é recorte central puro do mesmo tamanho — e a
duração deixa de ser visível para o modelo.

| Janela | Pares mantidos | Repetição no bonafide | Repetição no clone |
| ---: | ---: | ---: | ---: |
| 2,5 s | 99,5% | 0,001 | 0,003 |
| **3,0 s** | **95,2%** | **0** (por construção) | **0** |
| 4,0 s | 58,6% | 0,197 | 0,346 |
| 5,0 s | 20,4% | 0,534 | 0,710 |

Custo: 4,8% dos pares. Em troca, o critério da auditoria pôde ficar muito mais
severo — a diferença de taxa de repetição permitida é de **0,02**.

Por isso a duração **não** é tratada como confundidor de empacotamento (a AUC de
0,64 não reprova): ela é prosódia real do gerador, e o que precisa ser verificado
é se ela **chega** ao modelo. Com repetição zerada, não chega. O critério é o
mecanismo, não a proxy.

## 6. Áudio canônico

| Etapa | Política |
| --- | --- |
| Decodificação | mono, `float32` |
| Reamostragem | `soxr_hq` → 16 kHz (no-op no bonafide) |
| Duração válida | 1 a 30 s no corpus; o par só entra na partição se os DOIS lados alcançarem a janela |
| VAD | desativado |
| **Amplitude** | **RMS normalizado para −26 dBFS nas duas classes**, teto de pico −1 dBFS, ganho registrado em `applied_gain_db` (§5.1) |
| Gravação | WAV PCM_16, 16 kHz, mono — idêntico nas duas classes |
| **Janela exportada** | **3 s (48.000 amostras), recorte central puro — nenhuma amostra repetida** (§5.5) |

## 7. Garantias verificadas no artefato

A auditoria roda sobre o artefato, não sobre o plano, e falha com código ≠ 0
quando uma garantia é violada — pode ir para CI antes de qualquer treino:

```bash
python scripts/dataset/audit_paired_corpus.py
```

| Bloco | O que verifica |
| --- | --- |
| A | todo bonafide tem o seu clone do mesmo enunciado; zero duplicata exata **na partição**; zero arquivo ausente; um `text_id` por slot de frase |
| B | locutor, frase, texto, enunciado e hash de conteúdo **não atravessam partições** |
| C | 50/50 por partição **e por locutor** |
| D | oráculo de maioria por `source`, `speaker_id`, `sentence_index`, `text_id` ≤ 55% |
| E | AUC de descritor único na janela que o modelo recebe, com reprovação no de empacotamento (RMS ≤ 0,60) e na diferença de taxa de repetição (≤ 0,02) |
| F | quase-duplicatas por impressão espectral (cosseno ≥ 0,99): **F1** redundância interna e **F2** vazamento entre partições, comparando todos os pares de partições diferentes |

### Resultado — todas as garantias verificadas

| Bloco | Medido |
| --- | --- |
| A | 98.528 amostras, 49.264 enunciados, **0 sem par**, **0 arquivo ausente**; 30 grupos de duplicata exata no corpus → **0 na partição**; **0/1000** slots com texto divergente |
| B | locutor, frase, texto, enunciado e hash: **0 sobreposição** nos três pares de partições |
| C | 16.613/16.613, 1.988/1.988, 1.889/1.889 — **0 locutores desbalanceados** |
| D | `source` 50,00% · `speaker_id` 50,00% · `sentence_index` 50,00% · `text_id` 50,00% |
| E | empacotamento `rms_db` **0,5744** (limite 0,60); **taxa de repetição 0,000 nas duas classes**; pior descritor de sinal `crest_db` 0,6676 |
| F | F1: **8 pares** de gêmea interna em 40.980 amostras (0,04%). F2: **0 pares** acima de 0,99 entre partições em **272,7 milhões** de comparações (maior similaridade observada 0,9838) |

#### Descritores de sinal, na janela que o modelo recebe

| Descritor | AUC | bonafide | spoof |
| --- | ---: | ---: | ---: |
| `crest_db` | 0,6676 | 16,30 dB | 15,18 dB |
| `peak_db` | 0,6676 | −9,70 dB | −10,82 dB |
| `duration_sec` (não chega ao modelo) | 0,6477 | 5,23 s | 4,67 s |
| `rms_db` (empacotamento) | 0,5744 | −26,000 dB | −26,002 dB |
| `band_7k_8k_rel_db` | 0,5442 | −34,15 dB | −32,48 dB |
| `zcr` | 0,5429 | 0,093 | 0,098 |
| `spectral_centroid_hz` | 0,5261 | 552,5 | 517,9 |
| `spectral_rolloff95_hz` | 0,5154 | 1494,0 | 1473,0 |
| `spectral_flatness` | 0,5077 | 0,013 | 0,012 |

O teto de um detector trivial de descritor único é **0,67**, e vem do fator de
crista — artefato de síntese, não de canal (§5.2). Todo o resto está perto do
acaso.

## 8. Limites residuais, declarados

- **Gerador único.** Toda a classe spoof é XTTS-v2. O corpus mede comparação
  controlada *in-domain* e **não** demonstra generalização para geradores não
  vistos. Esta é a limitação dominante, e a prática recomendada após
  [Müller et al., 2022](#ref-muller) é reservar um conjunto externo com outro
  gerador. Extensão natural: ressíntese por vocoder do próprio bonafide (mesmo
  locutor, mesmo texto, ataque diferente), que preserva o pareamento e adiciona
  um eixo de ataque não visto sem introduzir atalho de locutor.
- **Faixa dinâmica.** O fator de crista separa as classes com AUC 0,67 (§5.2). A
  evidência aponta para artefato de síntese, mas é o teto de um detector trivial
  e precisa aparecer ao lado de qualquer métrica publicada: um modelo próximo de
  0,67 não aprendeu mais do que uma estatística escalar.
- **Prosódia do gerador.** Duração, banda e nível estão tratados (§5), mas não
  foi medido se pistas prosódicas de mais alto nível (ritmo, entonação) separam
  as classes. Seriam artefato genuíno de síntese, mas um detector que dependa
  delas generaliza pior que um que use artefatos espectrais.
- **Cobertura desigual do upstream.** Entre 218 e 1000 clones por locutor. O
  pareamento estrito mantém o balanceamento, mas a densidade do grid varia.
- **Custo do bloco diagonal.** Apenas a diagonal é utilizável (41,6%). É o preço
  da disjunção dupla.
- **Defeitos da fonte.** O CETUC contém gravações duplicadas com transcrições
  diferentes (§4.3). As detectadas por SHA-256 são excluídas; duplicatas *não
  exatas* com o mesmo problema não seriam detectadas por hash — o bloco F cobre
  parte disso por impressão espectral.
- **Uma única língua e um único canal.** Todo o CETUC foi gravado em condições
  controladas de estúdio.

### 8.1 Ética e consentimento — lacuna declarada, não resolvida

A classe *spoof* é clone de voz por XTTS-v2 sobre locutores **identificáveis**
do CETUC: `build_paired_pt_corpus.py` grava `speaker_name` (nome do locutor,
ex. `Patricia_F001`) em `manifest.jsonl`, extraído diretamente do caminho do
corpus de origem — não é um código anônimo, é o identificador do doador de
voz. Nenhum documento deste projeto (incluindo este) até agora estabelecia
qual é a base de consentimento para clonar a voz desses locutores, nem se
propagar o nome real no manifesto é apropriado.

Isto **não está resolvido** — está declarado aqui para que quem avalie ou
retome este trabalho saiba que precisa checar:

1. Os termos de uso publicados do CETUC (ex. licença OpenSLR) e do corpus de
   clones (Fake Voices/XTTS) quanto a uso para síntese de voz e pesquisa
   antispoofing — não confirmados neste levantamento.
2. Se o nome real deve permanecer no manifesto interno (`manifest.jsonl`,
   `speaker_manifest.json`) ou se deveria ser substituído por um código
   anônimo antes de qualquer distribuição do dataset além do uso interno de
   pesquisa. Trocar exigiria reexportar os artefatos que dependem do
   manifesto (splits, npz).

Trabalhos de referência em antispoofing (ASVspoof, citado na
[§10](#10-referências)) costumam trazer uma declaração explícita de base
ética/consentimento — este projeto ainda não tem uma, e isso deveria ser
resolvido antes de qualquer publicação ou distribuição externa do corpus.

### 8.2 Licença do corpus composto — não confirmada

[`docs/data/public-datasets.md`](public-datasets.md) rotula a licença do
corpus "CETUC-XTTS Pareado" como `MIT` (a licença do Fake Voices), mas lista
a licença do próprio CETUC separadamente como "livre/variável". Como o
corpus pareado deriva diretamente do áudio do CETUC, a licença efetiva do
composto é, na melhor hipótese, a interseção das duas — e "MIT" sozinho
supersimplifica isso. Os termos exatos do CETUC (ex. página OpenSLR 132)
não foram verificados neste levantamento; confirme antes de citar "MIT"
como a licença do corpus composto em qualquer publicação.

## 9. Reprodução

```bash
python scripts/dataset/build_paired_pt_corpus.py --plan
```

```bash
python scripts/dataset/build_paired_pt_corpus.py --build
```

```bash
python scripts/dataset/build_paired_splits.py --build --min-duration-sec 3.0
```

```bash
python scripts/dataset/audit_paired_corpus.py
```

```bash
python scripts/dataset/export_paired_npz.py --out data/datasets/benchmark_dataset.npz --no-compress
```

```bash
python scripts/dataset/freeze_benchmark_test.py --dataset data/datasets/benchmark_dataset.npz --declare-untouched
```

Para usar a partição de locutores publicada pelo CETUC em vez da estratificada:

```bash
python scripts/dataset/build_paired_splits.py --build --speaker-strategy official
```

### 9.0 Variantes de tamanho

Os passos 1 a 4 produzem a partição; o tamanho do `.npz` é escolhido no passo de
exportação. `--target-samples` rateia o total entre treino, validação e teste na
**proporção natural do bloco diagonal**, derivada do `assignment.jsonl` a cada
execução — não de uma constante em código, que ficaria errada se o corpus
mudasse. O rateio usa maior resto (quota de Hare), então a soma bate exatamente
com o alvo e nenhuma partição sofre viés sistemático de arredondamento.

| Variante | Amostras | treino / val / teste | SHA-256 | Comando |
| --- | ---: | --- | --- | --- |
| completa | 40.980 | 33.226 / 3.976 / 3.778 | `ae3662c9…` | sem `--target-samples` |
| reduzida | 15.000 | 12.162 / 1.456 / 1.382 | `3775b35f…` | `--target-samples 15000` |

```bash
python scripts/dataset/export_paired_npz.py \
    --out data/datasets/benchmark_dataset_15k.npz \
    --target-samples 15000 --no-compress
```

`--target-samples 40980` reproduz a partição inteira, o que serve de teste de
consistência do rateio. O alvo precisa ser par (cada par de enunciado gera duas
amostras) e não pode exceder os 20.490 pares disponíveis.

As duas variantes saem da **mesma** partição locutor × frase: muda a densidade
de enunciados por célula, nunca quem está em qual partição. As garantias de
disjunção são portanto idênticas, e cada `.npz` carrega em
`metadata_json.size_policy` os pares usados por partição — reproduzir não depende
de saber o número de fora.

Duas consequências que valem registrar. Resultados obtidos em variantes
diferentes **não são comparáveis**: o conjunto de teste muda. E o teste menor
custa precisão — o intervalo de confiança de 95% da acurácia passa de ≈ ±0,7 pp
com 3.778 amostras para ≈ ±1,2 pp com 1.382, o que não separa modelos
distantes por décimos de ponto.

Cada variante precisa do seu próprio selo:

```bash
python scripts/dataset/freeze_benchmark_test.py --dataset data/datasets/benchmark_dataset_15k.npz --declare-untouched
```

O construtor é resumível por locutor (`corpus/state.json`) e poda o cache do Hub
a cada locutor — sem a poda, os 56 pacotes somariam ~40 GB de cache inútil,
porque o Windows não suporta os symlinks do cache do Hub e guarda duas cópias de
cada arquivo.

### 9.1 Corpus e artefato de treino são coisas diferentes

O **corpus** não tem limite: 49.264 pares, tudo o que as duas fontes oferecem em
comum. A **partição** também não: os 40.980 que sobrevivem às duas disjunções
estão todos em `splits/`.

O **`.npz`** é outra coisa — é a entrada do treino. `BenchmarkData.from_npz`
usa `mmap_mode="r"` (`benchmarks/data.py`) para não materializar o arquivo
inteiro só para abri-lo — páginas mapeadas não contam como RAM anônima e o
kernel pode descartá-las sob pressão em vez de matar o processo. O pico real
de RAM acontece **depois**, quando o treino monta o tensor de treino (limpo +
cópia AWGN): para as 33.226 amostras de treino a 3 s, isso ainda soma vários
GB (ver `docker/compose/{benchmark,train}.nvidia.yml` para os números atuais
e o histórico de ajuste do limite de memória do container). Por isso o
exportador aceita `--max-pairs-train`, que corta **pares** (nunca amostras
isoladas, para o balanceamento sobreviver) em rodízio entre os locutores.

O limite é do hardware de treino, não do dataset: quem tiver memória exporta sem
`--max-pairs-*` e usa a partição inteira.

### 9.2 Janela-fonte: migração concluída

A janela de 3 s (48.000 amostras) é hoje o valor único e consistente em todo o
código — a pendência descrita anteriormente nesta seção foi resolvida:

| Local | Valor atual |
| --- | ---: |
| `app/domain/features/benchmark_frontend.py` → `DEFAULT_SOURCE_SAMPLES` | 48.000 |
| `registry.py`, `input_requirements["source_samples"]`/`target_sequence_length` (todos os modelos de áudio bruto) | 48.000 |
| `scripts/reporting/rebuild_inference_contracts.py` | 48.000 |

As referências a 80.000 (janela antiga de 5 s) e 64.600 (convenção antiga do
RawNet2/AASIST) que ainda aparecem no código são comentários históricos
explicando a mudança, não defaults ativos.

**Estado atual de `data/models/`: vazio.** Não há mais modelos treinados com a
janela antiga (nem com nenhuma outra) no diretório de produção — foram
removidos por estarem desalinhados com o protocolo vigente. Qualquer novo
treino (via `run_models_sequential.py`/`run_benchmark.py`) já usa a janela de
48.000 de ponta a ponta (treino → `benchmark_final/` → inferência do Gradio),
sem risco de skew treino/inferência por esse motivo.

## 10. Referências

<a id="ref-xtts"></a>Casanova, E., Davis, K., Gölge, E., Göknar, G., Gulea, I.,
Hart, L., Aljafari, A., Meyer, J., Morais, R., Olayemi, S., Weber, J. (2024).
*XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model.* Interspeech 2024.

<a id="ref-cetuc"></a>Alencar, V. F. S., Alcaim, A. (2008). *LSF and LPC Derived
Features for Large Vocabulary Distributed Continuous Speech Recognition in
Brazilian Portuguese.* Asilomar Conference on Signals, Systems and Computers.

<a id="ref-falabrasil"></a>Batista, C., Dias, A. L., Neto, N. (2018). *Baseline
Acoustic Models for Brazilian Portuguese Using Kaldi Tools.* IberSPEECH 2018.

<a id="ref-muller"></a>Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A.,
Böttinger, K. (2022). *Does Audio Deepfake Detection Generalize?* Interspeech 2022.

<a id="ref-geirhos"></a>Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R.,
Brendel, W., Bethge, M., Wichmann, F. A. (2020). *Shortcut Learning in Deep
Neural Networks.* Nature Machine Intelligence, 2, 665–673.

<a id="ref-torralba"></a>Torralba, A., Efros, A. A. (2011). *Unbiased Look at
Dataset Bias.* IEEE CVPR 2011.

<a id="ref-kaufman"></a>Kaufman, S., Rosset, S., Perlich, C., Stitelman, O.
(2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM
TKDD, 6(4).

<a id="ref-roberts"></a>Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017).
*Cross-validation strategies for data with temporal, spatial, hierarchical, or
phylogenetic structure.* Ecography, 40(8), 913–929.

<a id="ref-rawnet2"></a>Tak, H., Patino, J., Todisco, M., Nautsch, A., Evans, N.,
Larcher, A. (2021). *End-to-End Anti-Spoofing with RawNet2.* IEEE ICASSP 2021.

<a id="ref-aasist"></a>Jung, J.-w., Heo, H.-S., Tak, H., Shim, H.-j., Chung,
J. S., Lee, B.-J., Yu, H.-J., Evans, N. (2022). *AASIST: Audio Anti-Spoofing
Using Integrated Spectro-Temporal Graph Attention Networks.* IEEE ICASSP 2022.

<a id="ref-asvspoof19"></a>Todisco, M., Wang, X., Vestman, V., Sahidullah, M.,
Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., Lee, K. A.
(2019). *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.*
Interspeech 2019.

<a id="ref-tdcf"></a>Kinnunen, T., Lee, K. A., Delgado, H., Evans, N., Todisco,
M., Sahidullah, M., Yamagishi, J., Reynolds, D. A. (2018). *t-DCF: a Detection
Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic
Speaker Verification.* Odyssey 2018.

<a id="ref-efron"></a>Efron, B., Tibshirani, R. J. (1993). *An Introduction to
the Bootstrap.* Chapman & Hall.
