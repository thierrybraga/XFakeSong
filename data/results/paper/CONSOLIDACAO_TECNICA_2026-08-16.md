# Consolidação técnica — modelos, parâmetros, protocolos e implementação

Auditoria sistemática do estado do projeto, com foco em bugs, incompatibilidades
e incoerências de nomenclatura. Tudo marcado **verificado** foi medido contra o
código ou os artefatos, não inferido.

---

## 0. Sumário executivo

| Dimensão | Estado |
|---|---|
| Fidelidade das métricas publicadas | **11/11 reproduzem** a partir das predições gravadas |
| Coerência das 3 fontes de hiperparâmetro | **6/7 coerentes**; AST diverge em 3 campos |
| `decay_steps` vs orçamento de passos | **2 corrigidos** (CCT, AST); 3 já corretos |
| Divergência código↔artefato | **1 real** (RawGAT-ST) |
| Estabilidade de treino | **2 instáveis** (Conformer, RawGAT-ST) |
| Critério de seleção de checkpoint | **desalinhado** da métrica de avaliação |
| Ferramental XAI | **5 defeitos corrigidos**; SHAP agora executa |
| Guarda de colapso | **ponto cego corrigido** |

> **Este documento tem quatro retificações de 2026-08-17**, marcadas em bloco de
> citação nas seções 1.2, 3.1, 3.2 e 7. Duas correções factuais (o `decay_steps`
> dos braços do RawGAT-ST; CCT e AST fora da lista de retreino) e duas de
> implementação (o `checkpoint_monitor` não chegava ao treinador; o prazo da
> guarda de colapso matava warmup longo). O plano de retreino vigente é o de
> [retraining-adjustments.md](../../../docs/evaluation/retraining-adjustments.md),
> seção "2026-08-17".

---

## 1. Modelos e artefatos

### 1.1 Fidelidade — íntegra

Recomputei a acurácia de cada um dos 11 modelos a partir do
`predictions_clean.csv` gravado e confrontei com o `results.json`. **Todas
reproduzem** dentro de 10⁻⁴. Sem *scores* constantes, não finitos ou épocas
incompletas. **Verificado.**

### 1.2 Quem precisa de retreino

| Modelo | Motivo | Prioridade |
|---|---|---|
| **RawGAT-ST** | Divergência real (dropout 0,35→0,5; l2 1e-3→3e-3; decay 100k→152,1k) + instável + seleção custa 5,3 p.p. + `decay_steps` cobria só 66% do treino | **1** |
| **CCT** | `decay_steps` corrigido nesta auditoria (65.700 → 76.100) | 2 |
| **AST** | `decay_steps` corrigido (262.500 → 304.100) + drift de 3 campos entre registry e planning | 2 |
| **Conformer** | `unstable_oscillation` (sem divergência de config) | 3 |

Íntegros: Res2Net, RawNet2, AASIST, SVM, RandomForest, WavLM, HuBERT.

> **Retificação (2026-08-17) — CCT e AST saem da lista.** A divergência de
> `decay_steps` é real, mas **não afeta os artefatos publicados**: os
> checkpoints selecionados são das épocas 48 (CCT) e 33 (AST), enquanto o LR só
> chegava ao piso na época 86,3. As 14 épocas congeladas são inteiramente
> posteriores à época que virou artefato. Retreinar os dois melhores modelos do
> escopo (99,57% e 99,71%) gastaria ~12 h para corrigir um defeito que não os
> tocou — a divergência fica **registrada**, não retreinada.
>
> Também muda a leitura do RawGAT-ST: dos três itens de divergência, só
> `decay_steps` sobrevive. Dropout 0,50 e L2 3e-3 foram **revertidos** depois
> que o fatorial os mediu (o dropout trava a validação no acaso). O retreino
> continua obrigatório, mas com a configuração publicada + decay completo +
> `val_eer`.

### 1.3 Três falsos positivos que registrei e retiro

Cheguei a acusar problemas que não existem, todos por **comparar contra a fonte
errada**. Registro porque o método de auditoria importa tanto quanto o
resultado:

1. `training_config` no nível raiz → os campos de regularização estão
   aninhados em `model_parameters`;
2. `registry.default_params` → o benchmark aplica o **plano** por cima
   (`effective_hyperparameters`);
3. `batch_size` nominal → `_fit_to_device` aplica `cap = 32` em GPU, então o 64
   do Res2Net nunca chega ao treino.

**A fonte correta é `planning.effective_hyperparameters()` contra
`training_config.model_parameters`, considerando o cap de dispositivo.**

---

## 2. Parâmetros

### 2.1 Drift entre as três fontes

O `CLAUDE.md` adverte que hiperparâmetros vivem em três lugares. Medi:

| Arquitetura | Estado |
|---|---|
| CCT, Conformer, Res2Net, RawNet2, AASIST, RawGAT-ST | **coerentes** |
| **AST** | `dropout_rate` 0,3 (registry) vs 0,25 (planning); `decay_steps` 50.000 vs 304.100; `warmup_steps` 2.000 vs 3.000 |

O plano vence na execução, então o AST **treina** com 0,25/304.100/3.000. Os
valores do registry são código morto que aparenta ser configuração — risco de
alguém ajustá-los esperando efeito.

### 2.2 `decay_steps` — bug encontrado e corrigido

O cronograma de cosseno decai ao longo de `decay_steps`. Se menor que o total
de passos, o LR chega ao piso antes do fim.

| | supunha | real | era | agora |
|---|---:|---:|---:|---:|
| CCT | 21.024 amostras | 24.324 | 65.700 | **76.100** |
| AST | 21.000 amostras | 24.324 | 262.500 | **304.100** |

Ambos exatamente **86,3 épocas** — a mesma premissa defasada (provável divisão
70/30 sobre 15.000 em vez das 12.162 reais). As 14 épocas finais rodavam com LR
congelado.

O **RawGAT-ST** era pior: 100.000 contra 152.100 = **66%**, ou seja 34 épocas
congeladas. Já corrigido no código, mas o artefato publicado carrega o defeito —
o que ajuda a explicar sua curva travada em ~89%.

Após a correção, os cinco batem em 1,00 (AASIST em 0,99). **Verificado.**

---

## 3. Protocolo de treino

### 3.1 Critério de seleção de checkpoint — desalinhado

Seleciona por `val_loss` (**calibração**), avalia por EER/AUC/t-DCF
(**ordenação**). O próprio TCC argumenta na §4.5 que essas métricas medem
ordenação, e por isso as computa sobre a margem bruta dos clássicos — aplica o
princípio na avaliação e o contraria na seleção.

Custo medido: **0 p.p. em 8 dos 9** neurais; **5,3 p.p. no RawGAT-ST**. No
retune com L2=3e-3, o mínimo cai na **época 1** (`val_loss` 0,6923 ≈ ln 2), um
modelo desinformativo — porque um detector que aprende mas erra com confiança
nunca bate esse piso.

**Implementado:** callback `ValidationEER` publicando `val_eer`, e
`checkpoint_monitor` configurável (padrão `val_loss`, preservando os artefatos
publicados). 6 testes.

> **Retificação (2026-08-17): a opção não tinha efeito.** Os 6 testes
> exercitavam o callback ISOLADO e passavam com o encanamento roto. O
> `checkpoint_monitor` era setado no `BenchmarkConfig` e lido do
> `TrainingConfig`, sem nada ligando os dois — e o `TrainingService` descarta em
> silêncio toda chave que não seja campo declarado do dataclass. O smoke de 2
> épocas pediu `val_eer` e gravou `{"monitor": "val_loss"}` no `best.json`
> (evidência em `data/results/_smoke_eer/`). Corrigido declarando o campo nos
> dois dataclasses e propagando em `runner.py::_run_neural`, com 4 testes de
> encanamento.

### 3.2 Guarda de colapso — ponto cego corrigido

`CollapseAbort` só armava após o modelo cruzar 60% de validação; um treino que
**nunca aprende** jamais era abortado. Custou 7,6 h no braço (d) do RawGAT-ST,
com validação em 0,5000 exato por 25 épocas.

**Implementado:** `arm_deadline = 15`. Calibrado com dados: todas as nove
arquiteturas cruzaram 0,6 **até a época 3** no run oficial — margem de 5×. 5
testes de regressão.

> **Retificação (2026-08-17): o prazo sozinho era ambíguo e quebrou uma guarda
> anterior.** Olhando só `val_accuracy`, "nunca aprendeu" e "warmup longo"
> produzem a MESMA curva, e o prazo matava os dois —
> `test_collapse_abort_ignora_inicio_lento` (de 06/08, um modelo no acaso por 30
> épocas que depois sobe a 0,97) passou a falhar. O sinal que separa os casos é
> a **folga treino-validação**: 40 pontos no braço (d) na época 15; zero num
> warmup genuíno. A guarda passa a exigir `generalization_gap ≥ 0,20` e se cala
> sem a métrica de treino nos logs.

### 3.3 Tabela de custo — corrigida

Quatro entradas eram extrapolações que superestimavam de 1,9× a 5,5×.
Substituídas pelo medido, normalizado a `_REFERENCE_FIT_SAMPLES`:

| | antiga | medida | erro |
|---|---:|---:|---:|
| RawNet2 | 65,1 | 11,7 | 5,5× |
| RawGAT-ST | 196,0 | 73,4 | 2,7× |
| AASIST | 101,0 | 42,2 | 2,4× |
| AST | 57,0 | 30,1 | 1,9× |

Timeouts derivados mantêm folga de 3,0× sobre o real. **Verificado.**

---

## 4. Protocolo de benchmark

### 4.1 Compose — restrição vencida removida

O `benchmark.nvidia.yml` fixava `--models RawGAT-ST AASIST`, marcado como
TEMPORÁRIO até que ambos concluíssem. Ambos constam `status: ok`. Mantida, a
flag faria o benchmark rodar **2 das 11** arquiteturas em silêncio.

### 4.2 Coluna de 5 dB

O nível não visto — declarado como "o resultado mais informativo do recorte" —
não era tabulado. Acrescentado ao gerador `update_tcc_latex.py`, de modo a
sobreviver à regeração.

### 4.3 Testes pareados sob ruído

O aparato (McNemar + *bootstrap* + Holm) só rodava no conjunto limpo. Estendido
a 10 e 5 dB: **nenhum dos dez pares do topo se separa** por nenhum dos dois
testes. O ordenamento sob ruído é bloco indistinguível, não *ranking*.

---

## 5. Geração de resultados e XAI

### 5.1 Cinco defeitos que impediam o ferramental de rodar

`data/results/xai/` nunca existiu porque o script **não podia** executar:

1. contrato de 63 colunas (v1) contra modelos de 183 (v2);
2. `--dataset` default apontando para `.npz` retirado do disco;
3. `--synthetic` gerando 32 colunas, incompatível com ambos;
4. `--max-samples` reindexando partições pré-definidas → `IndexError`;
5. SVM: `'SVC' has no attribute 'predict_proba'` — a calibração isotônica vive
   no invólucro do pipeline.

O defeito 4 tinha camada científica: subamostrar quebraria a garantia do próprio
script de explicar o **mesmo** conjunto de teste das métricas. A correção
recusa a opção em vez de fazê-la funcionar.

O defeito 5 foi resolvido explicando a **margem** (`decision_function`) — que é
a grandeza sobre a qual o TCC declara computar as métricas de ordenação.

### 5.2 Validação cruzada do SHAP

SHAP (TreeExplainer) × permutação, sobre o mesmo RF: **top-3 idêntico**, 7/10 de
sobreposição no top-10. Implementação válida.

**Mas as famílias invertem:** SHAP dá LFCC 55,7% / Temporal 27,1%; permutação dá
Temporal 44,7% / LFCC 41,5%. A permutação subestima famílias com features
redundantes (120 colunas LFCC correlacionadas contra 11 temporais). O TCC hoje
apresenta a leitura da permutação como fato — **precisa qualificar**.

---

## 6. Nomenclatura e configuração

### 6.1 Constantes com nomes enganosos

| Constante | Cardinalidade | Observação |
|---|---:|---|
| `ALL_TCC_ARCHITECTURES` | **9** | "ALL" mas **exclui** WavLM e HuBERT Original |
| `DOCKER_TRAINING_ARCHITECTURES` | 11 | o recorte completo |
| `OFFICIAL_TCC_RESULT_ORDER` | 11 | usa nomes de exibição (AST/CCT/Res2Net) |
| `NEURAL_TCC_ARCHITECTURES` | 7 | exclui SSL e clássicos |
| `NEURAL_DOCKER_ARCHITECTURES` | 9 | inclui SSL |

`ALL_TCC_ARCHITECTURES` é o problema: o nome promete o conjunto completo e
entrega 9 de 11. Renomear para `KERAS_TCC_ARCHITECTURES` (ou
`NON_SSL_TCC_ARCHITECTURES`) elimina a armadilha.

### 6.2 Três camadas de nome por modelo

`result_key` (AST) → `benchmark_name` (SpectrogramTransformer) → módulo
(`spectrogram_transformer.py`). O manifesto documenta a correspondência e o
apêndice do TCC a reproduz — **coerente**, mas exige que qualquer script novo
consulte o manifesto em vez de assumir.

### 6.3 Configs de treino — coerentes

Oito arquivos, sem sobreposição indevida; a união dos `scope: official` cobre
exatamente os 11 do manifesto. `retune_ajustado.yaml` declara apenas RawGAT-ST,
o que **confere** com a auditoria de divergência.

---

## 7. Plano de retreino recomendado

Ordem por retorno sobre custo de GPU:

> ⚠️ **SUPERSEDIDO EM 2026-08-17.** Esta seção partia de uma premissa falsa
> (ver a retificação logo abaixo) e recomendava ~66 h de GPU, das quais ~54 h
> eram desnecessárias. O plano vigente está em
> [retraining-adjustments.md](../../../docs/evaluation/retraining-adjustments.md),
> seção "2026-08-17". Mantido aqui como trilha de auditoria.

| # | Execução | Config | Custo | Justificativa |
|---|---|---|---:|---|
| 1 | RawGAT-ST braço (l) | L2 3e-3, dropout 0,35, **decay 152.100**, `monitor=val_eer` | ~27 h | isola o L2 com cronograma correto |
| 2 | RawGAT-ST braço (d) | dropout 0,50, L2 1e-3, decay 152.100 | ~27 h | o braço anterior colapsou sob decay defasado |
| 3 | CCT | decay 76.100 | ~1 h | corrige as 14 épocas congeladas |
| 4 | AST | decay 304.100 | ~11 h | idem, e resolve o drift de 3 campos |
| 5 | Conformer | inalterado, múltiplas sementes | ~2 h/semente | instabilidade exige variância |

**O braço (l) em curso deve ser abortado.** Ele roda com `decay_steps` = 100.000
(66% do treino), então parte do efeito que eu atribuiria ao L2 é LR congelado —
exatamente o erro de duas variáveis que a investigação existe para evitar.

### Retificação (2026-08-17)

**Os dois braços rodaram com `decay_steps` = 152.100, não 100.000.** Verificado
por três vias independentes: o `effective_training_config.json` que cada run
gravou, o valor em `planning.py` no HEAD daquele dia, e o default de
`rawgat_st.py::create_model`. O 100.000 que eu li pertence ao **AASIST**, outra
entrada do mesmo dicionário — erro de leitura, não de configuração.

Consequências:

1. **Os itens 1 e 2 caem** (~54 h). Os braços são medida válida e o fatorial se
   apoia neles: dropout 0,50 trava a validação em 0,5000 exato por 25 épocas com
   treino a 95,4% (fator letal); L2 3e-3 não move o teto (−0,13 p.p.). O código
   voltou a 0,35/1e-3 nas três fontes.
2. **A justificativa do item 2 se inverte.** O braço (d) não colapsou "sob decay
   defasado" — colapsou pelo dropout, com o cronograma correto.
3. **Os itens 3 e 4 caem por outro motivo** (ver §1.2 abaixo, também retificada):
   os checkpoints publicados de CCT e AST são das épocas 48 e 33, e o piso de LR
   só entrava na época 86,3. O trecho congelado é posterior à época selecionada
   e não tocou o artefato — a divergência é de reprodutibilidade, não de
   validade, e fica registrada em vez de custar ~12 h de GPU.
4. **Sobra o item 5** (opcional) e um único retreino obrigatório: RawGAT-ST com
   dropout 0,35, L2 1e-3, decay 152.100 e `--checkpoint-monitor val_eer`.

O aborto do braço (l) na época 66/100, decidido por esta seção, ficou sem a
justificativa que o motivou — mas não custou o resultado: a curva de validação
já mostrava o teto em 0,8984 (época 40) e as 34 épocas restantes não mudariam a
leitura do fatorial.

---

## 8. Pendências que não são técnicas

- Nome completo e titulação da orientadora (3 lugares no TCC)
- 5 campos da ficha catalográfica (Biblioteca da UFSJ)
- Três decisões de limpeza (~2,2 GB, irreversíveis)
- Incorporar ao texto os achados desta sessão ainda não escritos: colapso do
  braço (d), custo de seleção, divergência SHAP × permutação, correções de
  `decay_steps`
