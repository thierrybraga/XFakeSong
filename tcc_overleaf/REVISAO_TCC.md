# Revisão acadêmica — `tcc.tex`

**Trabalho:** *Pipeline Modular para Detecção de Áudio Sintético usando Arquiteturas Neurais Híbridas e Análise Empírica de Características*
**Autor:** Thierry Gonçalves Braga · **Revisão em:** 25/06/2026

Revisão de rigor acadêmico do `tcc.tex` (2.233 linhas). Esta é uma **lista de melhorias e ajustes** — nada no arquivo foi alterado. As verificações automáticas executadas estão resumidas no final.

---

## 1. Avaliação geral

O trabalho está **bem estruturado, honesto e tecnicamente competente**. Destaca-se positivamente por algo raro em TCCs: uma seção de limitações genuinamente autocrítica, intervalos de confiança de Wilson, reconhecimento explícito do risco de vazamento por falante, discussão de uso dual e LGPD, e rastreabilidade de artefatos. A escrita é clara e o LaTeX, limpo (compila sem erros).

O ponto que decide a defesa é **um só, e é grave**: os resultados de topo (100,00% de acurácia / EER 0,00%) são, hoje, **indefensáveis sem uma divisão treino/teste disjunta por falante e por gerador**. O próprio texto admite isso, mas o trata como trabalho futuro. Uma banca vai tratar como pré-requisito. Resolver esse ponto (ou reformular as alegações) é a prioridade absoluta.

Abaixo, as melhorias estão ordenadas por **severidade**: 🔴 Crítico (resolver antes de defender) · 🟠 Alto · 🟡 Médio · ⚪ Baixo/cosmético.

---

## 2. 🔴 Críticos — resolver antes da defesa

### 2.1 Vazamento de dados / divisão não disjunta por falante (o problema central)
A divisão 70/15/15 estratifica **por classe**, mas não garante separação por falante, canal ou sistema gerador (Seções `sec:dataset_protocolo` e `sec:ameacas_validade`). Com isso, **100,00% de acurácia e EER 0,00% são o sintoma clássico de vazamento de fonte**: o modelo pode estar reconhecendo o falante/gerador, não a "sinteticidade".

Ações:
- Reexecutar pelo menos os modelos de topo (Conformer, Sonic Sleuth, Hybrid CNN-Transformer, MultiscaleCNN) com **split disjunto por falante E por gerador/TTS**. É o experimento que mais protege o trabalho.
- Se isso não for viável a tempo, **rebaixar explicitamente toda alegação numérica de topo** para "resultado em protocolo controlado, sujeito a possível vazamento de fonte" — e dizer isso já no Resumo e na Conclusão, não só nas limitações.
- Reportar a queda de desempenho do split antigo para o novo: essa diferença é, por si só, uma contribuição científica.

### 2.2 Ausência de validação externa
Todo o benchmark roda sobre uma base privada e não documentada. Sem **nenhum** ponto de ancoragem externo, os números não são interpretáveis fora do próprio dataset.
- Avaliar ao menos os 3–4 melhores modelos em **um** conjunto público padrão (ASVspoof 2019 LA *eval* é o mínimo esperado; In-the-Wild e WaveFake reforçam). Mesmo um resultado pior, externamente, vale mais que 100% interno.

### 2.3 Proveniência do dataset e o nome "BRSpeech-DF"
Faltam metadados essenciais: nº de falantes, geradores/TTS de origem, idioma, licenças, durações originais e verificação de duplicatas (reconhecido em `sec:brspeech_df`). Sem isso, o conjunto não é cientificamente caracterizável.
- O prefixo **"BR"** sugere português brasileiro, mas a proporção por idioma é "não documentada" (Tabela `tab:dataset_brspeechdf`). Isso é arriscado: a banca pode ler como afirmação não sustentada. **Renomeie** (ex.: `Internal-DF-15k`) até confirmar a composição linguística, ou documente a proporção antes de defender.
- Consolide um "datasheet" mínimo da base (origem de cada partição, geradores, idiomas, licenças).

### 2.4 Conformidade ABNT do sistema de citação
O documento usa `\usepackage{cite}` + `\begin{thebibliography}` (numérico, estilo IEEE) com as referências em **ordem alfabética**. No **sistema numérico da ABNT (NBR 10520/6023)**, as referências devem aparecer **na ordem de citação**, não alfabética — então a numeração atual fica fora de ordem no texto.
- Decida com a orientadora: (a) **autor-data** (`abntex2cite` ou `biblatex-abnt`), que é o mais comum em TCCs brasileiros; ou (b) manter numérico, mas **reordenar a bibliografia por ordem de aparição**. Do jeito atual, não é compatível com nenhum dos dois sistemas ABNT.

### 2.5 Classe `article` vs. template institucional / abnTeX2
O documento usa `\documentclass{article}` e reproduz a ABNT manualmente (margens, espaçamento, folha de aprovação). Muitos colegiados **exigem** o template oficial da instituição ou `abntex2`.
- Confirme com o colegiado/orientadora se o template livre é aceito. Se não for, migrar para `abntex2` resolve de uma vez a maioria dos itens de formatação pré-textual abaixo.

---

## 3. 🟠 Altos — importantes para o rigor

### 3.1 Tabelas e figuras não referenciadas no texto
A verificação automática confirmou que **a maioria das tabelas e todas as 6 figuras de resultados** nunca são chamadas por `\ref` na prosa. Em texto acadêmico, todo elemento flutuante deve ser citado e comentado ("conforme a Tabela X…", "a Figura Y mostra…").

Não referenciadas (precisam de menção no corpo):
- **Figuras:** `benchmark_accuracy_auc`, `benchmark_eer`, `benchmark_latency`, `benchmark_size`, `benchmark_robustness`, `training_stability`.
- **Tabelas:** `resultados_consolidados`, `intervalos_confianca`, `robustez_awgn`, `eficiencia_modelos`, `estabilidade_treinamento`, `hardware_benchmark`, `reprodutibilidade_benchmark`, `dataset_brspeechdf`, `decisao_final_modelos`, `hiperparametros_consolidados`, `rastreabilidade_artefatos`.

A `tab:resultados_consolidados` — a tabela principal do trabalho — está entre as não referenciadas. Corrigir isso é rápido e eleva muito a percepção de rigor.

### 3.2 Ordem dos elementos pré-textuais (NBR 14724)
Ordem atual: Resumo → Abstract → Lista de Siglas → **Sumário** → Lista de Figuras → Lista de Tabelas.
Ordem ABNT correta: …Resumo → Abstract → **Lista de Figuras → Lista de Tabelas → Lista de Abreviaturas e Siglas → Sumário** (o Sumário é sempre o **último** pré-textual).
- Mover `\tableofcontents` para depois de `\listoffigures`/`\listoftables` e reposicionar a lista de siglas.

### 3.3 Sem testes de significância entre modelos
Diferenças no topo (99,73% vs. 99,96% vs. 100%) não são estatisticamente testadas (reconhecido). Com as predições já salvas, é barato:
- **McNemar pareado** entre pares de modelos próximos e/ou **bootstrap** do IC da diferença de acurácia. Sem isso, o "ranking" do topo não é defensável como ordenação real.

### 3.4 Variância de treinamento: semente única
Tudo roda com **semente 42** apenas. O IC de Wilson captura a incerteza de **amostragem do teste**, mas **não** a estocasticidade do treino (inicialização, ordem de batches). Um modelo que dá 100% com uma semente pode dar 99,5% com outra.
- Rodar os modelos de topo com **≥3 sementes** e reportar média ± desvio. Deixe explícito no texto que o IC de Wilson e a variância entre sementes medem coisas diferentes.

### 3.5 Metodologia de latência frágil
Tabela `tab:eficiencia_modelos`: latências sem desvio-padrão (coluna "---", reconhecido) e com anomalias que a banca vai questionar:
- **WavLM (10,18 ms) mais rápido que HuBERT (21,17 ms)** apesar de tamanho idêntico (~94,5 M / 361 MB).
- **SVM 0,20 ms** vs. **Random Forest 26,75 ms** — comparáveis a redes neurais.
- Faltam: dispositivo de medição (CPU/GPU?), batch size, *warmup*, se inclui pré-processamento. SVM e redes parecem medidos em hardware diferente — o que torna a comparação direta injusta.
- Padronizar o protocolo de latência (mesmo device, com warmup, batch=1, n medições, reportar mediana/DP/p95) e descrevê-lo no texto.

### 3.6 Robustez: só AWGN, e colapso para 50%
A Tabela `tab:robustez_awgn` testa apenas ruído gaussiano. Vários modelos **colapsam para exatamente 50% (chance)** sob ruído — SVM em todos os SNR, HuBERT a 10 dB (50,00%), RawNet2 (50,36%), WavLM (51,29%), Ensemble (52%). Isso é **colapso**, não degradação graciosa, e merece ser nomeado como tal.
- A afirmação de aplicabilidade em "cenários reais" exige pelo menos **compressão de codec (MP3/Opus)** e idealmente reverberação. Sem isso, suavizar a alegação de prontidão para campo.

---

## 4. 🟡 Médios — qualidade técnica e de formatação

- **Limiar de decisão vs. EER:** Hybrid CNN-Transformer aparece com acurácia 99,96% **e** EER 0,00%; MultiscaleCNN com 99,73% e EER 0,18%. EER 0% implica separabilidade perfeita (existe limiar com 0 erro), mas a acurácia <100% é medida no limiar fixo 0,5. **Explicite** que a acurácia usa limiar 0,5 e o EER usa o limiar ótimo — caso contrário parece contradição.

- **`\resizebox{\textwidth}{!}{...}` em ~9 tabelas:** escala a fonte de forma arbitrária, deixando **tamanhos de fonte inconsistentes entre tabelas** e distorcendo a espessura das réguas `booktabs`. O próprio código tem comentários pedindo "fonte mínima 9pt", mas não há garantia. Preferir `\small`/`\footnotesize` + ajuste de larguras de coluna (ou `tabularx`) a `\resizebox`.

- **Nota dentro do `tabular` após `\bottomrule`** (Tab. `tab:eficiencia_modelos`, l. 1429): `\multicolumn{7}{l}{...}` abaixo da régua inferior foge do padrão `booktabs`. Usar `threeparttable` (notas de tabela) ou mover a nota para a legenda.

- **Inconsistência de unidades (siunitx):** o texto mistura `\SI{30}{\decibel}` com `-25$\,$dB` (l. 866) e `-23$\,$LUFS` (l. 845). Padronizar tudo via `siunitx` (definir `\DeclareSIUnit\lufs{LUFS}`).

- **Inconsistência departamento × título:** capa diz "Departamento de Engenharia de Telecomunicações"; a folha de aprovação diz "Bacharel em **Ciência da Computação**". Acertar qual é o curso/departamento.

- **Orientadora sem sobrenome:** "Prof.ª Ana Cláudia" aparece sem sobrenome na capa e na folha de aprovação. Completar.

- **Resumo em dois parágrafos:** a NBR 6028 recomenda **parágrafo único**. Considerar fundir os dois parágrafos do Resumo (e do Abstract). O Resumo também gera o único *overfull* notável (29 pt, l. 274–287) — ajuste de redação resolve.

- **Configuração de MFCC atípica:** "M=40 filtros, coeficientes: 40" (l. 945). Em geral retêm-se **menos** coeficientes que filtros (ex.: 40 filtros → 13–20 MFCC). Reter os 40 é incomum; justificar ou ajustar. (O Mel-spec usa 80 bandas, l. 957 — não é erro, mas vale harmonizar a narrativa.)

- **Comparação WavLM < HuBERT:** a análise (caixa em `sec:resultados`) está boa, mas some com a hipótese mais provável: **subajuste de *fine-tuning*/learning rate específico do SSL**. Reforçar que é provável artefato de protocolo, não da arquitetura — já que isso afeta a credibilidade geral dos resultados SSL.

---

## 5. ⚪ Baixos / cosméticos

- **Redundância "ver Equação X":** após exibir uma equação numerada, o texto frequentemente repete "(Equação~\ref{...})" / "ver Equação X". Enxugar — a equação está logo acima.
- **Pacote `pgfplotstable`** é carregado "para verificação de tamanho de fonte" mas **não é usado**; carrega `pgfplots` (pesado). Pode remover. `inputenc utf8` é redundante no LaTeX moderno (inofensivo).
- **Data de acesso inconsistente:** quase todas as referências usam "Acesso em: 15 jun. 2025"; `pham2024deepfake` usa "25 jun. 2026". Uniformizar.
- **Chaves de citação:** `dfrlab2024` e `farrugia2024` são ambas FARRUGIA/DFRLab — padronizar o esquema de chaves (ex.: `farrugia2024a`/`farrugia2024b`).
- **Cabeçalho com nome do autor** em todas as páginas e **número de página no rodapé central**: a ABNT pede número no **canto superior direito** e não exige cabeçalho corrente com autor. Verificar a norma do colegiado (resolvido automaticamente se migrar para `abntex2`).
- **Verificar números citados na introdução** contra as fontes: "78 casos" (`farrugia2024`) e "73% / 51%" de detecção humana (`muller2022warning`). Confirmar valores e, se possível, página exata — são os números mais "citáveis" do trabalho.

---

## 6. Pontos fortes (manter)

- Seção de **Ameaças à Validade** honesta e específica (validade interna/externa/construção).
- **Intervalos de confiança de Wilson** e a ressalva de que 100% ≠ erro populacional nulo.
- **Ética e LGPD** integradas (introdução → discussão → conclusão), incluindo uso dual.
- **Reprodutibilidade**: semente, manifesto, rastreabilidade de artefatos, comandos de execução.
- Estrutura de gerações de métodos (Tab. `survey_geracoes`) clara e bem fundamentada.
- LaTeX limpo: **todas as 30 citações e todos os `\ref` resolvem; as 14 matrizes de confusão existem; compila sem erros**.

---

## 7. Priorização sugerida

| Prioridade | Item | Esforço | Impacto na defesa |
|---|---|---|---|
| 1 | Split disjunto por falante/gerador (§2.1) | Alto | Decisivo |
| 2 | Validação externa de 3–4 modelos (§2.2) | Médio | Muito alto |
| 3 | Referenciar todas tabelas/figuras no texto (§3.1) | Baixo | Alto |
| 4 | Corrigir sistema de citação ABNT (§2.4) | Médio | Alto (formal) |
| 5 | Confirmar template `article` vs. institucional (§2.5) | Baixo–Alto | Alto (formal) |
| 6 | Documentar/renomear dataset (§2.3) | Médio | Alto |
| 7 | McNemar/bootstrap + ≥3 sementes (§3.3–3.4) | Médio | Alto |
| 8 | Ordem pré-textual + protocolo de latência (§3.2, §3.5) | Baixo–Médio | Médio |
| 9 | resizebox, unidades, MFCC, demais 🟡/⚪ | Baixo | Médio–baixo |

**Resumo em uma frase:** o trabalho é sólido e maduro na escrita e na autocrítica; o que falta é **fechar o flanco metodológico** (split por falante + validação externa + testes de significância) e **alinhar a formatação à ABNT** (citação, ordem pré-textual, template) antes da defesa.

---

### Apêndice — verificações automáticas executadas
- Cross-check de citações: **30 `\cite` ↔ 30 `\bibitem`, 0 órfãs, 0 ausentes**.
- Cross-check de `\label`/`\ref`: **0 referências quebradas**.
- Cross-check de figuras: **14/14 matrizes de confusão presentes** em `figures/confusion_matrices/`.
- Compilação (pdfLaTeX, com *stub* de `siunitx`/`babel` por limitação do ambiente de teste): **43 páginas, sem erros, sem referências indefinidas, 3 *overfull hboxes*** (1 de 29 pt no Resumo, 2 menores). No Overleaf, com os pacotes completos, compila normalmente.
- Tabelas/figuras **referenciadas** na prosa: `survey_geracoes`, `ranking_features`, `pareto_operacional`, `modulos_pipeline_real`, matrizes de confusão. As demais (listadas em §3.1) não são chamadas por `\ref`.
