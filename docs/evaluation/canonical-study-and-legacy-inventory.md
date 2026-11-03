# Estudo canônico do TCC e inventário de legado

## Referência escolhida

Por decisão do projeto, o artigo em `data/results/paper/main.tex` é a referência canônica do estudo de benchmark do TCC. Esta escolha fixa a pergunta científica, o recorte de dados, as partições, os controles e as métricas que o código e a documentação devem implementar. Ela não valida automaticamente os números publicados nas tabelas: os artefatos locais ainda apresentam divergências de proveniência descritas abaixo.

O artigo avalia uma subamostra balanceada de 15.000 amostras, armazenada em `data/datasets/benchmark_dataset_15k.npz` neste checkout. O projeto também documenta o corpus pareado de origem e uma exportação histórica de 40.980 amostras; não confundir esses artefatos, seus splits, o NPZ avaliado e execuções de outros protocolos.

## Dataset e protocolo descritos no artigo

| Item | Especificação canônica do estudo |
| --- | --- |
| Corpus | CETUC pareado a clones existentes no Fake Voices, gerados com XTTS-v2 |
| Pares | Mesmo enunciado, texto e locutor na amostra real e no clone |
| Escopo | 15.000 amostras: 7.500 reais e 7.500 falsas; 7.500 pares |
| Amostragem | Subamostra do corpus pareado, semente fixa e rodízio entre locutores e frases |
| Áudio | Mono, reamostrado a 16 kHz, recorte central de 3 s (48.000 amostras), sem padding ou repetição no dataset avaliado |
| Nível | RMS normalizado a −26 dBFS, teto de pico −1 dBFS, conforme contrato de exportação |
| Partições | 12.162 treino, 1.456 validação e 1.382 teste; 6.081, 728 e 691 pares |
| Isolamento | Bloco diagonal com locutor e frase disjuntos entre treino, validação e teste |
| Escopo de generalização | In-domain: uma fonte de fala (CETUC) e um gerador (XTTS-v2) |
| Controles | 11 entradas oficiais; orçamento de até 100 épocas, seed 42, lote conforme modelo |
| Ruído | AWGN na forma de onda: treino a 30/20/10 dB; teste a 30/20/10/5 dB |
| Seleção | Keras: menor EER limpa de validação; SSL PyTorch original: menor loss limpa de validação; teste reservado |
| Latência | 30 repetições, conforme configuração consolidada do runner |
| Relato | Acurácia, F1, AUC-ROC, EER, proxy min-tDCF, robustez por SNR, latência e tamanho; ICs por bootstrap de clusters de texto |

As 11 entradas são RandomForest, SVM, Hybrid CNN-Transformer (CCT), SpectrogramTransformer (AST), MultiscaleCNN (Res2Net), Conformer, RawNet2, AASIST, RawGAT-ST, WavLM Original e HuBERT Original. Para WavLM/HuBERT Original, o artigo descreve backbone SSL congelado com cabeça treinável.

As fontes estão fixadas em `falabrasil/cetuc` revisão `6dbc8081d35fff5bdd133d8a12b8a11f91959b01` e `unfake/fake_voices` revisão `541bf396da524f92a6d6594a0e9952210e7d7e7e`. Os clones XTTS-v2 já existem na fonte Fake Voices; o projeto os pareia com os originais, reamostra e prepara o recorte. A exportação integral do corpus pai é uma especificação separada: `data/datasets/benchmark_dataset.npz` não existe neste checkout, e o único NPZ presente é o reduzido usado pelo artigo. O corpus pareado e suas partições existem em `data/datasets/corpus/` (98.528 entradas em `manifest.jsonl`) e `data/datasets/splits/` (40.980 amostras em `split_manifest.json`: 33.226 de treino, 3.976 de validação e 3.778 de teste). O vínculo entre esses diretórios e o NPZ de 15 mil amostras não foi conferido amostra a amostra, porque isso exigiria ler a partição de teste.

O NPZ local tem SHA-256 `3775b35fb05155011b21a30ff60632aaa7c9f67f031bd6f7874a5f313363400f` e tamanho 2.925.524.332 bytes. O arquivo `data/datasets/benchmark_dataset_15k.npz.test-lock.json` registra o mesmo hash, as mesmas contagens e uma declaração de que o teste foi selado antes do treino; isso é uma declaração registrada, não uma auditoria independente do histórico de acesso. O lock registra a identidade ZIP `4f9b86306d4fa5221ee2aacce9bd0ed5fd9daa22b47640ffec996ef60efabab6`, calculada por nome de membro, CRC-32 e tamanho descompactado. Os relatórios do run atual, em `data/results/benchmark/`, gravam no campo `test_split_sha256` a impressão digital `66ffed2572294062c6d1a5855b278599e0c8972b9ebfed98afecbd1f620ea5d9`, calculada por hashes ordenados de amostras e rótulos.

O artigo cita `ab4c3a9fa97982624a21fdecce358c4fa046e6bfe37a01620c0fb561b16dd60a`. Esse valor não está no lock nem no run atual, mas foi localizado: é o `test_split_sha256` dos 11 `artifact_manifest.json` de `data/results/archive/pre_bateria_2026-08-20/models_benchmark_final/`, dos `results.json` de `data/results/archive/pre_bateria_2026-08-20/clean_benchmark_15k/` e de `data/results/paper/consolidated/benchmark_summary.json`. Os `results.json` arquivados gravam para ele o mesmo método de impressão digital do run atual, e não o método ZIP. A versão commitada de `main.tex` atribui a ele o método ZIP, o que é inconsistente. A cópia de trabalho, ainda não commitada em 2026-11-03, separa os três identificadores, mas afirma que o valor não foi localizado, o que também precisa de revisão.

Os dois runs registram o mesmo SHA-256 de NPZ, a mesma identidade ZIP e 1.382 amostras de teste, mas impressões digitais diferentes. Só o run atual declara `band_correction_hz: 7500`, e o código atual calcula a impressão digital depois dessa correção (`benchmarks/runner.py`), o que é compatível com a diferença. Essa explicação não foi recalculada, porque isso exigiria ler a partição de teste, e deve ser confirmada antes da publicação.

### Limites que devem acompanhar qualquer resultado

- O estudo avalia um corpus, um gerador, uma semente de treino e uma realização de ruído por condição; não demonstra generalização cross-generator, cross-corpus ou para gravações de campo.
- A janela central de 3 s não representa o enunciado completo.
- O artigo classifica RF e SVM como exploratórios porque o frontend foi revisto depois de observações no teste. Esses números não são evidência confirmatória.
- O artigo registra pendência ética e de licença para redistribuir clones, identificadores e corpus derivado. Publicar o código não autoriza publicar automaticamente os áudios ou o NPZ.

## Bloqueio de rastreabilidade dos resultados

O NPZ e o test lock estão presentes localmente e o hash do NPZ corresponde ao registrado no artigo. Isso corrige a afirmação anterior de que estavam ausentes. Os dois conjuntos de resultados também foram localizados, mas continuam sem reconciliação com o artigo:

1. O run atual está em `data/results/benchmark/`. Seu `run_summary.json` declara `status: running`, embora seu Markdown e os 11 registros de modelo indiquem `ok`.
2. Esse resumo e os `results.json` ainda gravam como saída `data/results/bateria_corrigida_15k_2026-08-20/`, pasta que não existe com esse nome. As métricas limpas dos 11 registros são iguais às dos `results.json` em `data/results/benchmark/`, mas os artefatos lidos não registram a mudança de nome. O registro do RawNet2 aponta para uma pasta `rawnet2/`, que saiu do run em 2026-11-03; o resultado válido está em `_superseded/rawnet2_d9f23c0d`.
3. O run antes dado como ausente está em `data/results/archive/pre_bateria_2026-08-20/clean_benchmark_15k/`. Seu `run_summary.md` coincide com a coluna "Artigo" da tabela abaixo nas 11 entradas, no arredondamento da tabela. `data/results/paper/consolidated/benchmark_summary.json` tem as mesmas métricas e ainda referencia o caminho antigo, `data/results/clean_benchmark_15k/`, enquanto o log `consolidated/significance_rebuild.stdout.log` aponta para `bateria_corrigida_15k_2026-08-20`.
4. A coincidência indica de onde vêm os números, mas não os reconcilia com o método descrito. O `README.md` do arquivo classifica esse run como protocolo antigo, sem correção de banda e com checkpoint por `val_loss`, mantido como registro histórico e não como resultado válido. O `benchmark_summary.json` registra `val_loss` nas 9 entradas neurais, e os `results.json` arquivados não declaram correção de banda. A cópia de trabalho de `main.tex` descreve seleção por EER de validação nas entradas Keras e dá o comando com `--band-correction-hz 7500`, parâmetros que o run atual declara. Já as métricas limpas de acurácia/EER do run atual diferem materialmente das do artigo em várias entradas; WavLM Original fica próximo, enquanto HuBERT Original também apresenta diferença. O comparativo por modelo está abaixo.
5. O PDF atual é anterior à fonte LaTeX; deve ser reconstruído somente depois de corrigir a proveniência e regenerar as tabelas e figuras.

| Entrada | Artigo: acurácia / EER (%) | Run atual: acurácia / EER (%) |
| --- | ---: | ---: |
| RandomForest | 91,53 / 6,95 | 86,11 / 8,25 |
| SVM | 93,20 / 6,95 | 91,75 / 7,09 |
| CCT | 99,57 / 0,43 | 98,12 / 1,16 |
| AST | 99,71 / 0,14 | 98,55 / 0,29 |
| Res2Net | 97,76 / 2,17 | 95,80 / 4,05 |
| Conformer | 98,91 / 0,29 | 98,70 / 1,45 |
| RawNet2 | 95,88 / 3,18 | 91,17 / 8,10 |
| AASIST | 94,72 / 2,60 | 94,50 / 3,18 |
| RawGAT-ST | 87,55 / 11,87 | 84,44 / 3,91 |
| WavLM Original | 96,09 / 3,62 | 96,16 / 3,62 |
| HuBERT Original | 93,92 / 5,93 | 93,42 / 6,80 |

**Decisão de auditoria:** o artigo é a especificação metodológica escolhida, mas suas métricas ficam marcadas como não reconciliadas até que código, configuração, versão, logs, checkpoints, predições e tabelas sejam ligados ao mesmo run identificável. A coincidência com o run arquivado não autoriza apresentar esses números como resultado do protocolo descrito no artigo. Não substituir a tabela pelo run atual sem que o autor decida qual geração o artigo reporta. Uma repetição no mesmo teste já observado serve para reproduzir a execução, mas não deve ser descrita como confirmação independente; uma avaliação confirmatória nova precisa de teste novo, selado antes de qualquer decisão baseada nele.

O ranking de clean accuracy/EER publicado coloca AST (99,71% / 0,14%), CCT (99,57% / 0,43%) e Conformer (98,91% / 0,29%) no topo. Esses são resultados declarados no artigo, coincidentes com o run arquivado e ainda não reconciliados com o run atual. As entradas RF/SVM são exploratórias por ajuste posterior ao teste, conforme a ressalva do manuscrito.

## Plano de auditoria e adequação

| Prioridade | Ação | Evidência de aceite |
| --- | --- | --- |
| P0 | Revisar licença/ética e disponibilidade dos dois upstreams e do derivado; decidir se publicação distribui apenas código e receita | Decisão institucional documentada; aviso de disponibilidade de dados/licença sem alegar redistribuição autorizada |
| P0 | Reconciliar hashes do NPZ, test lock e fingerprints; confirmar por que `ab4c3a9…` e `66ffed25…` diferem; decidir qual geração o artigo reporta; registrar por que `bateria_corrigida_15k_2026-08-20` não existe com esse nome | Identificadores com algoritmo, arquivo e split; logs/checkpoints/predições ligados ao mesmo run |
| P0 | Reproduzir o cálculo das tabelas a partir dos JSON/CSV fonte e mapear de onde vem cada número da tese | Tabelas regeneradas deterministicamente e comparadas com `tabelas_benchmark.tex`; diferenças justificadas |
| P0 | Decidir se o objetivo é reprodução do artigo no split já observado ou um novo estudo confirmatório | Reprodução rotulada como reprodução; novo estudo usa teste novo e selado antes da análise |
| P1 | Reexecutar ou recuperar a bateria de 11 modelos, preservando o run antigo e mantendo todos os controles do artigo | Manifesto com commit/versões/config, 11 status terminais, hashes de entrada/saída, logs e artefatos de todos os modelos |
| P1 | Regenerar figuras/tabelas e atualizar a declaração de limitações e resultados do manuscrito | Saídas todas ligadas ao mesmo run; RF/SVM identificados como exploratórios; limites de domínio explícitos |
| P1 | Compilar o LaTeX após congelar as fontes e revisar o PDF renderizado | PDF mais novo que `main.tex`, referências/figuras resolvidas e revisão visual registrada |
| P2 | Alinhar páginas, notebooks, comandos e defaults de benchmark ao protocolo do artigo; marcar fluxos antigos como históricos | Busca de caminhos, hashes e nomes sem instruções contraditórias para o leitor do TCC |
| P2 | Auditar gates de CI, instalação limpa, segurança e arquivos licenciados para release | CI e distribuição com critérios explícitos; dependências/artefatos e licenças inventariados |
| P3 | Classificar arquivos candidatos com dependências e cópias preservadas; apresentar a lista antes de remover | Inventário aprovado pelo responsável; nenhuma remoção sem substituto, proveniência e preservação |

## Pipeline de reprodução descrito no artigo

Os scripts de dados documentam a reconstrução do corpus e do artefato; o runner operacional do benchmark está em `benchmarks/`. Para reconstruir o recorte do artigo a partir das fontes, os comandos documentados são:

```bash
python tools/dataset/build_paired_pt_corpus.py --build
python tools/dataset/build_paired_splits.py --build --min-duration-sec 3.0
python tools/dataset/audit_paired_corpus.py
python tools/dataset/export_paired_npz.py --out data/datasets/benchmark_dataset_15k.npz --target-samples 15000 --no-compress
```

O comando a seguir só deve ser usado em um arquivo novo cujo teste ainda não tenha sido utilizado; não execute contra os arquivos deste estudo, pois o lock atual é imutável e o teste já foi observado:

```bash
python tools/dataset/freeze_benchmark_test.py --dataset data/datasets/benchmark_dataset_15k.npz --declare-untouched
```

Depois de validar ou criar o test lock adequado ao tipo de estudo, execute a bateria a partir da raiz:

1. Obter as revisões upstream fixadas no artigo/protocolo e verificar termos, licenças, consentimento e restrições do XTTS antes de redistribuir qualquer áudio ou derivado.
2. Construir os pares CETUC–Fake Voices e suas partições; auditar pareamento, balanceamento, disjunção e conteúdo usando os scripts de `tools/dataset/`.
3. Exportar a subamostra de 15.000 amostras a 16 kHz/3 s e validar hash, dimensões, rótulos e metadados.
4. Criar e validar o test lock antes do treino. Não usar teste para selecionar modelos, hiperparâmetros ou correções.
5. Executar as 11 entradas oficiais com `python benchmarks/run_models_sequential.py --dataset data/datasets/benchmark_dataset_15k.npz --out data/results/<run-novo> --scope official --epochs 100 --batch-size 32 --seed 42 --device-profile gpu --latency-runs 30 --snr 30 20 10 5 --train-aug-snr 30 20 10 --train-noise-copies 1 --band-correction-hz 7500 --checkpoint-monitor val_eer`, acrescido dos parâmetros específicos registrados para cada modelo. A augmentação é aplicada na forma de onda e o teste não participa da seleção.
6. Preservar logs, configuração efetiva, hashes de dados/código/artefatos, predições e métricas. Regenerar consolidação, figuras, tabelas e manifesto do artigo a partir do mesmo run.
7. Compilar e revisar visualmente o PDF depois que a consolidação estiver rastreável; a compilação anterior não substitui essa etapa.

Antes de iniciar uma campanha cara, fechar a reconciliação dos runs existentes ou registrar formalmente por que ela não é possível. A campanha nova não deve sobrescrever os artefatos históricos.

## Inventário prévio de arquivos antigos ou candidatos a legado

Esta lista é para revisão. **Nenhum dos itens abaixo foi excluído nesta auditoria.** “Candidato” não significa que esteja comprovadamente sem uso.

| Caminho/grupo | Evidência e relação com o artigo | Proposta antes de qualquer remoção |
| --- | --- | --- |
| `data/results/paper/old_main.tex` | Fonte local ignorada; versão alternativa ao `main.tex` atual | Comparar conteúdo/citações; preservar em arquivo histórico se houver material único |
| `data/results/paper/README.md` | Ignorado pelo Git e descrito como desatualizado | Migrar instruções válidas para esta documentação; depois propor remoção |
| `data/results/paper/main.aux`, `.lof`, `.log`, `.lot`, `.out`, `.synctex.gz`, `.toc` | Produtos auxiliares do LaTeX | Regenerar a partir da fonte e remover só após validar compilação nova |
| `data/results/paper/main.pdf` | PDF existente, anterior à fonte atual | Manter como evidência histórica até novo PDF ser compilado e revisado |
| `data/results/paper/current-delivery.txt` | Arquivo local de entrega, não versionado | Verificar se registra decisão ou submissão; então arquivar/remover |
| `data/results/paper/tabelas_benchmark.tex` | Incluído pela fonte LaTeX atual | **Ativo**; regenerar após reconciliar o run, não remover |
| `data/results/paper/figures/` e `consolidated/` | Figuras/tabelas consumidas pelo artigo; métricas de proveniência mista | **Ativos para investigação**; preservar e substituir apenas com saída rastreável |
| `data/results/benchmark/` | Run atual, gravado nos resumos como `bateria_corrigida_15k_2026-08-20`. NPZ/lock coincidem com o estudo; resumo inconsistente e métricas diferem do artigo. A pasta `rawnet2/`, com um retreino abortado, foi para a quarentena em 2026-11-03 | Preservar; investigar logs, pesos, predições e configuração antes de classificá-lo |
| `data/results/benchmark/_superseded/` | `conformer_249d18d1`, `hubert_original_f4d744c7`, `hybrid_cnn_transformer_07494671`, `multiscalecnn_c81606f3`, `randomforest_6fafba2a`, `rawnet2_d9f23c0d`, `svm_9fc3a268`, `wavlm_original_93e97d9b` | `rawnet2_d9f23c0d` guarda o resultado válido do RawNet2 no run atual e **não é candidato a remoção**. Os demais são candidatos a arquivo histórico; comparar hashes, checkpoints e logs e produzir manifesto antes de remover |
| `data/results/archive/pre_bateria_2026-08-20/` | `clean_benchmark_15k/` coincide com a coluna "Artigo"; `models_benchmark_final/` guarda os artefatos promovidos desse run, com `test_split_sha256` `ab4c3a9…`; `paper/` guarda o material do TCC anterior à bateria. O `README.md` da pasta classifica o run como protocolo antigo | **Não excluir**: é onde estão os artefatos de origem dos números do artigo. Tratar como registro histórico, não como resultado válido do protocolo atual |
| `data/results/archive/benchmark_40k_2026-08-02/`, `classicos_tabular_v1_2026-08-09/`, `conformer_colapso_2026-08-09/`, `ssl_janela64000_2026-08-09/` | Rodadas e diagnósticos anteriores, com linhagem científica potencialmente única | **Não excluir como limpeza comum**; manter em arquivo frio ou preservar com manifesto caso o responsável decida retirar do checkout |
| Quarentena local, fora do repositório | `data/models/benchmark/` (geração anterior; 33 dos 34 arquivos têm cópia com o mesmo SHA-256 em `models_benchmark_final/`, e só o `README.md` não tem), `data/results/benchmark/rawnet2/` (retreino abortado na época 5), `data/temp/gradio/` e `data/models/scalers/`. Movidos em 2026-11-03, com manifesto de tamanho e SHA-256. Nada foi apagado | A exclusão definitiva é decisão do responsável; o `LEIA-ME.md` da quarentena traz os comandos para desfazer |
| `data/results/xai/shap_bar_random_forest.png`, `shap_bar_svm.png`, `shap_mean_abs.csv`, `shap_summary_random_forest.png`, `shap_summary_svm.png`, `xai_report.md` | Artefatos auxiliares locais, uso no artigo não confirmado | Procurar consumidores e relação com figuras/artigos antes de classificar |
| `data/datasets/benchmark_dataset.npz` e seu lock (ausentes neste checkout) | Exportação intermediária documentada do corpus pareado; distinta do NPZ de 15 mil avaliado | O NPZ integral não existe aqui para excluir. Não remover o código de reconstrução sem analisar dependências |
| `data/datasets/corpus/` e `data/datasets/splits/` | Presentes neste checkout: corpus pareado com 98.528 entradas e partições históricas de 40.980 amostras, incluindo a de teste em `splits/test/` | **Não excluir como limpeza comum**; não remover manifests, splits ou código de reconstrução sem analisar dependências. Auditorias de proveniência não devem ler `splits/test/` |
| `benchmarks/run_tcc_pipeline.py` | Pipeline genérico legado por tiers, diferente da receita específica CETUC–XTTS-v2 do artigo | Mantido para compatibilidade e estudos auxiliares; README, docstring e guias o distinguem do runner canônico |
| `tools/dataset/build_dataset.py`, `tools/dataset/export_npz_from_splits.py` e tiers `test/small/medium/large` | Montam conjuntos genéricos multi-fonte com split estratificado ou por falante; o exportador de tiers usa janela padrão de 5 s | Manter como aquisição/análise auxiliar; não chamar `medium` de dataset canônico do artigo. Só remover após separar essa funcionalidade do catálogo/Gradio |
| Arquivos anteriormente em `scripts/benchmark/` | Foram relocados para `benchmarks/`: `benchmark_latency.py`, `eval_cross_generator.py`, `latency_single_device.py`, `reexecutar_ssl_com_registro.sh`, `robustness_test.py`, `run_ast_ablation.py`, `run_benchmark.py`, `run_clean_benchmark_pipeline.py`, `run_models_sequential.py`, `run_rawgat_retune.py`, `run_tcc_pipeline.py`, `run_wavlm_original_benchmark.py` e `__init__.py` | Relocação com conteúdo preservado; não tratar como descarte ou código sem uso sem auditoria de chamadores |
| `docs/evaluation/experimental-study.md`, `final-ml-protocol.md`, `benchmark.md`, `pipeline-audit.md`, `retraining-adjustments.md` | Sobreposição documental e registros de etapas anteriores | Não apagar; rotular como histórico/operacional e consolidar instruções atuais por referência ao artigo |
| `docs/data/dataset-protocol.md` e `docs/data/benchmark-dataset.md` | Explicam respectivamente método reprodutível e dados do estudo | Manter; alinhar ambos à distinção corpus pai vs. subamostra de 15 mil |
| `docs/notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb`, `02_training_model.ipynb`, `04_all_architectures_full_benchmark.ipynb` | Cadernos usam runner genérico, tiers, outra duração ou caminhos antigos; execução completa é controlada por flags | Manter como material histórico/educacional, conferir células e saída antes de atualizar; não anunciar como reprodução do artigo |
| `docs/evaluation/retraining-wsl2.md` e comandos de tier em `docs/models/training.md`, `docs/data/public-datasets.md`, `docs/deployment/hugging-face-spaces.md` | Descrevem campanhas e entrada de dataset diferentes | Marcados como fluxo histórico; remover somente após decidir se ainda servem para outros usuários |
| referências a `scripts/benchmark/` | Caminhos ativos no artigo, gerador e notebooks foram migrados para `benchmarks/` | Preservar referências antigas em logs, fingerprints e arquivos de auditoria: elas identificam o código que produziu os artefatos históricos |

### Gate para eventual limpeza

Antes de excluir qualquer item, registrar no inventário: caminho, estado Git, hash/tamanho, chamadores/imports/links encontrados, conteúdo exclusivo, substituto, decisão e responsável. Para artefatos científicos, adicionar manifesto e cópia de preservação. Só então apresentar a lista final ao responsável pelo TCC para decisão de remoção. Nesta etapa, nenhum arquivo foi apagado.
