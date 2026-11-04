# Auditoria de dataset e artefatos treinados

**Data da verificação:** 2026-10-17  
**Referência metodológica:** `data/results/paper/main.tex`  
**Escopo:** dataset de 15 mil amostras, test lock, fingerprints, modelos locais,
manifestos, métricas do artigo e execução em `data/results/benchmark_15k_2026-08-20/`.

## Veredito

O dataset do artigo está presente e passou na verificação criptográfica: o SHA-256
real coincide com o test lock, assim como a identidade dos membros ZIP. O dataset
tem 15.000 amostras e partições 12.162/1.456/1.382.

Há duas coleções diferentes de checkpoints. Os 11 modelos em
`data/models/benchmark_final/` têm hashes íntegros em relação aos manifestos e
métricas iguais às tabelas do artigo, mas os manifestos identificam outro
identificador de teste (`ab4c…`), sem registrar o método usado para calculá-lo;
ele difere do fingerprint (`66ff…`) presente no artigo e nos resultados locais.
Os manifestos apontam para diretórios de origem que não existem neste checkout
e declaram `results_copied: false`. Assim, não é possível provar que esse
identificador representa o mesmo split nem ligar os pesos às predições, logs e
resultados que sustentam o artigo.

A execução local em `data/results/benchmark_15k_2026-08-20/` tem 11 entradas
`ok` no resumo agregado, seed 42, 100 épocas para os modelos neurais, dados e
fingerprint de teste iguais aos registrados no artigo (`66ff…`). Há 10 arquivos
`results.json` por modelo e 10 checkpoints correspondentes sob a árvore de
resultados. O resumo contém as métricas de RawNet2, mas seu checkpoint final não
está no diretório de saída que o resumo referencia. Existe um `bench_rawnet2.keras`
solto em `data/models/`, sem hash de proveniência que o ligue a essa saída. As
métricas locais diferem das publicadas. O resumo da execução ainda marca
`status: running` e referencia saídas sob `bateria_corrigida_15k_2026-08-20/`,
diretório ausente. Portanto, esta é uma execução local parcialmente preservada,
mas não uma reprodução completa dos números do artigo.

**Conclusão:** nenhuma das duas coleções pode ser declarada, com as evidências
presentes, como reprodução completa dos resultados do artigo. O conjunto
`benchmark_final` não deve ser usado como evidência reproduzível até recuperar
sua origem; o conjunto local não deve substituir as métricas publicadas sem
atualizar e revisar o artigo. Os arquivos foram preservados, sem promoção,
substituição ou exclusão.

## Dataset e selos verificados

| Evidência | Valor observado | Resultado |
| --- | --- | --- |
| Arquivo | `data/datasets/benchmark_dataset_15k.npz` | presente |
| Tamanho | 2.925.524.332 bytes | igual ao lock |
| SHA-256 do arquivo | `3775b35fb05155011b21a30ff60632aaa7c9f67f031bd6f7874a5f313363400f` | igual ao lock e ao artigo |
| Identidade ZIP dos membros de teste | `4f9b86306d4fa5221ee2aacce9bd0ed5fd9daa22b47640ffec996ef60efabab6` | igual ao lock e ao artigo |
| Contagens | treino 12.162; validação 1.456; teste 1.382 | iguais ao lock e ao artigo |
| Fingerprint de teste dos resultados locais | `66ffed2572294062c6d1a5855b278599e0c8972b9ebfed98afecbd1f620ea5d9` | igual ao artigo e às 11 entradas do resumo local |
| Fingerprint em `benchmark_final` | `ab4c3a9fa97982624a21fdecce358c4fa046e6bfe37a01620c0fb561b16dd60a` | divergente dos resultados locais e do fingerprint descrito no artigo |

O lock declara que o teste foi selado antes do treino. A declaração e os hashes
confirmam a identidade dos arquivos, mas não provam independentemente o
histórico de acesso ao teste.

## Métricas: artigo versus execução local

Os valores do artigo abaixo coincidem com `metrics` dos manifestos em
`benchmark_final`. A execução local usa o mesmo dataset e fingerprint de teste,
mas não obteve os mesmos resultados.

| Modelo | Artigo: acurácia / EER (%) | Run local: acurácia / EER (%) |
| --- | ---: | ---: |
| AASIST | 94,72 / 2,60 | 94,50 / 3,18 |
| Conformer | 98,91 / 0,29 | 98,70 / 1,45 |
| HuBERT Original | 93,92 / 5,93 | 93,42 / 6,80 |
| Hybrid CNN-Transformer | 99,57 / 0,43 | 98,12 / 1,16 |
| MultiscaleCNN (Res2Net) | 97,76 / 2,17 | 95,80 / 4,05 |
| RandomForest | 91,53 / 6,95 | 86,11 / 8,25 |
| RawGAT-ST | 87,55 / 11,87 | 84,44 / 3,91 |
| RawNet2 | 95,88 / 3,18 | 91,17 / 8,10 |
| SpectrogramTransformer (AST) | 99,71 / 0,14 | 98,55 / 0,29 |
| SVM | 93,20 / 6,95 | 91,75 / 7,09 |
| WavLM Original | 96,09 / 3,62 | 96,16 / 3,62 |

As diferenças não são explicadas por uma troca de dataset: os resultados locais
registram o mesmo SHA-256 do NPZ, as mesmas contagens e o fingerprint `66ff…`.
É necessária a reconciliação de pesos, configuração efetiva, predições e código
antes de decidir se o artigo deve ser reproduzido ou corrigido.

* Para RawNet2, as métricas estão no resumo agregado e no manifesto promovido;
  o diretório local não contém `results.json` nem o checkpoint final indicado
  pelo resumo.

## Hashes dos checkpoints

Cada digest foi calculado sobre o arquivo local. A coluna `benchmark_final` foi
comparada também com `model_sha256` do manifesto de sua arquitetura; os 11
valores coincidem. A coluna `run local` corresponde aos dez checkpoints
encontrados sob
`data/results/benchmark_15k_2026-08-20/<modelo>/architectures/<modelo>/models/`;
para RawNet2, registra separadamente o hash do arquivo solto sem proveniência
confirmada.

| Modelo | SHA-256 `benchmark_final` | SHA-256 run local |
| --- | --- | --- |
| AASIST | `d8731bd60e55e5bcdd5ff439c2bf9acba20532016fe708031debb9fafdd6ea3d` | `4ea58f03c1b78f8d8f5c58b36d435506244dc6c0573abb77b04ac7795a686b45` |
| Conformer | `9b75d7420e7272389a0371b3f348c0c333333b1e852a0db28fe444f61a7568e2` | `5af14b27aacdfc4e9cbd270e55c1f2dee2280542a299406d3f1002231ee7a8db` |
| HuBERT Original | `ab4738abca45c406be1a9b5de5131e216ac13038aeaf0e75096ec2fc92686441` | `373a60a1b243e9ed6086d38559fea51975486a60dd863d1d0327f9ce845d79f7` |
| Hybrid CNN-Transformer | `c0d94daa3ee71996c86e609a576bd56eb03f1d308f6f5ad6badd21bd7de80d20` | `4a200081c04e7339e9ddb7fbf52700df4d87c912b43e6754a416b4e4571c1d6d` |
| MultiscaleCNN | `6864559fe09951fdeeae34b46b8dbf7593ce024721b6429761039504a5c0279f` | `8bfb8ff59e3bf69b9bf6d107de56f96293b779cd9ddac37750b270442e264d3a` |
| RandomForest | `92eca70fb55a570edbe88283d524676e825c28f36b2355e2202add82ce5a0236` | `438ba873beb0ccebc6b05c04650cb736a08ad71f9deedf3ffa9f4c6ead725292` |
| RawGAT-ST | `f5f5ec86ef9718c40498302b0c82198c7e2da9e9926d44bd711d7216313e84c6` | `909442ec4ed51ee6e2eb651cac22176c5708c1d265d60d8c8824a9d496962224` |
| RawNet2 | `7d0e0dcb0e0f3ef3c52f4653290dbf95e510db3d36e5a0f0ea941773cb56e2f9` | saída ausente; arquivo solto candidato: `212c2922e7fe90d17e6447ccf1a82c67e87b48d24edd5e911704a3ed4eff35cf` |
| SpectrogramTransformer | `b3b8a2f64637cf327c94f900de41ece718853f508ee60abe4d7acb1a3394f351` | `b1b50159e05dbace88af9f36b94366e45ba30e6000e2294dbce34d3b8e454824` |
| SVM | `dbc6c1ce7555b1013f0bc86f30706895139f25682e10c798e18c8cdcaf71755b` | `02b0df76e2093b9bb530042ffc619b67f681c74666064aa25eed2c50687e9674` |
| WavLM Original | `9302acb27e47ead991598cba0515ac8e45ed95878eea0e7846ec1beb0d425749` | `4a8f3e7435b107fed8abb93329e21c7355fbdbfa8b71fa10776ded3a298d5b2c` |

Os dez checkpoints encontrados sob a árvore do run local têm hashes diferentes
dos checkpoints promovidos correspondentes. O arquivo solto
`data/models/bench_rawnet2.keras` também difere do RawNet2 promovido, mas não tem
proveniência demonstrada. Oito arquivos soltos `data/models/bench_*` (AASIST,
Conformer, Hybrid, Res2Net, RF, RawGAT-ST, AST e SVM) são cópias byte a byte dos
checkpoints encontrados no run local. Os dois modelos SSL têm checkpoints na
árvore do run, mas não têm cópias soltas na raiz de `data/models/`.

## Integridade e proveniência não são a mesma coisa

- **Integridade confirmada:** SHA-256 do dataset versus test lock; identidade
  ZIP versus lock; checkpoints em `benchmark_final` versus manifestos; hashes
  dos dez checkpoints locais encontrados versus seus arquivos.
- **Protocolo local compatível:** o run local usa 15 mil amostras, split selado,
  seed 42, 100 épocas para redes neurais, 11 modelos oficiais, dados a 16 kHz e
  janela de 48.000 amostras; seus relatórios dizem que o test lock foi validado.
- **Proveniência incompleta:** fingerprints de execução registram `git commit:
  null` e `dirty: null` porque Git não estava disponível no container; o resumo
  ainda diz `running`; os caminhos de saída do resumo não existem; os hashes de
  código referem caminhos antigos sob `scripts/benchmark/`.
- **Modelos promovidos não reconciliados:** todos os manifestos em
  `benchmark_final` dizem `results_copied: false` e apontam para a antiga raiz
  `D:\XFakeSong`; registram o identificador `ab4c…`, sem método, em vez do
  fingerprint `66ff…`.
- **Risco de execução:** `model_loader.py` dá prioridade a
  `data/models/benchmark_final/`, portanto essa é a coleção carregada pela
  aplicação quando disponível.
- **Métricas não reproduzidas:** a execução local está ligada ao dataset correto,
  mas difere materialmente do artigo em 10 das 11 acurácias/EER publicadas.

## Consolidação recomendada

1. Preservar `benchmark_final/` e o run local como coleções históricas distintas;
   não copiar um conjunto sobre o outro nem excluir os arquivos soltos.
2. Recuperar o diretório de origem `clean_benchmark_15k` ou seus resultados
   exportados, predições, logs, configurações e checkpoints originais. Recalcular
   os hashes e explicar o fingerprint `ab4c…` antes de manter as métricas do
   artigo.
3. Corrigir o resumo local a partir das dez saídas `results.json` e da entrada
   RawNet2 agregada; recuperar ou marcar explicitamente como ausentes os
   artefatos RawNet2. Validar caminhos e gerar um manifesto único com hash do
   dataset, test lock, fingerprint `66ff…`, config, commit/código e hash de cada
   checkpoint.
4. Comparar novamente modelos, predições e configuração, e então escolher uma
   decisão editorial: reproduzir os números do artigo com a coleção de origem
   recuperada ou atualizar artigo/tabelas para a execução local com limitações
   explícitas. Não apresentar a repetição no mesmo teste como confirmação
   independente.
5. Só depois da reconciliação promover uma coleção única para
   `data/models/benchmark_final/` e atualizar os arquivos que servem inferência.

Esta auditoria não valida a licença de redistribuição dos áudios/modelos,
execução de inferência de cada checkpoint ou compilação do artigo; esses pontos
continuam sujeitos às verificações de release já documentadas.
