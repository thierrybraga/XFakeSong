# Pacote Overleaf - TCC XFakeSong

Faça upload de todos os arquivos desta pasta no Overleaf e compile `tcc.tex` com pdfLaTeX.

Conteúdo:
- `tcc.tex`: versão otimizada do TCC com caminhos relativos.
- `figures/`: gráficos consolidados do benchmark utilizados no texto.
- `figures/confusion_matrices/`: matrizes de confusão por arquitetura.

Origem dos dados:
- Dataset do benchmark atual: `app/datasets/benchmark_audio_raw_balanced_15k.npz`.
- Modelos finais default da aplicação: `app/models/bench_*`.
- Modelos completos por arquitetura: `app/models/benchmark_final/`.
- Métricas e relatórios completos: `results/`.

O pacote `tcc_overleaf.zip` deve conter apenas `tcc.tex`, este README e as
figuras. Não inclua PDFs ou arquivos auxiliares (`.aux`, `.log`, `.out`,
`.toc`) no upload.

Observação: os placeholders da banca permanecem no texto para preenchimento final.
