# feature_extraction

Nucleo de orquestracao de extracao de features.

## Responsabilidade

Separa carregamento de extratores, validacao de configuracao e execucao do core
de extracao.

## Quando usar

Use quando precisar alterar o fluxo de extracao sem tocar diretamente nos
extratores acusticos.

## Arquivos

- `core.py`: executa extratores solicitados.
- `extractor_loader.py`: registra e carrega extratores disponiveis.
- `types.py`: configuracao de extracao.
- `validator.py`: validacoes de entrada e configuracao.
