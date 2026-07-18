# adapters

Adaptadores dos extratores de features.

## Responsabilidade

Conecta implementacoes concretas de extratores ao contrato comum usado pelo
servico de extracao.

## Quando usar

Use quando um extrator existente precisa ser exposto por `FeatureType` sem
alterar sua implementacao interna.

## Arquivos

Cada arquivo corresponde a uma familia de features: cepstral, spectral,
prosodic, temporal, formant, perceptual, predictive, speech, time-frequency,
transform, complexity e voice quality.
