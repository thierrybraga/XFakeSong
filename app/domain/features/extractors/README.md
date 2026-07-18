# extractors

Implementacoes de extracao acustica.

## Responsabilidade

Agrupa extratores por familia do dominio acustico, preservando coesao entre
componentes matematicos relacionados.

## Quando usar

Use para implementar algoritmos de DSP e features que podem ser chamados pelo
registry.

## Familias

- `cepstral/`: MFCC, PLP, LPCC e derivados.
- `spectral/`: centroide, rolloff, contraste e magnitude.
- `temporal/`: energia, envelope, dinamica e estatisticas temporais.
- `prosodic/`: pitch, jitter, shimmer e qualidade prosodica.
- `formant/`: LPC e formantes.
- `voice_quality/`: perturbacao e ruido vocal.
- `complexity/`: entropia, fractais e caos.
- `perceptual/`: escalas psicoacusticas e loudness.
- `speech/`: medidas especificas de fala.
- `timefreq/` e `transform/`: transformadas tempo-frequencia.
- `mel/`: espectrograma mel.
