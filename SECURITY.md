# Política de Segurança

## Versões Suportadas

Atualmente, apenas a versão mais recente do XfakeSong recebe atualizações de segurança.

| Versão | Suportada          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reportando uma Vulnerabilidade

Levamos a segurança do nosso software muito a sério. Se você encontrar alguma vulnerabilidade, por favor, siga os passos abaixo:

1.  **NÃO** abra uma issue pública no GitHub.
2.  Envie um e-mail para `security@xfakesong.com` (ou o e-mail do mantenedor).
3.  Descreva a vulnerabilidade com detalhes suficientes para reprodução.

Faremos o possível para responder rapidamente e corrigir o problema antes de torná-lo público.

## Práticas de Segurança no Projeto

Este projeto adota as seguintes práticas para garantir a segurança:

*   **Análise Estática (SAST)**: Utilizamos CodeQL e Bandit para identificar vulnerabilidades no código.
*   **Dependências**: Monitoramos dependências vulneráveis via Dependabot.
*   **Container**: As imagens Docker são construídas minimizando a superfície de ataque (imagens slim).
