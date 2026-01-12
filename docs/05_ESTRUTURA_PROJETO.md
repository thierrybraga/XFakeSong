# Estrutura de Arquivos e Pastas

Este documento fornece um mapa detalhado da estrutura do projeto XfakeSong para facilitar a navegação e entendimento.

## Estrutura de Diretórios

```
TCC/
├── app/                        # Código fonte principal
│   ├── application/            # Camada de Aplicação (Use Cases, Pipeline)
│   │   ├── dto/                # Data Transfer Objects
│   │   ├── pipeline/           # Orquestração do fluxo de processamento
│   │   │   ├── stages/         # Estágios individuais (Upload, Extraction, etc)
│   │   │   └── workflows/      # Definições de fluxos de trabalho
│   │   └── use_cases/          # Casos de uso específicos
│   │
│   ├── config/                 # Arquivos de configuração (YAML, JSON)
│   │
│   ├── controllers/            # Controladores (Adapters)
│   │   ├── api_controller.py   # Controlador para API REST
│   │   └── main_controller.py  # Controlador principal
│   │
│   ├── core/                   # Núcleo e Infraestrutura
│   │   ├── config/             # Configurações do sistema (Settings)
│   │   ├── exceptions/         # Exceções customizadas
│   │   ├── interfaces/         # Interfaces base (Abstrações)
│   │   ├── training/           # Pipelines de treinamento seguro
│   │   ├── utils/              # Utilitários (Audio, File, System)
│   │   └── db_setup.py         # Configuração de banco de dados
│   │
│   ├── datasets/               # Dados para treinamento/teste
│   │   ├── fake/               # Amostras de áudio deepfake
│   │   └── real/               # Amostras de áudio real
│   │
│   ├── domain/                 # Camada de Domínio (Regras de Negócio)
│   │   └── features/           # Núcleo de extração de features
│   │       ├── adapters/       # Adaptadores para diferentes bibliotecas
│   │       ├── exporters/      # Exportação de dados (CSV, JSON)
│   │       ├── extractors/     # Implementações dos algoritmos (Cepstral, etc)
│   │       └── extractor_registry.py # Registro dinâmico de extratores
│   │
│   └── dependencies.py         # Injeção de dependências
│
├── docs/                       # Documentação do projeto (Gerada neste passo)
├── logs/                       # Arquivos de log do sistema
├── .env.example                # Modelo de variáveis de ambiente
├── main.py                     # Ponto de entrada da aplicação
├── README.md                   # Documentação principal
└── requirements.txt            # Dependências Python
```

## Descrição dos Principais Diretórios

### `app/application/pipeline`
Contém a lógica sequencial do sistema. O `orchestrator.py` é responsável por encadear os estágios (`stages`) definidos aqui.

### `app/domain/features`
Onde a "mágica" acontece.
- **`extractors/`**: Contém a matemática pesada para extrair MFCCs, entropia, fractais, etc.
- **`adapters/`**: Garante que a saída dos extratores esteja num formato padrão para o sistema.

### `app/core/interfaces`
Define os contratos que garantem o desacoplamento. Se você quer saber o que uma classe *deve* fazer, olhe aqui.

### `datasets/`
Diretório padrão para armazenamento de áudios. O sistema espera encontrar subpastas `real` e `fake` para treinamento supervisionado.
