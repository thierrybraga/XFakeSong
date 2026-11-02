# Contributing to SpeechChecker

Obrigado por contribuir com o SpeechChecker. Este arquivo mantem apenas o fluxo de
contribuicao; as regras tecnicas detalhadas ficam nos documentos canonicos:

- [Guia do Desenvolvedor](docs/development/developer-guide.md)
- [Qualidade e Testes](docs/development/quality-and-testing.md)
- [Arquitetura](docs/architecture/overview.md)
- [Codigo de Conduta](CODE_OF_CONDUCT.md)

## Fluxo de Contribuicao

1. Verifique se ja existe uma issue ou pull request sobre o tema.
2. Abra uma issue quando a mudanca alterar comportamento, arquitetura ou API.
3. Crie uma branch descritiva a partir da base atual.
4. Instale as dependencias e rode a aplicacao localmente.
5. Faca mudancas pequenas, testaveis e alinhadas a Clean Architecture.
6. Rode os testes relevantes antes de abrir o pull request.
7. Descreva no PR o problema, a solucao e a validacao executada.

## Setup Local

### Windows (PowerShell)

```powershell
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main.py --bootstrap-dirs
.\.venv\Scripts\python.exe main.py --gradio
```

### Linux / macOS

```bash
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python main.py --bootstrap-dirs
python main.py --gradio
```

Os alvos `make` usam Bash. No Windows, execute-os no Git Bash ou WSL; no
PowerShell, use os comandos nativos acima.
Para editar documentação ou notebooks, instale também `requirements-docs.txt`
ou `requirements-notebooks.txt`, conforme o perfil descrito em
[`docs/development/release-gates.md`](docs/development/release-gates.md).

## Validacao Basica

No Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
.\.venv\Scripts\python.exe -m pytest --cov=app --cov=benchmarks tests/
```

Linux/macOS, ou Windows pelo Git Bash/WSL:

```bash
pytest tests/
pytest --cov=app --cov=benchmarks tests/
```

Use testes mais focados durante o desenvolvimento:

```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/api/
pytest tests/functional/
```

## Padroes Essenciais

- Codigo de dominio fica em `app/domain/` e nao deve depender de interfaces,
  roteadores ou frameworks de entrada.
- Use `logging.getLogger(__name__)`; evite `print()` em codigo de aplicacao.
- Adicione type hints em funcoes publicas.
- Atualize documentacao e testes quando mudar contratos publicos.
- Nao inclua datasets, modelos treinados grandes, caches ou segredos no PR.

## Reporte de Bugs e Sugestoes

Ao abrir uma issue, inclua:

- versao do Python e sistema operacional;
- comando executado;
- stack trace ou log relevante;
- passos para reproduzir;
- resultado esperado e resultado observado.
