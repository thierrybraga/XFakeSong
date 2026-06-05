# Contributing to XfakeSong

Obrigado por contribuir com o XfakeSong. Este arquivo mantem apenas o fluxo de
contribuicao; as regras tecnicas detalhadas ficam nos documentos canonicos:

- [Guia do Desenvolvedor](docs/05_GUIA_DEV.md)
- [Testes e Qualidade](docs/06_TESTES.md)
- [Arquitetura](docs/03_ARQUITETURA.md)
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

```bash
git clone https://github.com/YOUR_USERNAME/XFakeSong.git
cd XFakeSong
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --bootstrap-dirs
python main.py --gradio
```

## Validacao Basica

```bash
pytest tests/
pytest --cov=app tests/
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
