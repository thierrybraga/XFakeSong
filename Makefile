# =====================================================================
# XFakeSong — Makefile
# Auto-detecta `docker compose` (v2) ou `docker-compose` (v1)
# Compatível com Make 3.81+ no Windows (Git Bash / MinGW / WSL)
# =====================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory

# Auto-detect Docker Compose v2 vs v1
DC := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

PYTHON ?= python3
PORT ?= 7860
IMAGE_NAME := xfakesong
IMAGE_TAG ?= latest

# GPU profile (set GPU=1)
COMPOSE_FILES := -f docker-compose.yml
ifeq ($(GPU),1)
COMPOSE_FILES += -f docker-compose.gpu.yml
endif

# =====================================================================
.PHONY: help
help:  ## Mostra esta ajuda
	@echo "XFakeSong — comandos disponíveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variáveis:"
	@echo "  GPU=1               Habilita GPU (NVIDIA)"
	@echo "  PORT=7860           Porta Gradio (default 7860)"
	@echo "  PYTHON=python3      Binário Python para venv local"

# =====================================================================
# Docker
# =====================================================================

.PHONY: build
build:  ## Build da imagem Docker (com cache)
	$(DC) $(COMPOSE_FILES) build

.PHONY: build-nocache
build-nocache:  ## Build sem cache + pull base images
	$(DC) $(COMPOSE_FILES) build --no-cache --pull

.PHONY: up
up:  ## Sobe containers em background
	$(DC) $(COMPOSE_FILES) up -d
	@echo ""
	@echo "Aplicação em http://localhost:$(PORT)"

.PHONY: down
down:  ## Para e remove containers
	$(DC) down

.PHONY: restart
restart: down up  ## Reinicia containers

.PHONY: rebuild
rebuild: down build-nocache up  ## Rebuild completo + restart

.PHONY: logs
logs:  ## Tail dos logs (follow)
	$(DC) logs -f --tail=100 app

.PHONY: ps
ps:  ## Status do compose
	$(DC) ps

.PHONY: shell
shell:  ## Abre shell no container
	$(DC) exec app /bin/bash

.PHONY: stats
stats:  ## Stats em tempo real
	docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

.PHONY: health
health:  ## Status do healthcheck
	@docker inspect -f '{{.State.Health.Status}}' xfakesong_app 2>/dev/null || echo "Container não encontrado"

.PHONY: clean
clean:  ## Down + remove volumes anônimos + prune
	$(DC) down -v --remove-orphans
	docker image prune -f
	docker builder prune -f

# =====================================================================
# Desenvolvimento local (Python venv)
# =====================================================================

.PHONY: venv
venv:  ## Cria virtualenv local em .venv
	$(PYTHON) -m venv .venv
	@echo "Ative com:  source .venv/bin/activate  (ou .venv\\Scripts\\activate no Windows)"

.PHONY: install
install:  ## Instala dependências em .venv (cria se não existe)
	@if [ ! -d .venv ]; then $(PYTHON) -m venv .venv; fi
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

.PHONY: install-dev
install-dev: install  ## Instala dependências + dev tools
	.venv/bin/pip install -r requirements-dev.txt

.PHONY: dev
dev:  ## Roda Gradio local (sem Docker)
	$(PYTHON) main.py --gradio --gradio-port $(PORT)

.PHONY: bootstrap
bootstrap:  ## Cria estrutura de diretórios padrão
	$(PYTHON) main.py --bootstrap-dirs

# =====================================================================
# Qualidade de código
# =====================================================================

.PHONY: lint
lint:  ## Roda ruff lint
	.venv/bin/ruff check app/ tests/

.PHONY: format
format:  ## Formata com black + isort
	.venv/bin/black app/ tests/
	.venv/bin/isort app/ tests/

# Runner padronizado (escolhe o python do venv). Veja docs/06_TESTES.md.
RUN_TESTS := ./scripts/run_tests.sh

.PHONY: test
test:  ## Roda a suíte rápida (tudo EXCETO smoke) — run padrão da CI
	$(RUN_TESTS) fast

.PHONY: test-unit
test-unit:  ## Só testes unitários
	$(RUN_TESTS) unit

.PHONY: test-api
test-api:  ## Só contrato da API REST
	$(RUN_TESTS) api

.PHONY: test-functional
test-functional:  ## Só fluxos de frontend/usuário
	$(RUN_TESTS) functional

.PHONY: test-integration
test-integration:  ## Só integração (pode treinar modelos — mais lento)
	$(RUN_TESTS) integration

.PHONY: test-smoke
test-smoke:  ## Smoke pesado com TF real (14 archs + wizard + API)
	$(RUN_TESTS) smoke

.PHONY: test-all
test-all:  ## TUDO, inclusive smoke
	$(RUN_TESTS) all

.PHONY: test-cov
test-cov:  ## Suíte rápida + coverage (app + benchmarks)
	$(RUN_TESTS) cov

# =====================================================================
# Utilitários
# =====================================================================

.PHONY: env
env:  ## Cria .env a partir de .env.example
	@if [ -f .env ]; then \
	    echo ".env já existe — abortando para não sobrescrever."; \
	    exit 1; \
	fi
	cp .env.example .env
	@echo "✓ .env criado. Edite conforme necessário."

.PHONY: version
version:  ## Mostra versões críticas
	@echo "Make:          $$(make --version | head -n1)"
	@echo "Docker:        $$(docker --version 2>/dev/null || echo 'não instalado')"
	@echo "Docker Compose: $(DC) ($$($(DC) version --short 2>/dev/null || echo 'n/a'))"
	@echo "Python local:  $$($(PYTHON) --version 2>/dev/null || echo 'não instalado')"

.PHONY: deploy-hf
deploy-hf:  ## Deploy para Hugging Face Spaces
	$(PYTHON) main.py --deploy

.PHONY: doctor
doctor:  ## Diagnóstico do ambiente (deps, porta, db)
	$(PYTHON) scripts/doctor.py

.PHONY: doctor-fix
doctor-fix:  ## Diagnóstico + tenta corrigir automaticamente
	$(PYTHON) scripts/doctor.py --fix
