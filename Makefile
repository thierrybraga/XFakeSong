# =====================================================================
# SpeechChecker — Makefile
# Auto-detecta `docker compose` (v2) ou `docker-compose` (v1)
# Requer GNU Make e Bash. No Windows, use Git Bash ou WSL; PowerShell usa
# os comandos documentados nos guias, não os alvos Make.
# =====================================================================

ifeq ($(OS),Windows_NT)
SHELL := bash
PYTHON ?= python
VENV_PYTHON := .venv/Scripts/python.exe
VENV_ACTIVATION_MSG := Git Bash: source .venv/Scripts/activate; PowerShell: .venv/Scripts/Activate.ps1
else
SHELL := /bin/bash
PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
VENV_ACTIVATION_MSG := source .venv/bin/activate
endif
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory

# Auto-detect Docker Compose v2 vs v1
DC := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Só de `make dev`. Cada perfil Docker publica a sua: 7860 (GPU=1) e 7861 (CPU).
PORT ?= 7862
IMAGE_NAME := speechchecker
IMAGE_TAG ?= latest

# Perfis de desenvolvimento/treino ficam em infrastructure/docker/compose/.
# O Dockerfile da raiz e mantido para deploy em Hugging Face Spaces.
SERVICE ?= inference-api
COMPOSE_FILE := infrastructure/docker/compose/inference.cpu.yml
RELEASE_PROFILE := cpu
OTHER_CONTAINER := speechchecker_inference_nvidia
ifeq ($(GPU),1)
COMPOSE_FILE := infrastructure/docker/compose/inference.nvidia.yml
RELEASE_PROFILE := nvidia
OTHER_CONTAINER := speechchecker_inference_cpu
endif
# Sem `--env-file`, o Compose procura o `.env` na pasta do arquivo compose, e não
# na raiz: a inferência recusa subir e treino/benchmark ignoram o arquivo em silêncio.
ENV_FILE ?= .env
ENV_FLAG := $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE))
COMPOSE_FILES := $(ENV_FLAG) -f $(COMPOSE_FILE)

TRAIN_CPU_COMPOSE := infrastructure/docker/compose/train.cpu.yml
TRAIN_NVIDIA_COMPOSE := infrastructure/docker/compose/train.nvidia.yml
BENCHMARK_NVIDIA_COMPOSE := infrastructure/docker/compose/benchmark.nvidia.yml

# =====================================================================
.PHONY: help
help:  ## Mostra esta ajuda
	@echo "SpeechChecker — comandos disponíveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variáveis:"
	@echo "  GPU=1               Habilita GPU (NVIDIA)"
	@echo "  PORT=7862           Porta de make dev (Docker: GRADIO_PORT_NVIDIA e GRADIO_PORT_CPU)"
	@echo "  SERVICE=inference-api Serviço Compose alvo"
	@echo "  PYTHON              Python base (Windows: python; Linux/macOS: python3)"

# =====================================================================
# Docker
# =====================================================================

.PHONY: build
build:  ## Build da imagem Docker (com cache)
	$(DC) $(COMPOSE_FILES) build $(SERVICE)

.PHONY: build-nocache
build-nocache:  ## Build sem cache + pull base images
	$(DC) $(COMPOSE_FILES) build --no-cache --pull $(SERVICE)

# Os dois perfis montam o mesmo data/ e gravariam no mesmo SQLite.
.PHONY: check-other-instance
check-other-instance:
	@if [ "$$(docker inspect -f '{{.State.Running}}' $(OTHER_CONTAINER) 2>/dev/null)" = "true" ]; then \
	    echo "$(OTHER_CONTAINER) está rodando com o mesmo data/. Pare-o antes: docker stop $(OTHER_CONTAINER)"; \
	    exit 1; \
	fi

.PHONY: up
up: check-other-instance  ## Sobe containers em background
	$(DC) $(COMPOSE_FILES) up -d $(SERVICE)
	@echo ""
	@echo "Aplicação em http://$$($(DC) $(COMPOSE_FILES) port $(SERVICE) 7860)"

.PHONY: down
down:  ## Para e remove containers
	$(DC) $(COMPOSE_FILES) down

.PHONY: restart
restart: down up  ## Reinicia containers

.PHONY: rebuild
rebuild: down build-nocache up  ## Rebuild completo + restart

.PHONY: logs
logs:  ## Tail dos logs (follow)
	$(DC) $(COMPOSE_FILES) logs -f --tail=100 $(SERVICE)

.PHONY: ps
ps:  ## Status do compose
	$(DC) $(COMPOSE_FILES) ps

.PHONY: shell
shell:  ## Abre shell no container
	$(DC) $(COMPOSE_FILES) exec $(SERVICE) /bin/bash

.PHONY: stats
stats:  ## Stats em tempo real
	docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

.PHONY: health
health:  ## Status do healthcheck
	@docker inspect -f '{{.State.Status}}' speechchecker_inference_cpu 2>/dev/null || \
	docker inspect -f '{{.State.Status}}' speechchecker_inference_nvidia 2>/dev/null || \
	echo "Container não encontrado"

# Só o que é do projeto. `docker image prune` sem filtro e `docker builder prune`
# alcançam também imagens e cache de outros projetos da máquina.
.PHONY: clean
clean:  ## Down + remove volumes anônimos + imagens órfãs deste projeto
	$(DC) $(COMPOSE_FILES) down -v --remove-orphans
	docker image prune -f --filter label=org.opencontainers.image.title=speechchecker-inference-api

# O caminho de volta: `make down snapshot build up`; se der errado, `make rollback`.
.PHONY: snapshot
snapshot:  ## Guarda a imagem atual e copia o banco (com o container parado)
	"$(VENV_PYTHON)" tools/ops/docker_release.py snapshot $(RELEASE_PROFILE)

.PHONY: rollback
rollback:  ## Volta para a imagem guardada e espera o health check
	"$(VENV_PYTHON)" tools/ops/docker_release.py rollback $(RELEASE_PROFILE)

.PHONY: train-cpu
train-cpu:  ## Executa perfil de treino CPU/classical
	$(DC) $(ENV_FLAG) -f $(TRAIN_CPU_COMPOSE) run --rm classical-ml

.PHONY: train-nvidia
train-nvidia:  ## Executa perfil de treino NVIDIA TensorFlow/Keras
	$(DC) $(ENV_FLAG) -f $(TRAIN_NVIDIA_COMPOSE) run --rm tensorflow-keras

.PHONY: train-pytorch
train-pytorch:  ## Executa perfil de treino NVIDIA PyTorch (RawNet2/AASIST/RawGAT-ST/Conformer/Hybrid)
	$(DC) $(ENV_FLAG) -f $(TRAIN_NVIDIA_COMPOSE) run --rm pytorch-audio

.PHONY: train-ssl
train-ssl:  ## Executa perfil de treino NVIDIA SSL (WavLM/HuBERT originais)
	$(DC) $(ENV_FLAG) -f $(TRAIN_NVIDIA_COMPOSE) run --rm ssl-transformers

.PHONY: benchmark-nvidia
benchmark-nvidia:  ## Executa benchmark completo NVIDIA/WSL2
	$(DC) $(ENV_FLAG) -f $(BENCHMARK_NVIDIA_COMPOSE) run --rm benchmark

.PHONY: docker-config
docker-config:  ## Valida configuração Compose dos perfis segmentados
	"$(VENV_PYTHON)" tools/ops/docker_build.py inference-cpu config
	"$(VENV_PYTHON)" tools/ops/docker_build.py inference-nvidia config
	"$(VENV_PYTHON)" tools/ops/docker_build.py train-cpu config
	"$(VENV_PYTHON)" tools/ops/docker_build.py train-nvidia config
	"$(VENV_PYTHON)" tools/ops/docker_build.py benchmark-nvidia config

# =====================================================================
# Desenvolvimento local (Python venv)
# =====================================================================

.PHONY: venv
venv:  ## Cria virtualenv local em .venv
	$(PYTHON) -m venv .venv
	@echo "Ative com: $(VENV_ACTIVATION_MSG)"

.PHONY: install
install:  ## Instala dependências em .venv (cria se não existe)
	@if [ ! -x "$(VENV_PYTHON)" ]; then $(PYTHON) -m venv .venv; fi
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install -r requirements.txt

.PHONY: install-dev
install-dev: install  ## Instala dependências + dev tools
	"$(VENV_PYTHON)" -m pip install -r requirements-dev.txt

.PHONY: dev
dev:  ## Roda Gradio local (sem Docker)
	"$(VENV_PYTHON)" main.py --gradio --gradio-port $(PORT)

.PHONY: bootstrap
bootstrap:  ## Cria estrutura de diretórios padrão
	"$(VENV_PYTHON)" main.py --bootstrap-dirs

# =====================================================================
# Qualidade de código
# =====================================================================

.PHONY: lint
lint:  ## Roda ruff lint
	"$(VENV_PYTHON)" -m ruff check app/ benchmarks/ tools/ tests/

.PHONY: format
format:  ## Formata com black + isort
	"$(VENV_PYTHON)" -m black app/ benchmarks/ tools/ tests/
	"$(VENV_PYTHON)" -m isort app/ benchmarks/ tools/ tests/

# Runner padronizado. Veja docs/development/quality-and-testing.md.
RUN_TESTS := bash tools/ops/run_tests.sh

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
	"$(VENV_PYTHON)" main.py --deploy

.PHONY: doctor
doctor:  ## Diagnóstico do ambiente (deps, porta, db)
	"$(VENV_PYTHON)" tools/ops/doctor.py

.PHONY: doctor-fix
doctor-fix:  ## Diagnóstico + tenta corrigir automaticamente
	"$(VENV_PYTHON)" tools/ops/doctor.py --fix
