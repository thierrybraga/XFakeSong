#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_header() {
    clear
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}                              XFAKESONG LAUNCHER                               ${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    echo ""
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 não encontrado!${NC}"
        exit 1
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker não encontrado! Por favor, instale o Docker.${NC}"
        read -p "Pressione Enter para voltar..."
        return 1
    fi
    return 0
}

mode_test() {
    show_header
    echo -e "${GREEN}Iniciando Modo TESTE (Local)...${NC}"
    
    if [ ! -d ".venv" ]; then
        echo "Criando ambiente virtual..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    else
        source .venv/bin/activate
    fi

    echo "Iniciando aplicação..."
    python3 main.py --gradio --gradio-port 7860
    
    read -p "Pressione Enter para voltar..."
}

mode_prod() {
    show_header
    echo -e "${GREEN}Iniciando Modo PRODUÇÃO (Docker)...${NC}"
    
    check_docker || return

    echo "Construindo e subindo containers..."
    docker-compose up --build -d
    
    echo -e "${GREEN}Aplicação iniciada!${NC}"
    echo "Acesse: http://localhost:7860"
    echo ""
    echo "Comandos úteis:"
    echo "  Ver logs: docker-compose logs -f"
    echo "  Parar:    docker-compose down"
    
    read -p "Pressione Enter para voltar..."
}

install_deps() {
    show_header
    echo "Instalando dependências..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -r requirements.txt
    echo -e "${GREEN}Concluído!${NC}"
    read -p "Pressione Enter para voltar..."
}

bootstrap_dirs() {
    show_header
    echo "Criando estrutura de diretórios..."
    python3 main.py --bootstrap-dirs
    echo -e "${GREEN}Concluído!${NC}"
    read -p "Pressione Enter para voltar..."
}

# Loop do menu principal
while true; do
    show_header
    echo "Selecione o modo de inicialização:"
    echo ""
    echo "1) Modo TESTE (Execução Local)"
    echo "2) Modo PRODUÇÃO (Docker)"
    echo "3) Instalar Dependências"
    echo "4) Bootstrap Diretórios"
    echo "0) Sair"
    echo ""
    read -p "Escolha uma opção: " option

    case $option in
        1) mode_test ;;
        2) mode_prod ;;
        3) install_deps ;;
        4) bootstrap_dirs ;;
        0) exit 0 ;;
        *) echo -e "${RED}Opção inválida!${NC}"; sleep 1 ;;
    esac
done
