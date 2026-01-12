#!/bin/bash
set -e

# Criar diretórios necessários se não existirem
mkdir -p /app/logs /app/app/models /app/app/results /app/data/fake /app/data/real

# Se estivermos rodando como root, ajustar permissões (útil se volumes forem montados)
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser /app/logs /app/app/models /app/app/results /app/data
    exec gosu appuser "$@"
fi

exec "$@"
