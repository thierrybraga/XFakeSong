# Usar imagem oficial leve do Python
FROM python:3.11-slim

# Definir variáveis de ambiente para evitar arquivos .pyc e logs em buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependências de sistema necessárias
# ffmpeg é crucial para processamento de áudio (librosa)
# curl é necessário para o healthcheck
# gosu para downgrade de privilégios no entrypoint
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    curl \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-privilegiado
RUN useradd -m -u 1000 appuser

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivo de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar script de entrypoint e dar permissão de execução
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copiar o restante do código da aplicação
COPY . .

# Criar diretórios necessários e ajustar permissões
RUN mkdir -p logs app/models app/results data/fake data/real && \
    chown -R appuser:appuser /app

# Expor a porta do Gradio
EXPOSE 7860

# Definir Entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Comando padrão
CMD ["python", "main.py", "--gradio", "--gradio-port", "7860"]
