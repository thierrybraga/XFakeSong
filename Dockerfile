# Usar imagem oficial leve do Python
FROM python:3.11-slim

# Definir variáveis de ambiente para evitar arquivos .pyc e logs em buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependências de sistema necessárias
# ffmpeg é crucial para processamento de áudio (librosa)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivo de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código da aplicação
COPY . .

# Criar diretórios necessários se não existirem (baseado no bootstrap)
RUN mkdir -p logs app/models app/results data/fake data/real

# Expor a porta do Gradio
EXPOSE 7860

# Comando padrão para iniciar a aplicação
# Usa o modo Gradio por padrão
CMD ["python", "main.py", "--gradio", "--gradio-port", "7860", "--listen-all"]
