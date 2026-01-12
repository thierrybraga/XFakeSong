@echo off
SETLOCAL EnableDelayedExpansion

TITLE XfakeSong Launcher

:MENU
CLS
ECHO ===============================================================================
ECHO                               XFAKESONG LAUNCHER                               
ECHO ===============================================================================
ECHO.
ECHO Selecione o modo de inicializacao:
ECHO.
ECHO [1] Modo TESTE (Execucao Local com Python)
ECHO [2] Modo PRODUCAO (Docker Container)
ECHO [3] Instalar Dependencias (Local)
ECHO [4] Bootstrap Diretorios
ECHO [0] Sair
ECHO.
SET /P "OPTION=Escolha uma opcao: "

IF "%OPTION%"=="1" GOTO MODE_TEST
IF "%OPTION%"=="2" GOTO MODE_PROD
IF "%OPTION%"=="3" GOTO INSTALL_DEPS
IF "%OPTION%"=="4" GOTO BOOTSTRAP
IF "%OPTION%"=="0" GOTO EOF

ECHO Opcao invalida!
PAUSE
GOTO MENU

:MODE_TEST
CLS
ECHO ===============================================================================
ECHO                            INICIANDO MODO TESTE                                
ECHO ===============================================================================
ECHO.
ECHO Verificando ambiente virtual...
IF NOT EXIST .venv (
    ECHO Ambiente virtual nao encontrado. Criando...
    python -m venv .venv
    CALL .venv\Scripts\activate
    pip install -r requirements.txt
) ELSE (
    CALL .venv\Scripts\activate
)

ECHO Iniciando aplicacao...
python main.py --gradio --gradio-port 7860
PAUSE
GOTO MENU

:MODE_PROD
CLS
ECHO ===============================================================================
ECHO                          INICIANDO MODO PRODUCAO                               
ECHO ===============================================================================
ECHO.
ECHO Verificando Docker...
docker --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Docker nao encontrado! Por favor, instale o Docker Desktop.
    PAUSE
    GOTO MENU
)

ECHO Construindo e iniciando containers...
docker-compose up --build -d

ECHO.
ECHO Aplicacao iniciada em background.
ECHO Acesse: http://localhost:7860
ECHO.
ECHO Para ver logs: docker-compose logs -f
ECHO Para parar: docker-compose down
PAUSE
GOTO MENU

:INSTALL_DEPS
CLS
ECHO Instalando dependencias no ambiente local...
IF NOT EXIST .venv (
    python -m venv .venv
)
CALL .venv\Scripts\activate
pip install -r requirements.txt
ECHO Concluido!
PAUSE
GOTO MENU

:BOOTSTRAP
CLS
ECHO Criando estrutura de diretorios...
python main.py --bootstrap-dirs
ECHO Concluido!
PAUSE
GOTO MENU

:EOF
ENDLOCAL
EXIT /B 0
