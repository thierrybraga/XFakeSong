@echo off
SETLOCAL EnableDelayedExpansion

TITLE XfakeSong Launcher

ECHO Verificando versao do Python...
python --version
ECHO.

SET AUTO_MODE=0
IF "%1"=="test" (
    SET AUTO_MODE=1
    GOTO MODE_TEST
)
IF "%1"=="prod" (
    SET AUTO_MODE=1
    GOTO MODE_PROD
)
IF "%1"=="install" (
    SET AUTO_MODE=1
    GOTO INSTALL_DEPS
)
IF "%1"=="bootstrap" (
    SET AUTO_MODE=1
    GOTO BOOTSTRAP
)

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
ECHO [5] Deploy to Hugging Face
ECHO [0] Sair
ECHO.
SET /P "OPTION=Escolha uma opcao: "

IF "%OPTION%"=="1" GOTO MODE_TEST
IF "%OPTION%"=="2" GOTO MODE_PROD
IF "%OPTION%"=="3" GOTO INSTALL_DEPS
IF "%OPTION%"=="4" GOTO BOOTSTRAP
IF "%OPTION%"=="5" GOTO DEPLOY
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
IF "%AUTO_MODE%"=="0" PAUSE
IF "%AUTO_MODE%"=="1" GOTO EOF
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
    IF "%AUTO_MODE%"=="0" PAUSE
    IF "%AUTO_MODE%"=="1" EXIT /B 1
    GOTO MENU
)

ECHO Construindo e iniciando containers...
docker-compose up --build -d

ECHO.
ECHO Aguardando inicializacao do servico (Health Check)...
:CHECK_HEALTH
timeout /t 5 /nobreak >nul
FOR /F "tokens=*" %%i IN ('docker inspect -f "{{.State.Health.Status}}" xfakesong_app 2^>nul') DO SET HEALTH=%%i

IF "!HEALTH!"=="" (
   ECHO Container nao encontrado ou erro no Docker.
   GOTO PROD_ERROR
)

ECHO Status atual: !HEALTH!

IF "!HEALTH!"=="healthy" GOTO PROD_READY
IF "!HEALTH!"=="unhealthy" GOTO PROD_ERROR
IF "!HEALTH!"=="exited" GOTO PROD_ERROR
IF "!HEALTH!"=="dead" GOTO PROD_ERROR

GOTO CHECK_HEALTH

:PROD_READY
ECHO.
ECHO =========================================================
ECHO           APLICACAO INICIADA COM SUCESSO!
ECHO =========================================================
ECHO.
ECHO Acesse: http://localhost:7860
ECHO.
ECHO Para ver logs: docker-compose logs -f
ECHO Para parar: docker-compose down
IF "%AUTO_MODE%"=="0" PAUSE
IF "%AUTO_MODE%"=="1" GOTO EOF
GOTO MENU

:PROD_ERROR
ECHO.
ECHO =========================================================
ECHO           ERRO AO INICIAR APLICACAO
ECHO =========================================================
ECHO.
ECHO O container reportou status: !HEALTH!
ECHO Verifique os logs para mais detalhes:
ECHO docker-compose logs
ECHO.
IF "%AUTO_MODE%"=="0" PAUSE
IF "%AUTO_MODE%"=="1" EXIT /B 1
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
IF "%AUTO_MODE%"=="0" PAUSE
IF "%AUTO_MODE%"=="1" GOTO EOF
GOTO MENU

:BOOTSTRAP
CLS
ECHO Criando estrutura de diretorios...
python main.py --bootstrap-dirs
ECHO Concluido!
PAUSE
GOTO MENU

:DEPLOY
CLS
ECHO ===============================================================================
ECHO                       DEPLOY PARA HUGGING FACE SPACES
ECHO ===============================================================================
ECHO.
ECHO Verificando dependencias...
IF NOT EXIST .venv (
    ECHO Ambiente virtual nao encontrado. Criando e instalando dependencias...
    python -m venv .venv
    CALL .venv\Scripts\activate
    pip install -r requirements.txt
) ELSE (
    CALL .venv\Scripts\activate
    REM Garantir que huggingface_hub esta instalado
    pip install huggingface_hub >nul 2>&1
)

python main.py --deploy
PAUSE
GOTO MENU

:EOF
ENDLOCAL
EXIT /B 0
