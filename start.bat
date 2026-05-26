@echo off
REM ====================================================================
REM XFakeSong Launcher (Windows)
REM Uso: start.bat [test|prod|gpu|stop|logs|rebuild|clean|status|install|bootstrap]
REM Sem argumento: abre menu interativo.
REM ====================================================================
chcp 65001 >nul 2>&1
SETLOCAL EnableDelayedExpansion EnableExtensions
TITLE XFakeSong Launcher

REM Cores ANSI (Windows 10+ com cmd.exe / Terminal moderno)
SET "ESC="
SET "C_RESET=%ESC%[0m"
SET "C_GREEN=%ESC%[32m"
SET "C_BLUE=%ESC%[34m"
SET "C_YELLOW=%ESC%[33m"
SET "C_RED=%ESC%[31m"

REM Auto mode quando passa argumento direto (CI/scripts)
SET "AUTO_MODE=0"
IF NOT "%~1"=="" SET "AUTO_MODE=1"

REM Detecta variante do Docker Compose ANTES de qualquer ação
CALL :DETECT_COMPOSE
IF "%DC%"=="" (
    SET "COMPOSE_OK=0"
) ELSE (
    SET "COMPOSE_OK=1"
)

REM Dispatcher de argumentos
IF /I "%~1"=="test"      GOTO MODE_TEST
IF /I "%~1"=="prod"      GOTO MODE_PROD
IF /I "%~1"=="gpu"       GOTO MODE_PROD_GPU
IF /I "%~1"=="stop"      GOTO MODE_STOP
IF /I "%~1"=="logs"      GOTO MODE_LOGS
IF /I "%~1"=="rebuild"   GOTO MODE_REBUILD
IF /I "%~1"=="clean"     GOTO MODE_CLEAN
IF /I "%~1"=="status"    GOTO MODE_STATUS
IF /I "%~1"=="install"   GOTO INSTALL_DEPS
IF /I "%~1"=="bootstrap" GOTO BOOTSTRAP
IF /I "%~1"=="deploy"    GOTO DEPLOY
IF /I "%~1"=="help"      GOTO SHOW_HELP
IF /I "%~1"=="-h"        GOTO SHOW_HELP
IF /I "%~1"=="--help"    GOTO SHOW_HELP

REM ====================================================================
:MENU
CLS
ECHO ===============================================================================
ECHO                              XFAKESONG LAUNCHER
ECHO ===============================================================================
ECHO.
ECHO  [1]  Modo TESTE (Python local)
ECHO  [2]  Modo PRODUCAO (Docker)
ECHO  [3]  Modo PRODUCAO + GPU (NVIDIA)
ECHO  [4]  Stop containers
ECHO  [5]  Ver logs (follow)
ECHO  [6]  Rebuild SEM cache
ECHO  [7]  Status / health
ECHO  [8]  Limpeza profunda (down -v + prune)
ECHO  [9]  Instalar dependencias locais
ECHO  [10] Bootstrap diretorios
ECHO  [11] Deploy Hugging Face Spaces
ECHO  [0]  Sair
ECHO.
SET /P "OPTION=Escolha uma opcao: "

IF "%OPTION%"=="1"  GOTO MODE_TEST
IF "%OPTION%"=="2"  GOTO MODE_PROD
IF "%OPTION%"=="3"  GOTO MODE_PROD_GPU
IF "%OPTION%"=="4"  GOTO MODE_STOP
IF "%OPTION%"=="5"  GOTO MODE_LOGS
IF "%OPTION%"=="6"  GOTO MODE_REBUILD
IF "%OPTION%"=="7"  GOTO MODE_STATUS
IF "%OPTION%"=="8"  GOTO MODE_CLEAN
IF "%OPTION%"=="9"  GOTO INSTALL_DEPS
IF "%OPTION%"=="10" GOTO BOOTSTRAP
IF "%OPTION%"=="11" GOTO DEPLOY
IF "%OPTION%"=="0"  GOTO EOF

ECHO Opcao invalida!
TIMEOUT /T 2 /NOBREAK >nul
GOTO MENU

REM ====================================================================
:MODE_TEST
CLS
ECHO ===============================================================================
ECHO                           MODO TESTE (Python Local)
ECHO ===============================================================================
ECHO.
CALL :CHECK_PYTHON
IF ERRORLEVEL 1 GOTO MENU_RETURN

IF NOT EXIST .venv (
    ECHO Criando ambiente virtual...
    python -m venv .venv
    IF ERRORLEVEL 1 (
        ECHO Falha ao criar venv. Verifique sua instalacao do Python.
        GOTO MENU_RETURN
    )
)
CALL .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO Falha na instalacao das dependencias.
    GOTO MENU_RETURN
)

REM BUG.Startup.3: verifica porta 7860 antes de iniciar
ECHO.
ECHO Verificando porta 7860...
SET "PORT_BUSY=0"
FOR /F "tokens=5" %%P IN ('netstat -ano 2^>nul ^| findstr ":7860 " ^| findstr "LISTENING"') DO (
    SET "PORT_BUSY=1"
    SET "OLD_PID=%%P"
)
IF "%PORT_BUSY%"=="1" (
    ECHO.
    ECHO ===============================================================
    ECHO  Porta 7860 ja em uso pelo PID %OLD_PID%.
    ECHO ===============================================================
    ECHO Encerre a instancia anterior antes de continuar:
    ECHO   taskkill /F /PID %OLD_PID%
    ECHO Ou rode em outra porta:
    ECHO   python main.py --gradio --gradio-port 7861
    ECHO.
    SET /P "KILL=Encerrar PID %OLD_PID% automaticamente? [y/N] "
    IF /I "!KILL!"=="y" (
        taskkill /F /PID %OLD_PID% >nul 2>&1
        IF ERRORLEVEL 1 (
            ECHO Falha ao encerrar. Tente manualmente.
            GOTO MENU_RETURN
        )
        ECHO PID encerrado. Aguardando porta liberar...
        TIMEOUT /T 3 /NOBREAK >nul
    ) ELSE (
        GOTO MENU_RETURN
    )
)

ECHO.
ECHO Aplicacao em http://localhost:7860
ECHO Pressione Ctrl+C para parar.
ECHO.
python main.py --gradio --gradio-port 7860
GOTO MENU_RETURN

REM ====================================================================
:MODE_PROD
SET "GPU=0"
GOTO MODE_PROD_RUN

:MODE_PROD_GPU
SET "GPU=1"
GOTO MODE_PROD_RUN

:MODE_PROD_RUN
CLS
ECHO ===============================================================================
ECHO                         MODO PRODUCAO (Docker)
ECHO ===============================================================================
ECHO.
CALL :CHECK_DOCKER_RUNNING
IF ERRORLEVEL 1 GOTO MENU_RETURN
IF "%COMPOSE_OK%"=="0" (
    ECHO Docker Compose nao encontrado. Atualize seu Docker Desktop.
    GOTO MENU_RETURN
)

ECHO Compose detectado: %DC%
IF "%GPU%"=="1" (
    ECHO GPU mode habilitado ^(NVIDIA^)
    SET "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
) ELSE (
    SET "COMPOSE_FILES=-f docker-compose.yml"
)

ECHO.
ECHO Build + Up...
%DC% %COMPOSE_FILES% up --build -d
IF ERRORLEVEL 1 (
    ECHO Falha no docker compose up. Veja a saida acima.
    GOTO MENU_RETURN
)

CALL :WAIT_HEALTHY
IF ERRORLEVEL 1 (
    ECHO.
    ECHO Container nao ficou healthy a tempo. Logs:
    %DC% logs --tail=50 app
    GOTO MENU_RETURN
)

ECHO.
ECHO ===============================================================
ECHO   Aplicacao rodando em http://localhost:7860
ECHO ===============================================================
ECHO.
ECHO Comandos uteis:
ECHO   Logs:    start.bat logs
ECHO   Stop:    start.bat stop
ECHO   Rebuild: start.bat rebuild
ECHO.
GOTO MENU_RETURN

REM ====================================================================
:MODE_STOP
CLS
ECHO Parando containers...
CALL :CHECK_DOCKER_RUNNING
IF ERRORLEVEL 1 GOTO MENU_RETURN
%DC% down
IF ERRORLEVEL 1 GOTO MENU_RETURN
ECHO Containers parados.
GOTO MENU_RETURN

REM ====================================================================
:MODE_LOGS
CLS
CALL :CHECK_DOCKER_RUNNING
IF ERRORLEVEL 1 GOTO MENU_RETURN
ECHO Pressione Ctrl+C para sair dos logs...
ECHO.
%DC% logs -f --tail=100 app
GOTO MENU_RETURN

REM ====================================================================
:MODE_REBUILD
CLS
ECHO Forcando rebuild SEM cache...
CALL :CHECK_DOCKER_RUNNING
IF ERRORLEVEL 1 GOTO MENU_RETURN

%DC% down
%DC% build --no-cache --pull
IF ERRORLEVEL 1 GOTO MENU_RETURN
%DC% up -d
CALL :WAIT_HEALTHY
GOTO MENU_RETURN

REM ====================================================================
:MODE_CLEAN
CLS
ECHO ===============================================================
ECHO  LIMPEZA: containers, volumes anonimos e imagens dangling
ECHO ===============================================================
ECHO.
SET /P "CONFIRM=Continuar? [y/N] "
IF /I NOT "%CONFIRM%"=="y" GOTO MENU_RETURN

CALL :CHECK_DOCKER_RUNNING
IF ERRORLEVEL 1 GOTO MENU_RETURN

%DC% down -v --remove-orphans
docker image prune -f
docker builder prune -f
ECHO Limpeza concluida.
GOTO MENU_RETURN

REM ====================================================================
:MODE_STATUS
CLS
CALL :CHECK_DOCKER_RUNNING
IF ERRORLEVEL 1 GOTO MENU_RETURN
ECHO --- Compose status ---
%DC% ps
ECHO.
ECHO --- docker stats (snapshot) ---
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>nul
GOTO MENU_RETURN

REM ====================================================================
:INSTALL_DEPS
CLS
ECHO Instalando dependencias locais...
CALL :CHECK_PYTHON
IF ERRORLEVEL 1 GOTO MENU_RETURN
IF NOT EXIST .venv python -m venv .venv
CALL .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
ECHO Concluido.
GOTO MENU_RETURN

REM ====================================================================
:BOOTSTRAP
CLS
ECHO Criando estrutura de diretorios...
CALL :CHECK_PYTHON
IF ERRORLEVEL 1 GOTO MENU_RETURN
python main.py --bootstrap-dirs
ECHO Concluido.
GOTO MENU_RETURN

REM ====================================================================
:DEPLOY
CLS
ECHO ===============================================================
ECHO            DEPLOY PARA HUGGING FACE SPACES
ECHO ===============================================================
CALL :CHECK_PYTHON
IF ERRORLEVEL 1 GOTO MENU_RETURN
IF NOT EXIST .venv (
    python -m venv .venv
    CALL .venv\Scripts\activate.bat
    pip install -r requirements.txt
) ELSE (
    CALL .venv\Scripts\activate.bat
    pip install --quiet huggingface_hub
)
python main.py --deploy
GOTO MENU_RETURN

REM ====================================================================
:SHOW_HELP
ECHO.
ECHO Uso: start.bat [comando]
ECHO.
ECHO Comandos:
ECHO   test       Roda Python local em .venv
ECHO   prod       Sobe Docker em modo producao
ECHO   gpu        Sobe Docker em producao com GPU (NVIDIA)
ECHO   stop       Para containers
ECHO   logs       Tail dos logs do container
ECHO   rebuild    Force rebuild sem cache
ECHO   clean      Limpeza profunda (down -v + prune)
ECHO   status     Status do compose + docker stats
ECHO   install    Instala dependencias em .venv
ECHO   bootstrap  Cria diretorios padrao
ECHO   deploy     Deploy para Hugging Face Spaces
ECHO   help       Mostra esta ajuda
ECHO.
ECHO Sem argumentos: abre menu interativo.
ECHO.
GOTO EOF

REM ====================================================================
REM SUBROTINAS
REM ====================================================================

:CHECK_PYTHON
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO Python nao encontrado no PATH.
    ECHO Instale Python 3.11+ ou use https://www.python.org/downloads/
    EXIT /B 1
)
EXIT /B 0

:CHECK_DOCKER_RUNNING
docker --version >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO Docker nao encontrado. Instale Docker Desktop:
    ECHO   https://www.docker.com/products/docker-desktop/
    EXIT /B 1
)
docker info >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO Docker daemon nao esta rodando.
    ECHO Abra o Docker Desktop e aguarde inicializacao completa.
    EXIT /B 1
)
EXIT /B 0

:DETECT_COMPOSE
REM Prefer Compose v2 (docker compose) sobre v1 (docker-compose)
docker compose version >nul 2>&1
IF NOT ERRORLEVEL 1 (
    SET "DC=docker compose"
    EXIT /B 0
)
docker-compose version >nul 2>&1
IF NOT ERRORLEVEL 1 (
    SET "DC=docker-compose"
    EXIT /B 0
)
SET "DC="
EXIT /B 1

:WAIT_HEALTHY
REM Aguarda container ficar healthy com timeout de 300s
SET "MAX_WAIT=300"
SET "ELAPSED=0"
SET "INTERVAL=5"
ECHO.
ECHO Aguardando container ficar healthy (max %MAX_WAIT%s)...

:WAIT_HEALTHY_LOOP
FOR /F "delims=" %%H IN ('docker inspect -f "{{.State.Health.Status}}" xfakesong_app 2^>nul') DO SET "STATUS=%%H"

IF "%STATUS%"=="" (
    ECHO   Container nao encontrado.
    EXIT /B 1
)

<nul SET /P "=  Status: %STATUS%       (%ELAPSED%s)              "
ECHO.

IF "%STATUS%"=="healthy" (
    ECHO Container healthy.
    EXIT /B 0
)
IF "%STATUS%"=="unhealthy" (
    ECHO Container unhealthy.
    EXIT /B 1
)
IF "%STATUS%"=="exited" (
    ECHO Container exited.
    EXIT /B 1
)

IF %ELAPSED% GEQ %MAX_WAIT% (
    ECHO Timeout aguardando healthy.
    EXIT /B 1
)

TIMEOUT /T %INTERVAL% /NOBREAK >nul
SET /A ELAPSED=ELAPSED+INTERVAL
GOTO WAIT_HEALTHY_LOOP

REM ====================================================================
:MENU_RETURN
IF "%AUTO_MODE%"=="1" GOTO EOF
ECHO.
PAUSE
GOTO MENU

:EOF
ENDLOCAL
EXIT /B 0
