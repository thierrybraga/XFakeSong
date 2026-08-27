<#
.SYNOPSIS
    Prepara WSL2 + Docker Engine + GPU NVIDIA para rodar o benchmark do XFakeSong.

.DESCRIPTION
    Instala a distro Ubuntu no WSL2 e, DENTRO dela, o Docker Engine e o
    NVIDIA Container Toolkit. Nao usa Docker Desktop: o Engine dentro da distro
    resolve o mesmo caso de uso do projeto (build + run com --gpus all) sem
    servico no host nem licenca comercial.

    O acesso a GPU nao precisa de driver dentro do Linux: o WSL2 expoe o driver
    do Windows em /usr/lib/wsl/lib. Instalar driver NVIDIA dentro da distro
    QUEBRA esse mecanismo -- o script nao faz isso e voce tambem nao deve.

    Cada etapa e idempotente: reexecutar depois de uma falha retoma sem estragar
    o que ja funcionou.

.NOTES
    Execute em PowerShell COMO ADMINISTRADOR.
    Tempo: ~15-30 min (downloads) + ~20-40 min do build da imagem.
    Espaco: ~25 GB (distro + camadas Docker + imagem).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ops\setup_wsl2_docker_gpu.ps1
#>

[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu-24.04',
    [switch]$SkipDistroInstall,
    [switch]$BuildImage
)

$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Text) Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "  [ok] $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "  [!]  $Text" -ForegroundColor Yellow }

# --- 0. Elevacao -------------------------------------------------------------
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Execute este script em um PowerShell COMO ADMINISTRADOR.'
}
Write-Ok 'Elevado'

# --- 1. Pre-requisitos do host ----------------------------------------------
Write-Step 'Pre-requisitos do host'

$cs = Get-CimInstance Win32_ComputerSystem
if (-not $cs.HypervisorPresent) {
    throw ('Hipervisor ausente. Habilite SVM/AMD-V (ou VT-x) na BIOS e as ' +
           'features VirtualMachinePlatform e Microsoft-Windows-Subsystem-Linux.')
}
Write-Ok "Hipervisor presente | RAM $([math]::Round($cs.TotalPhysicalMemory/1GB,1)) GB"

try {
    $smi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Ok "GPU: $smi"
} catch {
    throw 'nvidia-smi nao respondeu. Instale/atualize o driver NVIDIA no WINDOWS antes.'
}

foreach ($feature in 'VirtualMachinePlatform', 'Microsoft-Windows-Subsystem-Linux') {
    $state = (Get-WindowsOptionalFeature -Online -FeatureName $feature).State
    if ($state -ne 'Enabled') {
        Write-Warn "$feature esta '$state' -- habilitando (pode exigir reboot)"
        Enable-WindowsOptionalFeature -Online -FeatureName $feature -NoRestart | Out-Null
    } else {
        Write-Ok "$feature habilitado"
    }
}

# --- 2. Distro WSL2 ----------------------------------------------------------
Write-Step "Distro WSL2 ($Distro)"

& wsl --update 2>&1 | Out-Null
$installed = (& wsl -l -q) -join "`n" -replace "`0", ''

if ($installed -match [regex]::Escape($Distro)) {
    Write-Ok "$Distro ja instalada"
} elseif ($SkipDistroInstall) {
    throw "Distro ausente e -SkipDistroInstall foi passado."
} else {
    Write-Host "  instalando $Distro (download da Microsoft Store)..."
    # --no-launch evita o setup interativo de usuario; usamos root, que e o que
    # o Docker Engine precisa de qualquer forma.
    & wsl --install -d $Distro --no-launch
    if ($LASTEXITCODE -ne 0) { throw "wsl --install falhou ($LASTEXITCODE)" }
    Write-Ok "$Distro instalada"
}

& wsl --set-default $Distro 2>&1 | Out-Null
& wsl --set-version $Distro 2 2>&1 | Out-Null

# --- 3. Docker Engine + NVIDIA Container Toolkit dentro da distro ------------
Write-Step 'Docker Engine + NVIDIA Container Toolkit (dentro da distro)'

# Repositorios OFICIAIS com chave GPG verificada -- nao usamos `curl | sh`.
$provision = @'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg lsb-release >/dev/null

# ---- Docker Engine (repositorio oficial docker.com) ----
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin >/dev/null
fi
echo "docker: $(docker --version)"

# ---- NVIDIA Container Toolkit (repositorio oficial nvidia.github.io) ----
# NAO instalar driver NVIDIA aqui: o WSL2 monta o driver do Windows em
# /usr/lib/wsl/lib e um driver nativo quebraria esse caminho.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit >/dev/null
fi
nvidia-ctk runtime configure --runtime=docker >/dev/null
echo "nvidia-ctk: $(nvidia-ctk --version | head -1)"

# ---- systemd/daemon ----
# A distro roda com systemd (default no Ubuntu 24.04 sob WSL >= 0.67).
if pidof systemd >/dev/null 2>&1; then
  systemctl enable --now docker >/dev/null 2>&1 || true
else
  grep -q 'systemd=true' /etc/wsl.conf 2>/dev/null || printf '[boot]\nsystemd=true\n' >> /etc/wsl.conf
  service docker start >/dev/null 2>&1 || true
fi
docker info --format '  storage-driver={{.Driver}} runtime-default={{.DefaultRuntime}}' 2>/dev/null || true
'@

$provision = $provision -replace "`r`n", "`n"
$tmp = Join-Path $env:TEMP 'xfakesong_provision.sh'
[IO.File]::WriteAllText($tmp, $provision, (New-Object Text.UTF8Encoding $false))
$wslTmp = & wsl -d $Distro -u root -- wslpath -a ($tmp -replace '\\', '/')
& wsl -d $Distro -u root -- bash $wslTmp
if ($LASTEXITCODE -ne 0) { throw "provisionamento dentro da distro falhou ($LASTEXITCODE)" }
Write-Ok 'Docker Engine e NVIDIA Container Toolkit instalados'

# systemd so entra em vigor apos reiniciar a distro
Write-Host '  reiniciando a distro para aplicar systemd/daemon...'
& wsl --terminate $Distro | Out-Null
Start-Sleep -Seconds 3
& wsl -d $Distro -u root -- bash -lc 'pidof systemd >/dev/null && systemctl start docker || service docker start' 2>&1 | Out-Null

# --- 4. Verificacao da GPU dentro do container -------------------------------
Write-Step 'Verificacao: GPU visivel dentro de um container'

$check = & wsl -d $Distro -u root -- bash -lc `
    'docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1'
if ($LASTEXITCODE -ne 0) {
    Write-Warn "container GPU falhou:`n$check"
    throw 'GPU nao visivel no container. Confira driver do Windows e nvidia-ctk.'
}
Write-Ok "GPU no container: $check"

# --- 5. Build da imagem do benchmark ----------------------------------------
if ($BuildImage) {
    Write-Step 'Build da imagem xfakesong/benchmark:nvidia'
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $repoWsl = & wsl -d $Distro -u root -- wslpath -a ($repo -replace '\\', '/')
    & wsl -d $Distro -u root -- bash -lc `
        "cd '$repoWsl' && docker compose -f docker/compose/benchmark.nvidia.yml build"
    if ($LASTEXITCODE -ne 0) { throw "build falhou ($LASTEXITCODE)" }
    Write-Ok 'Imagem construida'
}

Write-Step 'Pronto'
Write-Host @"
  Ambiente pronto. Proximos passos (dentro da distro $Distro):

    wsl -d $Distro -u root
    cd <repo>
    docker compose -f docker/compose/benchmark.nvidia.yml build      # se ainda nao fez
    docker compose -f docker/compose/benchmark.nvidia.yml up -d
    docker logs -f xfakesong_benchmark_nvidia
"@ -ForegroundColor Green
