param(
    [string]$DatasetContainer = "xfakesong_academic_dataset",
    [string]$RunName = "academic_20260719_2155"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Dataset = Join-Path $Root "data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz"
$Lock = "$Dataset.test-lock.json"
$RunDir = Join-Path $Root "data/results/$RunName"
$Log = Join-Path $Root "data/logs/$RunName-orchestrator.log"
$Image = "xfakesong/benchmark:nvidia"
$ExpectedModels = @(
    "RandomForest", "SVM", "Hybrid CNN-Transformer",
    "SpectrogramTransformer", "MultiscaleCNN", "Conformer", "RawNet2",
    "AASIST", "RawGAT-ST", "WavLM Original", "HuBERT Original"
)
$NvidiaPip = "/usr/local/lib/python3.11/site-packages/nvidia"
$LdLibraryPath = @(
    "$NvidiaPip/cuda_cupti/lib", "$NvidiaPip/cuda_nvrtc/lib",
    "$NvidiaPip/cusparse/lib", "$NvidiaPip/nccl/lib",
    "$NvidiaPip/cuda_runtime/lib", "$NvidiaPip/cudnn/lib",
    "$NvidiaPip/cufft/lib", "$NvidiaPip/cusolver/lib",
    "$NvidiaPip/cublas/lib", "$NvidiaPip/nvjitlink/lib",
    "$NvidiaPip/curand/lib"
) -join ":"

New-Item -ItemType Directory -Force -Path (Split-Path $Log), $RunDir | Out-Null

function Write-RunLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $Log -Value $line -Encoding utf8
}

function Invoke-DockerPython([string[]]$Arguments, [switch]$Gpu) {
    $dockerArgs = @("run", "--rm", "--user", "0", "--entrypoint", "python")
    if ($Gpu) {
        $dockerArgs += @(
            "--gpus", "all",
            "-e", "LD_LIBRARY_PATH=$LdLibraryPath",
            "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
            "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async"
        )
    }
    $dockerArgs += @(
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "DATABASE_URL=sqlite:////app/data/app.db",
        "-e", "DEEPFAKE_MODELS_DIR=/app/data/models",
        "-v", "${Root}:/app", "-w", "/app", $Image
    )
    $dockerArgs += $Arguments
    & docker @dockerArgs 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Docker/Python falhou com exit code $LASTEXITCODE"
    }
}

try {
    Write-RunLog "Aguardando o container de dataset $DatasetContainer"
    $datasetExit = (& docker wait $DatasetContainer).Trim()
    if ($datasetExit -ne "0") {
        throw "Montagem do dataset falhou com exit code $datasetExit"
    }
    Write-RunLog "Dataset bruto e splits concluídos"

    if (Test-Path -LiteralPath $Dataset) {
        throw "O NPZ acadêmico já existe; recuso sobrescrever: $Dataset"
    }
    Invoke-DockerPython @(
        "scripts/dataset/export_npz_from_splits.py",
        "--out", "/app/data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz",
        "--seed", "42"
    )
    Invoke-DockerPython @(
        "scripts/dataset/freeze_benchmark_test.py",
        "--dataset", "/app/data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz",
        "--declare-untouched"
    )
    Invoke-DockerPython @(
        "scripts/dataset/audit_dataset_leakage.py",
        "--dataset", "/app/data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz",
        "--out", "/app/data/results/$RunName/audits/leakage"
    )
    Invoke-DockerPython @(
        "scripts/dataset/audit_source_shortcut.py",
        "--dataset", "/app/data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz",
        "--out", "/app/data/results/$RunName/audits/source_shortcut"
    )

    Write-RunLog "Iniciando benchmark acadêmico GPU das 11 arquiteturas"
    Invoke-DockerPython -Gpu @(
        "scripts/benchmark/run_models_sequential.py",
        "--dataset", "/app/data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz",
        "--test-lock", "/app/data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz.test-lock.json",
        "--out", "/app/data/results/$RunName",
        "--epochs", "100", "--batch-size", "16",
        "--device-profile", "gpu", "--timeout-min", "480",
        "--latency-runs", "30", "--seed", "42", "--resume"
    )

    $summaryPath = Join-Path $RunDir "run_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "Resumo acadêmico ausente: $summaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $successful = @($summary.models | Where-Object { $_.status -eq "ok" } | ForEach-Object { $_.model })
    $missing = @($ExpectedModels | Where-Object { $_ -notin $successful })
    if ($summary.status -ne "ok" -or $missing.Count -gt 0) {
        throw "Benchmark incompleto; modelos ausentes/falhos: $($missing -join ', ')"
    }
    $resultFiles = @(Get-ChildItem -LiteralPath $RunDir -Filter results.json -File -Recurse)
    $modelFiles = @(Get-ChildItem -LiteralPath $RunDir -File -Recurse |
        Where-Object { $_.Extension -in @(".keras", ".h5", ".pkl", ".pt") })
    $figureFiles = @(Get-ChildItem -LiteralPath $RunDir -Filter *.png -File -Recurse)
    if ($resultFiles.Count -lt 11 -or $modelFiles.Count -lt 11 -or $figureFiles.Count -lt 33) {
        throw "Validação de artefatos falhou: results=$($resultFiles.Count), models=$($modelFiles.Count), figures=$($figureFiles.Count)"
    }

    Write-RunLog "Validação aprovada; iniciando remoção dos artefatos legados"
    $modelsRoot = (Resolve-Path (Join-Path $Root "data/models")).Path
    $resultsRoot = (Resolve-Path (Join-Path $Root "data/results")).Path
    Get-ChildItem -LiteralPath $modelsRoot -Force | Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $resultsRoot -Force |
        Where-Object { $_.FullName -ne $RunDir } |
        Remove-Item -Recurse -Force
    foreach ($legacy in @("figures", "results", "app/results", "data/figures")) {
        $candidate = Join-Path $Root $legacy
        if (Test-Path -LiteralPath $candidate) {
            Remove-Item -LiteralPath $candidate -Recurse -Force
        }
    }
    Write-RunLog "Pipeline acadêmico concluído e legados removidos"
}
catch {
    Write-RunLog "FALHA: $($_.Exception.Message)"
    exit 1
}
