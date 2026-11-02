param(
    [int]$Epochs = 100,
    [string]$Dataset = "data/datasets/benchmark_dataset_15k.npz",
    [string]$Out = "data/results/tcc_article_reproduction_2026-10-17"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

if (-not (Test-Path $Dataset)) {
    throw "Dataset do artigo não encontrado: $Dataset. Reconstrua-o conforme docs/evaluation/canonical-study-and-legacy-inventory.md."
}
$DatasetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Dataset).Hash.ToLowerInvariant()
$ExpectedHash = "3775b35fb05155011b21a30ff60632aaa7c9f67f031bd6f7874a5f313363400f"
if ($DatasetHash -ne $ExpectedHash) {
    throw "Hash de dataset divergente: esperado $ExpectedHash; encontrado $DatasetHash."
}
if (-not (Test-Path "$Dataset.test-lock.json")) {
    throw "Test lock do artigo não encontrado: $Dataset.test-lock.json"
}

$env:DOCKER_TRAIN_CPU_LIMIT = if ($env:DOCKER_TRAIN_CPU_LIMIT) { $env:DOCKER_TRAIN_CPU_LIMIT } else { "8" }

Write-Host "== Build/validate GPU image =="
python tools/ops/docker_build.py benchmark-nvidia build

Write-Host "== Reprodução do protocolo do artigo: 11 entradas oficiais =="
docker compose -f infrastructure/docker/compose/benchmark.nvidia.yml --env-file .env run --rm benchmark `
    python benchmarks/run_models_sequential.py `
        --dataset $Dataset `
        --test-lock "$Dataset.test-lock.json" `
        --out $Out `
        --epochs $Epochs `
        --batch-size 32 `
        --scope official `
        --seed 42 `
        --device-profile gpu `
        --latency-runs 30 `
        --snr 30 20 10 5 `
        --train-aug-snr 30 20 10 `
        --train-noise-copies 1 `
        --band-correction-hz 7500 `
        --checkpoint-monitor val_eer `
        --resume
