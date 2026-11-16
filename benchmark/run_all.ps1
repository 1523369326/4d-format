param(
    [switch]$SkipTests,
    [string]$Results = 'benchmark\results_single_thread_v2'
)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$codecPython = Join-Path $workspacePath '.venv-compression\Scripts\python.exe'
$renderPython = 'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe'
Push-Location -LiteralPath $workspacePath
try {
    if (-not $SkipTests) {
        & $codecPython benchmark\test_adapters.py
        if ($LASTEXITCODE -ne 0) { throw 'Adapter smoke tests failed' }
        & $codecPython benchmark\test_temporal.py
        if ($LASTEXITCODE -ne 0) { throw 'Temporal codec tests failed' }
    }
    & $codecPython benchmark\prepare.py
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark input preparation failed' }
    $environmentRecord = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'environment.json') -Raw | ConvertFrom-Json
    foreach ($methodName in @('morton_av1_opt','smv','tsog','gsd_v1','gsd_v2','texture_h264','morton_av1')) {
        $timingPath = Join-Path (Join-Path $workspacePath $Results) ($methodName + '\timing.json')
        if (Test-Path -LiteralPath $timingPath) {
            $state = Get-Content -LiteralPath $timingPath -Raw | ConvertFrom-Json
            if ($state.status -eq 'completed') {
                if ($state.dataset_sha256 -ne $environmentRecord.dataset_sha256) {
                    throw ('Input fingerprint changed; use a fresh results directory: ' + $methodName)
                }
                if ($state.cpu_math_threads -ne 1) { throw ('Cached result is not single-thread: ' + $methodName) }
                Write-Output ('Using completed result: ' + $methodName)
                continue
            }
            throw ('Incomplete result exists; inspect it before re-running: ' + $timingPath)
        }
        & $codecPython benchmark\run_codec.py $methodName --threads 1 --results $Results
        if ($LASTEXITCODE -ne 0) { throw ('Codec failed: ' + $methodName) }
    }
    & $renderPython benchmark\render_psnr.py --results $Results
    if ($LASTEXITCODE -ne 0) { throw 'Rendering/PSNR failed' }
    & $codecPython benchmark\summarize_v2.py --results $Results
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
} finally {
    Pop-Location
}
