$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$codecPython = Join-Path $workspacePath '.venv-compression\Scripts\python.exe'
$renderPython = 'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe'
$pdfPython = 'C:\Users\fn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$sourcePath = Join-Path $PSScriptRoot 'baselines\historical_20261116\source'
$baselinePath = Join-Path $PSScriptRoot 'runs\baseline_single_cpu'
$dictionaryPath = Join-Path $PSScriptRoot 'runs\sh_dictionary'
$referencePath = Join-Path $PSScriptRoot 'references\latest_1607'
$referenceSource = Join-Path $referencePath 'av1_temporal.py'
$stageLog = Join-Path $PSScriptRoot 'stage_latest.log'
function Invoke-Checked {
    param([string]$Executable,[string[]]$Arguments)
    Write-Output ('Running: ' + $Executable + ' ' + ($Arguments -join ' '))
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $stageLog -Append
    if ($LASTEXITCODE -ne 0) { throw ('Command failed: ' + $Executable + ' ' + ($Arguments -join ' ')) }
}
function Measure-Candidate {
    param([string]$Output,[string]$Method,[string]$Candidate,[string]$Options='{}',[int]$Repetitions=3,[string]$Sequence='')
    $argumentsList = @((Join-Path $PSScriptRoot 'run_exploration_suite.py'),'--source',$sourcePath,'--output',$Output,'--methods',$Method,'--repetitions',([string]$Repetitions),'--options',$Options)
    if ($Candidate) { $argumentsList += @('--candidate',$Candidate) }
    if ($Sequence) { $argumentsList += @('--sequence',$Sequence) }
    Invoke-Checked -Executable $codecPython -Arguments $argumentsList
}
function Verify-And-Render {
    param([string]$Suite,[string]$Method,[string]$Order)
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'verify_equivalence.py'),'--baseline',(Join-Path $baselinePath 'rep_01\morton_av1'),'--candidate',(Join-Path $Suite ('rep_01\'+$Method)),'--candidate-order',$Order)
    Invoke-Checked -Executable $renderPython -Arguments @((Join-Path $PSScriptRoot 'evaluate.py'),'--source',$sourcePath,'--results',(Join-Path $Suite 'rep_01'),'--methods',$Method)
}
Push-Location -LiteralPath $workspacePath
try {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'HOLD_NEW_SUITES')) { throw 'Remove the owned suite-boundary hold only after the old stage exits.' }
    $dictionaryState = Get-Content -LiteralPath (Join-Path $dictionaryPath 'suite_state.json') -Raw | ConvertFrom-Json
    if ($dictionaryState.status -ne 'completed') { throw 'Finish active dictionary suite first; do not overlap codec experiments.' }
    if (-not (Test-Path -LiteralPath (Join-Path $referencePath 'encoded'))) {
        Copy-Item -LiteralPath (Join-Path $workspacePath 'benchmark\results_single_thread_v2\morton_av1_opt\encoded') -Destination (Join-Path $referencePath 'encoded') -Recurse
    }
    $rootReferenceManifest = Join-Path $referencePath 'manifest.json'
    $referenceFiles = Get-ChildItem -LiteralPath $referencePath -Recurse -File | Where-Object { $_.FullName -ne $rootReferenceManifest }
    $referenceManifest = foreach ($referenceFile in $referenceFiles) {
        [pscustomobject]@{ path=[System.IO.Path]::GetRelativePath($referencePath,$referenceFile.FullName); bytes=$referenceFile.Length; sha256=(Get-FileHash -LiteralPath $referenceFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    $referenceManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $referencePath 'manifest.json') -Encoding utf8NoBOM
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'test_cached_decode.py'))
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'test_gop.py'))
    Invoke-Checked -Executable $pdfPython -Arguments @((Join-Path $PSScriptRoot 'read_papers.py'),'--render','GSCV++.pdf','--pages','5','6','7','14')
    Invoke-Checked -Executable $pdfPython -Arguments @((Join-Path $PSScriptRoot 'read_papers.py'),'--render','TETRIS-4DGS.pdf','--pages','3','4')
    Invoke-Checked -Executable $pdfPython -Arguments @((Join-Path $PSScriptRoot 'read_papers.py'),'--render','SmoothMotionVectors.pdf','--pages','3','4')
    Invoke-Checked -Executable $pdfPython -Arguments @((Join-Path $PSScriptRoot 'read_papers.py'),'--render','TSOG.pdf','--pages','3')
    $tsogPath = Join-Path $PSScriptRoot 'runs\tsog_current_control'
    Measure-Candidate -Output $tsogPath -Method 'tsog'
    Invoke-Checked -Executable $renderPython -Arguments @((Join-Path $PSScriptRoot 'evaluate.py'),'--source',$sourcePath,'--results',(Join-Path $tsogPath 'rep_01'),'--methods','tsog')
    $latestPath = Join-Path $PSScriptRoot 'runs\latest_temporal_reference'
    Measure-Candidate -Output $latestPath -Method 'temporal' -Candidate $referenceSource
    Verify-And-Render -Suite $latestPath -Method 'temporal' -Order 'source'
    $cachedPath = Join-Path $PSScriptRoot 'runs\latest_cached_decode'
    $encoderOptions = @{ encoder_source=$referenceSource } | ConvertTo-Json -Compress
    Measure-Candidate -Output $cachedPath -Method 'temporal' -Candidate (Join-Path $workspacePath 'adapt\av1_cached_decode.py') -Options $encoderOptions
    Verify-And-Render -Suite $cachedPath -Method 'temporal' -Order 'source'
    $gopPath = Join-Path $PSScriptRoot 'runs\gop_cached_lzma_source_order'
    Measure-Candidate -Output $gopPath -Method 'temporal' -Candidate (Join-Path $workspacePath 'adapt\av1_gop.py')
    Verify-And-Render -Suite $gopPath -Method 'temporal' -Order 'source'
    $screenPath = Join-Path $PSScriptRoot 'screen_sequence.json'
    Measure-Candidate -Output (Join-Path $PSScriptRoot 'runs\noncodebook_sh_color_screen') -Method 'predictive' -Candidate (Join-Path $workspacePath 'adapt\av1_predictive.py') -Options '{"prediction":"sh_color"}' -Repetitions 1 -Sequence $screenPath
    Measure-Candidate -Output (Join-Path $PSScriptRoot 'runs\noncodebook_v1_control_screen') -Method 'morton_av1' -Repetitions 1 -Sequence $screenPath
    $linearPath = Join-Path $PSScriptRoot 'runs\gop_linear_lzma_source_order'
    Measure-Candidate -Output $linearPath -Method 'temporal' -Candidate (Join-Path $workspacePath 'adapt\av1_gop.py') -Options '{"predictor":"linear"}'
    Verify-And-Render -Suite $linearPath -Method 'temporal' -Order 'source'
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'report.py'),'--baseline',$baselinePath,'--tsog-baseline',$tsogPath,'--candidate',($dictionaryPath+';shdict;SH 精确去重消融'),'--candidate',($latestPath+';temporal;16.07MB 最新参考复测'),'--candidate',($cachedPath+';temporal;原 v2 格式缓存解码'),'--candidate',($gopPath+';temporal;GOP v5 保序与静态解码缓存'),'--candidate',($linearPath+';temporal;GOP v5 保序线性预测与缓存'),'--output',(Join-Path $PSScriptRoot 'reports\stage_source_order'))
    Write-Output 'Latest-reference stage finished; inspect report, screening, source preservation and next evidence before ending the goal.'
} finally {
    Pop-Location
}
