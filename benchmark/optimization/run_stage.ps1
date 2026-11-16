param([switch]$WaitForBaseline)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$codecPython = Join-Path $workspacePath '.venv-compression\Scripts\python.exe'
$renderPython = 'E:\ProgramData\Anaconda\envs\cu121-3dgs\python.exe'
$pdfPython = 'C:\Users\fn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$sourcePath = Join-Path $PSScriptRoot 'baselines\historical_20261116\source'
$baselinePath = Join-Path $PSScriptRoot 'runs\baseline_single_cpu'
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Command failed: ' + $Executable + ' ' + ($Arguments -join ' ')) }
}
Push-Location -LiteralPath $workspacePath
try {
    if ($WaitForBaseline) {
        Write-Output 'Waiting for the serial baseline suite; no extra computation while it is timing.'
        while ($true) {
            $baselineState = Get-Content -LiteralPath (Join-Path $baselinePath 'suite_state.json') -Raw | ConvertFrom-Json
            if ($baselineState.status -eq 'completed') { break }
            if ($baselineState.status -eq 'failed') { throw 'Baseline suite failed; inspect its results before resuming.' }
            Start-Sleep -Seconds 10
        }
    }
    $baselineState = Get-Content -LiteralPath (Join-Path $baselinePath 'suite_state.json') -Raw | ConvertFrom-Json
    if ($baselineState.status -ne 'completed') { throw 'Complete the baseline suite first.' }
    foreach ($testName in @('test_shdict.py','test_gop.py','test_predictive.py')) {
        Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot $testName))
    }
    Invoke-Checked -Executable $pdfPython -Arguments @((Join-Path $PSScriptRoot 'read_papers.py'),'--render','GSCV++.pdf','--pages','1','3','4')
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'screen_sequence.json'))) {
        Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'make_screen_sequence.py'))
    }
    Invoke-Checked -Executable $renderPython -Arguments @((Join-Path $PSScriptRoot 'evaluate.py'),'--source',$sourcePath,'--results',(Join-Path $baselinePath 'rep_01'),'--methods','morton_av1','tsog')
    $dictionaryPath = Join-Path $PSScriptRoot 'runs\sh_dictionary'
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'run_exploration_suite.py'),'--source',$sourcePath,'--output',$dictionaryPath,'--methods','shdict','--candidate',(Join-Path $workspacePath 'adapt\av1_shdict.py'),'--repetitions','3')
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'verify_equivalence.py'),'--baseline',(Join-Path $baselinePath 'rep_01\morton_av1'),'--candidate',(Join-Path $dictionaryPath 'rep_01\shdict'))
    Invoke-Checked -Executable $renderPython -Arguments @((Join-Path $PSScriptRoot 'evaluate.py'),'--source',$sourcePath,'--results',(Join-Path $dictionaryPath 'rep_01'),'--methods','shdict')
    $temporalPath = Join-Path $PSScriptRoot 'runs\temporal_existing'
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'run_exploration_suite.py'),'--source',$sourcePath,'--output',$temporalPath,'--methods','temporal','--repetitions','3')
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'verify_equivalence.py'),'--baseline',(Join-Path $baselinePath 'rep_01\morton_av1'),'--candidate',(Join-Path $temporalPath 'rep_01\temporal'),'--candidate-order','source')
    Invoke-Checked -Executable $renderPython -Arguments @((Join-Path $PSScriptRoot 'evaluate.py'),'--source',$sourcePath,'--results',(Join-Path $temporalPath 'rep_01'),'--methods','temporal')
    $gopPath = Join-Path $PSScriptRoot 'runs\gop_cached'
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'run_exploration_suite.py'),'--source',$sourcePath,'--output',$gopPath,'--methods','temporal','--candidate',(Join-Path $workspacePath 'adapt\av1_gop.py'),'--repetitions','3')
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'verify_equivalence.py'),'--baseline',(Join-Path $baselinePath 'rep_01\morton_av1'),'--candidate',(Join-Path $gopPath 'rep_01\temporal'),'--candidate-order','gop')
    Invoke-Checked -Executable $renderPython -Arguments @((Join-Path $PSScriptRoot 'evaluate.py'),'--source',$sourcePath,'--results',(Join-Path $gopPath 'rep_01'),'--methods','temporal')
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'run_exploration_suite.py'),'--source',$sourcePath,'--output',(Join-Path $PSScriptRoot 'runs\noncodebook_sh_color_screen'),'--methods','predictive','--candidate',(Join-Path $workspacePath 'adapt\av1_predictive.py'),'--sequence',(Join-Path $PSScriptRoot 'screen_sequence.json'),'--options','{"prediction":"sh_color"}','--repetitions','1')
    Invoke-Checked -Executable $codecPython -Arguments @((Join-Path $PSScriptRoot 'report.py'),'--baseline',$baselinePath,'--candidate',($dictionaryPath+';shdict;SH 精确去重消融'),'--candidate',($temporalPath+';temporal;已有 I/P 时序候选'),'--candidate',($gopPath+';temporal;I/P 与静态解码缓存'),'--output',(Join-Path $PSScriptRoot 'reports\stage_01'))
    Write-Output 'Stage measurements and reports completed. Review evidence, literature and next hypotheses; this is not the end of the expanded goal.'
} finally {
    Pop-Location
}
