[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [switch]$SkipStaticTests
)

$ErrorActionPreference = "Stop"

$v6Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $v6Root "..\..")).Path
$pilotSplit = Join-Path $v6Root "data\protegi_prompt_scope_pilot_v1.json"
$officialSplit = Join-Path $v6Root "data\train_dev_test_split_v7.json"
$constrainedConfig = Join-Path $v6Root "protegi\configs\protegi_pilot_constrained.yaml"
$unconstrainedConfig = Join-Path $v6Root "protegi\configs\protegi_pilot_unconstrained.yaml"
$runner = Join-Path $v6Root "scripts\run_protegi.py"
$comparer = Join-Path $v6Root "scripts\compare_protegi_scope_pilot.py"
$protegiTestFile = Join-Path $v6Root "tests\test_protegi.py"
$packageTestFile = Join-Path $v6Root "tests\test_v6_package.py"
$pilotRoot = Join-Path $v6Root "results\protegi_optimization\pilot_scope_v1"
$constrainedOutput = Join-Path $pilotRoot "entity_constrained"
$unconstrainedOutput = Join-Path $pilotRoot "entity_unconstrained"
$comparisonOutput = Join-Path $pilotRoot "comparison"

foreach ($requiredPath in @(
    $pilotSplit,
    $officialSplit,
    $constrainedConfig,
    $unconstrainedConfig,
    $runner,
    $comparer,
    $protegiTestFile,
    $packageTestFile
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "缺少 pilot 必需文件: $requiredPath"
    }
}

$pilot = Get-Content -Raw -LiteralPath $pilotSplit | ConvertFrom-Json
$official = Get-Content -Raw -LiteralPath $officialSplit | ConvertFrom-Json
$officialHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $officialSplit).Hash.ToLowerInvariant()
$expectedOfficialHash = ([string]$pilot.parent_split_sha256).ToLowerInvariant()
if ($officialHash -ne $expectedOfficialHash) {
    throw "官方 v7 划分哈希与 pilot manifest 不一致，拒绝运行"
}
if (@($pilot.test).Count -ne 0) {
    throw "pilot manifest 的 test 必须为空"
}
foreach ($splitName in @("train", "dev")) {
    $windowCount = [int]$pilot.audit_counts.$splitName.runtime_windows
    if ($windowCount -le 0) {
        throw "pilot $splitName 窗口数必须为正: $windowCount"
    }
}
foreach ($configPath in @($constrainedConfig, $unconstrainedConfig)) {
    if ((Select-String -LiteralPath $configPath -Pattern '^task_max_workers:\s*8\s*$').Count -ne 1) {
        throw "pilot 配置未固定 task_max_workers=8: $configPath"
    }
    if ((Select-String -LiteralPath $configPath -Pattern '^eval_batch_size:\s*8\s*$').Count -ne 1) {
        throw "pilot 配置未固定 eval_batch_size=8: $configPath"
    }
}

$officialTrain = @{}
foreach ($docId in $official.train) { $officialTrain[[string]$docId] = $true }
$officialDev = @{}
foreach ($docId in $official.dev) { $officialDev[[string]$docId] = $true }
foreach ($docId in $pilot.train) {
    if (-not $officialTrain.ContainsKey([string]$docId)) {
        throw "pilot train 文档不属于官方 train: $docId"
    }
}
foreach ($docId in $pilot.dev) {
    if (-not $officialDev.ContainsKey([string]$docId)) {
        throw "pilot dev 文档不属于官方 dev: $docId"
    }
}

function Get-ComparableConfigLines([string]$Path) {
    return @(
        Get-Content -LiteralPath $Path |
            Where-Object {
                $_ -notmatch '^\s*#' -and
                $_ -notmatch '^\s*$' -and
                $_ -notmatch '^prompt_scope:'
            }
    )
}

$configDiff = Compare-Object (Get-ComparableConfigLines $constrainedConfig) (Get-ComparableConfigLines $unconstrainedConfig)
if ($configDiff) {
    throw "两个 pilot 配置除 prompt_scope 外并不相同，拒绝运行"
}

foreach ($newTarget in @($constrainedOutput, $unconstrainedOutput, $comparisonOutput)) {
    if (Test-Path -LiteralPath $newTarget) {
        throw "目标已存在，拒绝覆盖或续跑: $newTarget"
    }
}

Push-Location $repoRoot
try {
    if (-not $SkipStaticTests) {
        & $PythonCommand -X utf8 $packageTestFile
        if ($LASTEXITCODE -ne 0) {
            throw "v6 包边界测试失败，未启动任何模型实验"
        }
        & $PythonCommand -X utf8 $protegiTestFile
        if ($LASTEXITCODE -ne 0) {
            throw "ProTeGi 静态/单元测试失败，未启动任何模型实验"
        }
    }

    $constrainedArgs = @(
        "-X", "utf8", $runner,
        "--stage", "entity",
        "--method", "protegi",
        "--config", $constrainedConfig,
        "--split-file", $pilotSplit,
        "--allow-custom-split",
        "--output-dir", $constrainedOutput
    )
    & $PythonCommand @constrainedArgs
    if ($LASTEXITCODE -ne 0) {
        throw "constrained pilot 失败；unconstrained pilot 未启动"
    }

    $unconstrainedArgs = @(
        "-X", "utf8", $runner,
        "--stage", "entity",
        "--method", "protegi",
        "--config", $unconstrainedConfig,
        "--split-file", $pilotSplit,
        "--allow-custom-split",
        "--output-dir", $unconstrainedOutput
    )
    & $PythonCommand @unconstrainedArgs
    if ($LASTEXITCODE -ne 0) {
        throw "unconstrained pilot 失败；未生成比较报告"
    }

    $compareArgs = @(
        "-X", "utf8", $comparer,
        "--constrained-dir", $constrainedOutput,
        "--unconstrained-dir", $unconstrainedOutput,
        "--output-dir", $comparisonOutput
    )
    & $PythonCommand @compareArgs
    if ($LASTEXITCODE -ne 0) {
        throw "两臂运行完成，但比较报告生成失败"
    }
}
finally {
    Pop-Location
}

Write-Host "ProTeGi prompt-scope pilot 完成。比较报告: $comparisonOutput"
