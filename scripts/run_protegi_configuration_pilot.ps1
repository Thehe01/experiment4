[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [switch]$SkipStaticTests,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$v6Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $v6Root "..\..")).Path
$manifest = Join-Path $v6Root "data\protegi_configuration_pilot_v1.json"
$officialSplit = Join-Path $v6Root "data\train_dev_test_split_v7.json"
$config = Join-Path $v6Root "protegi\configs\protegi_configuration_pilot_v1.yaml"
$protocol = Join-Path $v6Root "protegi\CONFIGURATION_OPTIMIZATION_PROTOCOL.md"
$preflight = Join-Path $v6Root "scripts\check_protegi_configuration_pilot.py"
$runner = Join-Path $v6Root "scripts\run_protegi.py"
$protegiTests = Join-Path $v6Root "tests\test_protegi.py"
$packageTests = Join-Path $v6Root "tests\test_v6_package.py"
$output = Join-Path $v6Root "results\protegi_optimization\configuration_pilot_v1\entity_constrained"

foreach ($requiredPath in @(
    $manifest,
    $officialSplit,
    $config,
    $protocol,
    $preflight,
    $runner,
    $protegiTests,
    $packageTests
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "缺少 Configuration pilot 必需文件: $requiredPath"
    }
}

Push-Location $repoRoot
try {
    if (-not $SkipStaticTests) {
        & $PythonCommand -B -X utf8 $packageTests
        if ($LASTEXITCODE -ne 0) {
            throw "v6 包边界测试失败，未启动任何模型实验"
        }
        & $PythonCommand -B -X utf8 $protegiTests
        if ($LASTEXITCODE -ne 0) {
            throw "ProTeGi 单元测试失败，未启动任何模型实验"
        }
    }

    $preflightArgs = @(
        "-B", "-X", "utf8", $preflight,
        "--manifest", $manifest,
        "--config", $config,
        "--official-split", $officialSplit,
        "--output-dir", $output
    )
    & $PythonCommand @preflightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Configuration pilot 离线预检失败，未启动任何模型实验"
    }

    if ($PreflightOnly) {
        Write-Host "Configuration pilot 离线预检完成；未调用模型。"
        return
    }

    $runArgs = @(
        "-B", "-X", "utf8", $runner,
        "--stage", "entity",
        "--method", "protegi",
        "--config", $config,
        "--split-file", $manifest,
        "--output-dir", $output
    )
    & $PythonCommand @runArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Configuration-targeted constrained ProTeGi pilot 失败"
    }
}
finally {
    Pop-Location
}

Write-Host "Configuration pilot 完成。结果目录: $output"
