[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$EstimateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($ValidateOnly -and $EstimateOnly) {
    throw 'ValidateOnly and EstimateOnly are mutually exclusive.'
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$experimentDir = Split-Path -Parent $scriptDir
$keyCandidates = @(
    (Join-Path $experimentDir 'commandcode\_key'),
    (Join-Path $experimentDir 'commandcode_key')
)
$keyPath = $keyCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1

if (-not $keyPath) {
    throw (
        'Command Code API key not found. Expected commandcode\_key or ' +
        'commandcode_key under experiments\v5.'
    )
}

# Reuse the existing OpenAI-compatible relay profile without modifying any
# file bound by the frozen Rule/Multipass/Full manifest. The public run record
# still captures the exact Command Code base URL and model identifiers.
$env:V5_PROVIDER = 'relay'
$env:V5_RELAY_BASE_URL = 'https://api.commandcode.ai/provider/v1'
$env:V5_RELAY_API_KEY_FILE = [System.IO.Path]::GetFullPath($keyPath)
$env:V5_RELAY_API_MODEL = 'deepseek/deepseek-v4-flash-fast'
$env:V5_APO_OPTIMIZER_MODEL = 'meta/muse-spark-1.2-contributor'
$env:V5_APO_EDITOR_MODEL = 'meta/muse-spark-1.2-contributor'

# Task-model settings selected for the Configuration recall experiment.
# Omit the provider-specific `thinking` object and use Command Code's supported
# low reasoning effort explicitly.
$env:V5_LLM_THINKING = ' '
$env:V5_LLM_REASONING_EFFORT = 'low'
$env:V5_LLM_TEMPERATURE = '0.1'
$env:V5_LLM_MAX_TOKENS = '2048'
# Keep the normal response budget at 2048. Only retry with 4096 when the
# provider reports that the first response exhausted its output budget.
$env:V5_LLM_MAX_ESCALATED_TOKENS = '4096'
$env:V5_LLM_MAX_WORKERS = '32'

# Critic/Editor remain separate from the task model.
$env:V5_APO_OPTIMIZER_THINKING = ' '
$env:V5_APO_EDITOR_THINKING = ' '
$env:V5_APO_CRITIC_FALLBACK_THINKING = ' '

$optimizer = Join-Path $scriptDir 'apo_optimizer.py'
$arguments = @(
    '-X',
    'utf8',
    $optimizer,
    '--preset',
    'configuration_recall_only_p0'
)
if ($ValidateOnly) {
    $arguments += '--validate-only'
}
if ($EstimateOnly) {
    $arguments += '--estimate-only'
}

& python @arguments
exit $LASTEXITCODE
