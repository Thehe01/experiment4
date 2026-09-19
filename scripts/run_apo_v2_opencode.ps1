[CmdletBinding()]
param(
    [ValidateSet('Validate', 'SmokeEstimate', 'Smoke', 'PilotEstimate', 'Pilot', 'TuningEstimate', 'Tuning', 'Estimate', 'Formal')]
    [string]$Mode = 'Estimate',
    [switch]$RefreshPreflight
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$experimentDir = Split-Path -Parent $scriptDir
$probePath = Join-Path $experimentDir 'results\opencode_model_parameter_probe_v3.json'
$semanticPath = Join-Path $experimentDir 'results\opencode_apo_v2_stage1_relation_preflight_v1.json'
$optimizer = Join-Path $scriptDir 'apo_optimizer.py'
$semanticAudit = Join-Path $scriptDir 'audit_apo_v2_relation_capability.py'

# OpenCode Go requires one stable session identity across every request in an
# experiment process.  Reuse an explicitly supplied value, otherwise create a
# run-scoped opaque identifier that is never written with the API key.
if (-not $env:V5_OPENCODE_SESSION_ID) {
    $env:V5_OPENCODE_SESSION_ID = 'bron-v5-apo-v2-' + [guid]::NewGuid().ToString('N')
}
$env:V5_OPENCODE_USER_AGENT = 'bron-apo-research/1.0'

# Frozen task-model contract selected by the parameter probe.
$env:V3_PROVIDER = 'opencode'
$env:V3_OPENCODE_API_MODEL = 'hy3'
$env:V3_API_MODEL = 'hy3'
$env:V3_LLM_TEMPERATURE = '0.0'
$env:V3_LLM_THINKING = 'disabled'
$env:V3_LLM_REASONING_EFFORT = 'none'
$env:V3_LLM_MAX_TOKENS = '4096'
$env:V3_LLM_MAX_ESCALATED_TOKENS = '4096'
$env:V3_LLM_MAX_WORKERS = '8'

# Muse Spark 1.3 is used only by the Critic and Editor through /v1/responses.
$env:V3_APO_OPTIMIZER_MODEL = 'muse-spark-1.3-contributor'
$env:V3_APO_EDITOR_MODEL = 'muse-spark-1.3-contributor'
$env:V3_APO_CRITIC_TEMPERATURE = '0.1'
$env:V3_APO_EDITOR_TEMPERATURE = '0.1'
$env:V3_APO_OPTIMIZER_TOP_P = '1.0'
$env:V3_APO_EDITOR_TOP_P = '1.0'
$env:V3_APO_OPTIMIZER_MAX_TOKENS = '16384'
$env:V3_APO_EDITOR_MAX_TOKENS = '8192'
$env:V3_APO_CANDIDATE_GENERATION_MAX_ATTEMPTS = '4'

# APO-v2 allows only small paired-dev regressions on individual labels and on
# strict/normalized relation micro-F1.  P0 remains the mandatory fallback.
$env:V3_APO_TYPE_F1_MAX_DROP = '0.01'
$env:V3_APO_RELATION_F1_MAX_DROP = '0.005'
$env:V3_APO_NORMALIZED_RELATION_F1_MAX_DROP = '0.005'
$env:V3_APO_TARGET_F1_MIN_GAIN = '0.005'

if ($RefreshPreflight) {
    & python -X utf8 (Join-Path $scriptDir 'probe_opencode_apo_v2_models.py')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & python -X utf8 $semanticAudit `
        --output $semanticPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath $probePath -PathType Leaf)) {
    throw 'Missing OpenCode parameter probe; rerun with -RefreshPreflight.'
}
if (-not (Test-Path -LiteralPath $semanticPath -PathType Leaf)) {
    throw 'Missing Muse semantic preflight; rerun with -RefreshPreflight.'
}

$probe = Get-Content -Raw -LiteralPath $probePath | ConvertFrom-Json
$semantic = Get-Content -Raw -LiteralPath $semanticPath | ConvertFrom-Json
$optimizerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $optimizer).Hash.ToLowerInvariant()
$semanticAuditHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $semanticAudit).Hash.ToLowerInvariant()
if (
    $probe.passed -ne $true -or
    $probe.selected_parameters.task.model -ne 'hy3' -or
    $probe.selected_parameters.task.endpoint -ne '/chat/completions' -or
    $probe.selected_parameters.task.temperature -ne 0.0 -or
    $probe.selected_parameters.task.top_p -ne 0.95 -or
    $probe.selected_parameters.task.thinking -ne 'disabled' -or
    $probe.selected_parameters.task.reasoning_effort -ne 'none' -or
    $probe.selected_parameters.task.max_tokens -ne 4096 -or
    $probe.selected_parameters.critic_editor.model -ne 'muse-spark-1.3-contributor' -or
    $probe.selected_parameters.critic_editor.endpoint -ne '/responses' -or
    $probe.selected_parameters.critic_editor.temperature -ne 0.1 -or
    $probe.selected_parameters.critic_editor.top_p -ne 1.0 -or
    $probe.selected_parameters.critic_editor.reasoning_effort -ne 'high' -or
    $probe.selected_parameters.critic_editor.critic_max_tokens -ne 16384 -or
    $probe.selected_parameters.critic_editor.editor_max_tokens -ne 8192
) {
    throw 'OpenCode parameter gate does not match the registered APO-v2 profile.'
}
if (
    $semantic.passed -ne $true -or
    $semantic.run_kind -ne 'apo_v2_stage1_relation_optimizer_capability_preflight' -or
    $semantic.model -ne 'muse-spark-1.3-contributor' -or
    $semantic.endpoint -ne '/v1/responses' -or
    $semantic.runtime.critic_reasoning_effort -ne 'high' -or
    $semantic.runtime.critic_max_output_tokens -ne 16384 -or
    $semantic.runtime.editor_reasoning_effort -ne 'high' -or
    $semantic.runtime.editor_max_output_tokens -ne 8192 -or
    $semantic.optimizer_script_sha256 -ne $optimizerHash -or
    $semantic.preflight_script_sha256 -ne $semanticAuditHash -or
    $semantic.task_extraction_model_calls -ne 0 -or
    $semantic.dev_gold_loaded -ne $false -or
    $semantic.test_gold_loaded -ne $false
) {
    throw 'Muse Critic/Editor semantic gate failed or used an invalid runtime.'
}

switch ($Mode) {
    'Validate' {
        $arguments = @($optimizer, '--preset', 'apo_v2_cpe_stage1_relation_formal', '--validate-only')
    }
    'Estimate' {
        $arguments = @($optimizer, '--preset', 'apo_v2_cpe_stage1_relation_formal', '--estimate-only')
    }
    'PilotEstimate' {
        $arguments = @($optimizer, '--preset', 'apo_v2_relation_pilot', '--estimate-only')
    }
    'TuningEstimate' {
        $arguments = @($optimizer, '--preset', 'apo_v2_cpe_stage1_relation_tuning_small', '--estimate-only')
    }
    'SmokeEstimate' {
        $arguments = @($optimizer, '--preset', 'apo_v2_relation_smoke', '--estimate-only')
    }
    'Smoke' {
        $arguments = @($optimizer, '--preset', 'apo_v2_relation_smoke')
    }
    'Pilot' {
        $arguments = @($optimizer, '--preset', 'apo_v2_relation_pilot')
    }
    'Tuning' {
        $arguments = @($optimizer, '--preset', 'apo_v2_cpe_stage1_relation_tuning_small')
    }
    'Formal' {
        $arguments = @($optimizer, '--preset', 'apo_v2_cpe_stage1_relation_formal')
    }
}

& python -X utf8 @arguments
exit $LASTEXITCODE
