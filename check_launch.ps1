# Navigate to project root
Set-Location "C:\Users\ernij\OneDrive\Documents\Enterprise Agent"

# Ensure required environment context
$envFile = Join-Path (Get-Location) ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $parts = $trimmed.Split('=', 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ([string]::IsNullOrWhiteSpace($value)) { continue }
        switch ($key) {
            'SNYK_TOKEN' { if (-not $env:SNYK_TOKEN) { $env:SNYK_TOKEN = $value } }
            'OPENAI_API_KEY' { if (-not $env:OPENAI_API_KEY) { $env:OPENAI_API_KEY = $value } }
        }
    }
}
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = 'STUBBED_FALLBACK'
}

Write-Host "=== Python Version ==="
python --version

Write-Host "`n=== Pytest (unit + integration) ==="
python -m pytest tests/unit tests/integration

Write-Host "`n=== Black --check ==="
black --check src configs tests

Write-Host "`n=== Flake8 ==="
flake8 src

Write-Host "`n=== Isort --check-only ==="
isort --check-only src configs tests

Write-Host "`n=== Snyk Status (token stub allowed) ==="
if (Get-Command snyk -ErrorAction SilentlyContinue) {
    if ($env:SNYK_TOKEN) {
        $authSucceeded = $true
        try {
            snyk auth $env:SNYK_TOKEN | Out-Null
        } catch {
            $authSucceeded = $false
            Write-Host "Snyk auth failed: $($_.Exception.Message)"
        }
        if ($authSucceeded) {
            snyk test --severity-threshold=high
        } else {
            Write-Host "Skipping Snyk test due to authentication failure."
        }
    } else {
        Write-Host "Snyk token not configured; skipping test."
    }
} else {
    Write-Host "Snyk CLI not installed"
}

Write-Host "`n=== Sample Orchestrator Run ==="
$script = @"
from src.agent_orchestrator import AgentOrchestrator

orch = AgentOrchestrator()
result = orch.run_mode('coding', 'build_from_docs:: Hello World Check')
plan = result.get('plan')
if isinstance(plan, dict):
    plan_text = plan.get('text') or ''
elif plan is None:
    plan_text = ''
else:
    plan_text = str(plan)
fields = {
    'confidence': result.get('confidence'),
    'needs_reflect': result.get('needs_reflect'),
    'governance_blocked': result.get('governance_blocked', False),
    'plan_first_line': plan_text.split('\n')[0] if plan_text else '',
    'code_source': result.get('code_source'),
}
for key, value in fields.items():
    print(f"{key}: {value}")
"@
Set-Content tmp_launch_check.py $script
python tmp_launch_check.py
Remove-Item tmp_launch_check.py

Write-Host "`n=== Done ==="
