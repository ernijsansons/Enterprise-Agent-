# Enterprise Agent PowerShell Wrapper
# Usage: .\agent.ps1 "your task description"

param(
    [Parameter(Mandatory=$true)]
    [string]$Task
)

# Activate virtual environment
.\.venv\Scripts\Activate

# Run the agent
python agent_cli.py $Task


