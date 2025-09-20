# Runbooks

## API Quota Exhausted
- Inspect `cost_summary` from the orchestrator output.
- Reduce domain scope or switch to backup model (Claude) via YAML config.

## Validator Failing Repeatedly
- Increase `max_iterations` in the reflecting section of `configs/agent_config_v3.4.yaml`.
- Review domain validator thresholds in `configs/domains/*`.
