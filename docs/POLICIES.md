# Policies

## Data Handling
- PII scrubbing: Auto-redact emails, names in logs.
- Logging Levels: INFO default; DEBUG for dev.
- Retention: 30 days default (configurable in YAML).

## Export Process
- Use `make export` to dump anonymized logs.
