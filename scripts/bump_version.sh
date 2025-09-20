#!/bin/bash
set -euo pipefail
poetry version patch
VERSION=$(poetry version -s)
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version $VERSION"
printf 'Version bumped to %s\n' "$VERSION"
