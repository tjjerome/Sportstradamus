#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

MARKETPLACE="obra/superpowers-marketplace"
MARKETPLACE_NAME="superpowers-marketplace"
PLUGIN="superpowers"

if ! claude plugin marketplace list 2>/dev/null | grep -q "$MARKETPLACE_NAME"; then
  claude plugin marketplace add "$MARKETPLACE"
fi

if ! claude plugin list 2>/dev/null | grep -q "^  > ${PLUGIN}@"; then
  claude plugin install "${PLUGIN}@${MARKETPLACE_NAME}"
fi
