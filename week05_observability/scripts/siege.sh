#!/usr/bin/env bash
# Hammer /v1/predict with siege.
set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"
PAYLOAD="${3:-scripts/payload.json}"

siege -c 1 -H 'Content-Type: application/json' \
  "http://${HOST}:${PORT}/v1/predict POST < ${PAYLOAD}"
