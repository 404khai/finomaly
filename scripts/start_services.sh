#!/usr/bin/env bash
# Boot the Finomaly streaming + cache stack (Redpanda + Redpanda Console + Redis)
# and block until every service reports healthy.
#
# Usage:
#   scripts/start_services.sh          # start and wait for health
#   scripts/start_services.sh --down   # tear the stack back down
set -euo pipefail

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docker-compose.yml"

if [[ "${1:-}" == "--down" ]]; then
  echo ">> Stopping Finomaly stack..."
  docker compose -f "$COMPOSE_FILE" down
  exit 0
fi

echo ">> Starting Redpanda + Redis (docker compose up -d)..."
docker compose -f "$COMPOSE_FILE" up -d

echo ">> Waiting for services to become healthy..."
services=(redpanda redis)
for svc in "${services[@]}"; do
  # Blocks until the container's healthcheck turns healthy, up to ~60s.
  timeout=60
  while (( timeout > 0 )); do
    status="$(docker inspect \
      --format '{{.State.Health.Status}}' \
      "finomaly-${svc}" 2>/dev/null || echo "starting")"
    if [[ "$status" == "healthy" ]]; then
      echo "   ✓ ${svc} healthy"
      break
    fi
    sleep 2
    timeout=$((timeout - 2))
  done
  if [[ "$status" != "healthy" ]]; then
    echo "   ✗ ${svc} did not become healthy in time (last: ${status})"
    echo "     Run: docker compose -f \"$COMPOSE_FILE\" logs ${svc}"
    exit 1
  fi
done

cat <<'EOF'

>> Finomaly stack is up:
   Redpanda (Kafka API)   : localhost:19092
   Redpanda Console (web) : http://localhost:8080
   Redis                  : localhost:6379

   Stop with: scripts/start_services.sh --down
EOF
