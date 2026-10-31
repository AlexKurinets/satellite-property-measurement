#!/usr/bin/env bash
# Brings the stack up the way the README does and runs every command block the README documents.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${API_PORT:-8190}"
API="http://localhost:${API_PORT}"
TOKEN="${PROPMEASURE_API_TOKEN:-local-dev-token}"
PY="${PY:-python3}"
OUT="${SMOKE_OUT:-$(mktemp -d)}"
json() { "$PY" -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

echo "== compose up (seeds, trains and benchmarks on first start)"
docker compose up -d --build
for i in $(seq 1 240); do
  state=$(docker compose ps -a --format '{{.Service}} {{.State}} {{.ExitCode}}' | awk '$1=="trainer"{print $2" "$3}')
  if [[ "$state" == exited* && "$state" != "exited 0" ]]; then
    docker compose logs --no-color trainer | tail -n 60
    echo "trainer failed: $state"; exit 1
  fi
  if curl -fsS "$API/healthz" > "$OUT/health.json" 2>/dev/null; then break; fi
  sleep 10
done
cat "$OUT/health.json"; echo
[[ "$(json 'd["production_model"]' < "$OUT/health.json")" == "prop-unet-2.0" ]]

echo "== status"
docker compose run --rm trainer propmeasure status > "$OUT/status.txt"
cat "$OUT/status.txt"
grep -q "prop-unet-2.0   production" "$OUT/status.txt"
grep -q "lc-unet-1.0     retired" "$OUT/status.txt"

echo "== measure a live property through the API"
curl -fsS -X POST "$API/v1/measurements" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"property_id": "live-cs-0003"}' > "$OUT/measure.json"
cat "$OUT/measure.json"; echo
MID=$(json 'd["id"]' < "$OUT/measure.json")
curl -fsS "$API/v1/measurements/$MID/overlay.png" -o "$OUT/overlay.png"
[[ "$(head -c 4 "$OUT/overlay.png" | tail -c 3)" == "PNG" ]]

echo "== record an estimator's correction"
curl -fsS -X POST "$API/v1/measurements/$MID/corrections" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"feature": "lawn_sqft", "corrected_value": 4150, "corrected_by": "estimator-demo", "note": "back fence line"}' \
  > "$OUT/correction.json"
cat "$OUT/correction.json"; echo
curl -fsS "$API/v1/measurements/$MID" | json 'd["corrections"][0]["corrected_value"]' | grep -q 4150

echo "== label queue, models, benchmark"
curl -fsS "$API/v1/label-queue?limit=6" > "$OUT/queue.json"
json 'len(d)' < "$OUT/queue.json" | grep -qv '^0$'
curl -fsS "$API/v1/models" | json '[m["version"] + ":" + m["status"] for m in d]'
curl -fsS "$API/v1/benchmark" | json 'len(d)' | grep -q '^5$'

echo "== CLI blocks from the README"
MSYS_NO_PATHCONV=1 docker compose run --rm trainer sh -c \
  "propmeasure measure test-ma-0004 --overlay /tmp/test-ma-0004.png > /tmp/m.json && head -c 400 /tmp/m.json && ls -l /tmp/test-ma-0004.png"
docker compose run --rm trainer propmeasure evaluate --oracle --pipeline v1 > "$OUT/oracle.txt"
cat "$OUT/oracle.txt"
docker compose run --rm trainer propmeasure evaluate --model prop-unet-2.0 > "$OUT/eval.txt"
cat "$OUT/eval.txt"
grep -q "lawn_sqft" "$OUT/eval.txt"
docker compose run --rm trainer propmeasure label-queue --limit 6
if docker compose run --rm trainer propmeasure promote lc-unet-1.0 > "$OUT/promote.txt"; then
  echo "promoting the retired base model should have been refused"; exit 1
fi
cat "$OUT/promote.txt"
grep -q '"promoted": false' "$OUT/promote.txt"

echo "smoke OK"
