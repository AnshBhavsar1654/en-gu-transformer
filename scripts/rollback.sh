#!/usr/bin/env bash
# Roll a Cloud Run service back: send 100% of traffic to an earlier revision, then wait until the public URL
# answers /ready again. Prints how long it took (the target in RUNBOOKS.md is under 2 minutes).
#
#   SERVICE=en-gu-api bash scripts/rollback.sh                 # to the newest ready revision older than the live one
#   SERVICE=en-gu-api bash scripts/rollback.sh en-gu-api-abc1234-123-1    # to a named revision
#   DRY_RUN=1 SERVICE=en-gu-api bash scripts/rollback.sh       # only print what would happen
#
# Needs gcloud logged in with run.admin (or run.developer) on the project. Only traffic moves: no revision is
# deleted, so rolling forward again is the same command with the newer revision's name.
set -euo pipefail

: "${SERVICE:?set SERVICE to the Cloud Run service name (en-gu-api, en-gu-api-staging, ...)}"
REGION="${REGION:-us-central1}"
PROJECT="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
TARGET="${1:-}"
start=$(date +%s)

describe=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format=json)
url=$(jq -r '.status.url' <<<"$describe")
echo "Traffic on $SERVICE now:"
jq -r '.status.traffic[]? | select((.percent // 0) > 0) | "  \(.percent)%  \(.revisionName)"' <<<"$describe"
live=$(jq -r '[.status.traffic[]? | select(.revisionName != null and (.percent // 0) > 0)]
              | max_by(.percent) | .revisionName // empty' <<<"$describe")

if [ -z "$TARGET" ]; then
  # Newest first; skip everything down to and including the live revision, then take the first ready one
  TARGET=$(gcloud run revisions list --service "$SERVICE" --region "$REGION" --project "$PROJECT" \
             --sort-by '~metadata.creationTimestamp' --format=json |
           jq -r --arg live "$live" '
             [.[] | {name: .metadata.name,
                     ready: ([.status.conditions[]? | select(.type == "Ready") | .status] | first == "True")}]
             | (map(.name) | index($live)) as $i
             | if $i == null then empty else .[$i + 1:] | map(select(.ready)) | first | .name // empty end')
fi
if [ -z "$TARGET" ]; then
  echo "No earlier ready revision to roll back to (live: ${live:-none})." >&2
  exit 1
fi
if [ "$TARGET" = "$live" ]; then
  echo "$TARGET is already serving; nothing to do."
  exit 0
fi

echo "Rolling $SERVICE back: ${live:-?} -> $TARGET"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(dry run: no change made)"
  exit 0
fi
gcloud run services update-traffic "$SERVICE" --region "$REGION" --project "$PROJECT" \
  --to-revisions "${TARGET}=100" --quiet

# Cloud Run starts an instance of the old revision on demand; wait for it to serve
for _ in $(seq 1 60); do
  if body=$(curl -fsS --max-time 10 "$url/ready" 2>/dev/null); then
    echo "Ready: $body"
    echo "Rolled back to $TARGET in $(( $(date +%s) - start )) s. Check: $url/info"
    exit 0
  fi
  sleep 3
done
echo "Traffic moved to $TARGET, but $url/ready did not answer within 3 minutes." >&2
exit 1
