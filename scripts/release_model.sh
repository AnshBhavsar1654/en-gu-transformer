#!/usr/bin/env bash
# Run after CI has staged the checkpoint in MLflow. The production job owns the rollout lock.
set -euo pipefail

: "${PROJECT:?}" "${REGION:?}" "${SERVICE:?}" "${MODEL_BUCKET:?}" "${MLFLOW_TRACKING_URI:?}" "${SHADOW_SERVICE:?}"
: "${SHADOW_MINUTES:?}"
[[ "$SHADOW_MINUTES" =~ ^[1-9][0-9]?$ ]]
[[ "$SERVICE" != "$SHADOW_SERVICE" ]]
evidence=artifacts/release
gate=$evidence/gate/report.json
sha=$(jq -er '.candidate_sha256 | select(test("^[0-9a-f]{64}$"))' "$gate")
candidate=$(jq -er '.version | "v" + tostring' "$gate")
stable=$(jq -er '.champion.version | select(. != null) | "v" + tostring' "$gate")
jq -e '.passed and (.dry_run | not) and (.promoted | not)' "$gate" >/dev/null

live=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format=json)
prev=$(jq -er '[.status.traffic[] | select(.percent == 100)] | .[0].revisionName' <<< "$live")
live_url=$(jq -er '.status.url' <<< "$live")
previous=$(gcloud run revisions describe "$prev" --project "$PROJECT" --region "$REGION" --format=json)
image=$(jq -er '.status.imageDigest' <<< "$previous")
runtime_sa=$(jq -er '.spec.serviceAccountName' <<< "$previous")
shadow_url=$(gcloud run services describe "$SHADOW_SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')
jq -e --arg origin "$shadow_url" --arg bucket "$MODEL_BUCKET" '
  any(.spec.containers[0].env[]; .name == "EN_GU_SHADOW_URL" and .value == $origin)
  and any(.spec.containers[0].volumeMounts[]; .name == "models" and .mountPath == "/models")
  and any(.spec.volumes[]; .name == "models" and .csi.volumeAttributes.bucketName == $bucket)
' <<< "$previous" >/dev/null || {
  echo "::error::The stable revision needs the private shadow URL and the configured model bucket mounted at /models."
  exit 1
}
python -m en_gu.smoke --url "$live_url" --expect-version "$stable" --no-feedback

path="/models/releases/$sha/best.pt"
uri="gs://$MODEL_BUCKET/releases/$sha"
for name in best.pt best.pt.version.json; do
  gcloud storage cp --if-generation-match=0 "$evidence/$name" "$uri/$name" ||
    cmp "$evidence/$name" <(gcloud storage cat "$uri/$name")
done

# The shadow service is provisioned privately before the first release. Its IAM and URL stay in place.
shadow_state=$(gcloud run services get-iam-policy "$SHADOW_SERVICE" --project "$PROJECT" --region "$REGION" --format=json)
jq -e 'all(.bindings[]?.members[]?; . != "allUsers" and . != "allAuthenticatedUsers")' <<< "$shadow_state" >/dev/null
gcloud run services describe "$SHADOW_SERVICE" --project "$PROJECT" --region "$REGION" --format=json |
  jq -e '.metadata.annotations["run.googleapis.com/invoker-iam-disabled"] != "true"' >/dev/null
gcloud run services update "$SHADOW_SERVICE" --project "$PROJECT" --region "$REGION" --image "$image" \
  --update-env-vars "EN_GU_MODEL_SOURCE=path,EN_GU_MODEL_PATH=$path" --quiet
shadow_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Reserve a fixed comparison window. Allow time for Cloud Logging ingestion before querying.
sleep "$((SHADOW_MINUTES * 60))"
shadow_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sleep 30
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=\"$SERVICE\" AND
  resource.labels.location=\"$REGION\" AND resource.labels.revision_name=\"$prev\" AND
  (jsonPayload.event=\"shadow_comparison\" OR jsonPayload.event=\"shadow_dropped\") AND
  timestamp >= \"$shadow_started\" AND timestamp <= \"$shadow_end\"" --project "$PROJECT" \
  --limit 10001 --format=json > "$RUNNER_TEMP/shadow.json"
[[ "$(jq length "$RUNNER_TEMP/shadow.json")" -le 10000 ]]
python -m en_gu.release shadow --logs "$RUNNER_TEMP/shadow.json" --stable "$stable" --candidate "$candidate" \
  --minimum "${SHADOW_MIN_REQUESTS:-20}" --min-similarity "${SHADOW_MIN_SIMILARITY:-0.5}" --out "$evidence/shadow.json"

# Capture the stable revision before changing traffic. Recheck for out-of-band changes.
active=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format=json |
  jq -er '[.status.traffic[] | select(.percent == 100)] | .[0].revisionName')
[[ "$active" == "$prev" ]]
suffix="m-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
new="$SERVICE-$suffix"
gcloud run services update "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$image" \
  --service-account "$runtime_sa" --revision-suffix "$suffix" --no-traffic --tag candidate \
  --update-env-vars "EN_GU_MODEL_SOURCE=path,EN_GU_MODEL_PATH=$path" --quiet

traffic() {
  gcloud run services update-traffic "$SERVICE" --project "$PROJECT" --region "$REGION" "$@" --quiet
}
rollback_on_error() {
  code=$?
  if (( code != 0 )); then
    if traffic --to-revisions "$prev=100"; then
      echo "::error::Release failed. Traffic is back on $prev."
    else
      echo "::error::Automatic rollback failed. Restore $prev with the rollback runbook."
    fi
  fi
  return "$code"
}
trap rollback_on_error EXIT
trap 'exit 130' INT TERM
traffic --update-tags "stable=$prev"
urls=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format=json)
candidate_url=$(jq -er '.status.traffic[] | select(.tag == "candidate") | .url' <<< "$urls")
stable_url=$(jq -er '.status.traffic[] | select(.tag == "stable") | .url' <<< "$urls")

python -m en_gu.smoke --url "$candidate_url" --expect-version "$candidate" --no-feedback
floor=$(jq -er '.min_chrf' monitoring/golden-baseline.json)
python -m en_gu.quality_canary --url "$candidate_url" --expect-version "$candidate" --min-chrf "$floor" \
  --out "$evidence/golden"
traffic --to-revisions "$new=10,$prev=90"
canary_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python -m en_gu.canary --url "$candidate_url" --baseline-url "$stable_url" --expect-version "$candidate" \
  --stable-version "$stable" --minutes "${CANARY_MINUTES:-5}" --out "$evidence/canary.json"
canary_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sleep 30
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=\"$SERVICE\" AND
  resource.labels.location=\"$REGION\" AND (resource.labels.revision_name=\"$prev\" OR
  resource.labels.revision_name=\"$new\") AND log_id(\"run.googleapis.com/requests\") AND
  httpRequest.requestMethod=\"POST\" AND httpRequest.requestUrl=~\"/translate$\" AND
  timestamp >= \"$canary_started\" AND timestamp <= \"$canary_end\"" --project "$PROJECT" \
  --limit 10001 --format=json > "$RUNNER_TEMP/canary-traffic.json"
[[ "$(jq length "$RUNNER_TEMP/canary-traffic.json")" -le 10000 ]]
python -m en_gu.release traffic --logs "$RUNNER_TEMP/canary-traffic.json" --stable "$prev" --candidate "$new" \
  --minimum "${CANARY_MIN_REAL_REQUESTS:-10}" --out "$evidence/traffic.json"
traffic --to-revisions "$new=100"
python -m en_gu.smoke --url "$live_url" --expect-version "$candidate" --no-feedback
python -m en_gu.release finalize --tracking-uri "$MLFLOW_TRACKING_URI" --gate "$gate" \
  --shadow "$evidence/shadow.json" --canary "$evidence/canary.json" --golden "$evidence/golden.json" \
  --traffic "$evidence/traffic.json"
