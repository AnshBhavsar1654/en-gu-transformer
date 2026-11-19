#!/usr/bin/env bash
# One-time Google Cloud setup so GitHub Actions can deploy to Cloud Run without a stored key:
#   - enables the Cloud Run, Artifact Registry and IAM credential APIs;
#   - creates a Docker repository in Artifact Registry;
#   - creates a deployer service account with only the roles the pipeline needs;
#   - creates a Workload Identity pool + GitHub OIDC provider that trusts only this repository;
#   - sets the GCP_* repository variables that cd.yml reads (if the gh CLI is available).
#
#   PROJECT_ID=my-project bash scripts/setup_gcp_cicd.sh
#
# Needs gcloud logged in as a project owner, and billing enabled on the project. Safe to re-run: existing
# resources are kept. Set a budget alert in the console before the first deploy (see docs plan, Phase 4).
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID to your GCP project id}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-en-gu}"
GITHUB_REPO="${GITHUB_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo AnshBhavsar1654/en-gu-transformer)}"
SA_NAME="${SA_NAME:-github-deployer}"
POOL="${POOL:-github}"
PROVIDER="${PROVIDER:-github-oidc}"

SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
exists() { "$@" >/dev/null 2>&1; }

echo "== APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com --project "$PROJECT_ID"

echo "== Artifact Registry repository $AR_REPO ($REGION)"
exists gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" --project "$PROJECT_ID" ||
  gcloud artifacts repositories create "$AR_REPO" --repository-format docker --location "$REGION" \
    --description "English to Gujarati API images" --project "$PROJECT_ID"

echo "== Deployer service account $SA"
exists gcloud iam service-accounts describe "$SA" --project "$PROJECT_ID" ||
  gcloud iam service-accounts create "$SA_NAME" --display-name "GitHub Actions deployer" --project "$PROJECT_ID"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$SA" --role "$role" \
    --condition None --quiet >/dev/null
done
# Cloud Run revisions run as the default compute service account; the deployer may act as it, nothing more
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member "serviceAccount:$SA" \
  --role roles/iam.serviceAccountUser --project "$PROJECT_ID" --quiet >/dev/null

echo "== Workload Identity Federation for $GITHUB_REPO"
exists gcloud iam workload-identity-pools describe "$POOL" --location global --project "$PROJECT_ID" ||
  gcloud iam workload-identity-pools create "$POOL" --location global --display-name "GitHub Actions" \
    --project "$PROJECT_ID"
exists gcloud iam workload-identity-pools providers describe "$PROVIDER" --workload-identity-pool "$POOL" \
  --location global --project "$PROJECT_ID" ||
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --workload-identity-pool "$POOL" --location global --project "$PROJECT_ID" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${GITHUB_REPO}'"
gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT_ID" --quiet \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  >/dev/null

WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
echo
echo "Repository variables for $GITHUB_REPO (Settings > Secrets and variables > Actions > Variables):"
echo "  GCP_PROJECT_ID   = $PROJECT_ID"
echo "  GCP_REGION       = $REGION"
echo "  GCP_AR_REPO      = $AR_REPO"
echo "  GCP_WIF_PROVIDER = $WIF_PROVIDER"
echo "  GCP_DEPLOY_SA    = $SA"

if command -v gh >/dev/null && [ "${SET_GH_VARS:-1}" = "1" ]; then
  gh variable set GCP_PROJECT_ID --repo "$GITHUB_REPO" --body "$PROJECT_ID"
  gh variable set GCP_REGION --repo "$GITHUB_REPO" --body "$REGION"
  gh variable set GCP_AR_REPO --repo "$GITHUB_REPO" --body "$AR_REPO"
  gh variable set GCP_WIF_PROVIDER --repo "$GITHUB_REPO" --body "$WIF_PROVIDER"
  gh variable set GCP_DEPLOY_SA --repo "$GITHUB_REPO" --body "$SA"
  echo "Set them with gh. The next merge to main deploys to staging."
fi
