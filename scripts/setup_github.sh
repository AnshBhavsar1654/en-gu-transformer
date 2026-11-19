#!/usr/bin/env bash
# One-time GitHub setup for the CI/CD pipeline (needs the gh CLI, logged in as a repo admin):
#   - branch protection on main: changes only through pull requests, the three CI checks must pass,
#     no force pushes or deletions, and the rules apply to admins too (so nobody pushes to main directly);
#   - environments: `staging` (deploys unattended) and `production` (waits for your approval).
#
#   bash scripts/setup_github.sh                      # repo of the current checkout
#   REPO=owner/name bash scripts/setup_github.sh
#
# Run it after CI has run at least once (e.g. on the first pull request) so GitHub knows the check names.
# Required reviewers on environments are free for public repositories.
set -euo pipefail

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
BRANCH="${BRANCH:-main}"
ME_ID="$(gh api user -q .id)"
echo "Configuring $REPO (branch $BRANCH)"

# Job names from .github/workflows/ci.yml
gh api -X PUT "repos/$REPO/branches/$BRANCH/protection" --input - <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Lint", "Tests and model smoke test", "Docker build and container smoke test"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {"required_approving_review_count": 0},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
echo "  branch protection on $BRANCH: done"

gh api -X PUT "repos/$REPO/environments/staging" --input - <<JSON
{"deployment_branch_policy": {"protected_branches": true, "custom_branch_policies": false}}
JSON
echo "  environment staging: done"

gh api -X PUT "repos/$REPO/environments/production" --input - <<JSON
{
  "reviewers": [{"type": "User", "id": $ME_ID}],
  "deployment_branch_policy": {"protected_branches": true, "custom_branch_policies": false}
}
JSON
echo "  environment production (approval by you): done"
