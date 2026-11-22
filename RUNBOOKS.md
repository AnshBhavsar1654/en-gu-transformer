# Runbooks

Operating the deployed English → Gujarati service on Cloud Run. Each runbook says when to use it, the commands,
and how to check that it worked.

| Service | What | Default name |
|---|---|---|
| Production API | FastAPI inference service | `en-gu-api` |
| Production UI | Streamlit client of the production API | `en-gu-ui` |
| Staging API / UI | Same images, deployed first, no users | `en-gu-api-staging` / `en-gu-ui-staging` |

Region `us-central1` unless the `GCP_REGION` repository variable says otherwise. All commands need `gcloud` logged in
to the project (`gcloud auth login`, `gcloud config set project <id>`) and, for the scripts, `jq` and `bash`.

Useful views:

```bash
gcloud run services describe en-gu-api --region us-central1 --format='yaml(status.traffic)'   # who gets traffic
gcloud run revisions list --service en-gu-api --region us-central1                              # every revision
curl -s https://<api-url>/info | jq '{build, revision, model_version, uptime_seconds}'           # what answered
```

---

## 1. Roll back production (target: under 2 minutes)

**When**: a bad release is live (errors, wrong translations, latency) and the cause isn't obvious. Roll back
first, investigate after. The CD pipeline already does this automatically when a smoke test or the canary fails;
this runbook is for problems found later.

```bash
SERVICE=en-gu-api bash scripts/rollback.sh                  # back to the previous ready revision
SERVICE=en-gu-api bash scripts/rollback.sh <revision-name>  # or to a specific one
DRY_RUN=1 SERVICE=en-gu-api bash scripts/rollback.sh        # see which revision it would pick
make rollback SERVICE=en-gu-api                             # same thing through make
```

The script prints the current traffic split, moves 100% of traffic to the target revision, waits until the public
URL answers `/ready`, and prints the elapsed time. Revision names are `<service>-<sha7>-<run id>-<attempt>`, so the
commit of each revision is visible in the name (and in its `commit-sha` label).

Without the script (console or a machine without `jq`):

```bash
gcloud run services update-traffic en-gu-api --region us-central1 --to-revisions <previous-revision>=100
```

or Cloud Run console → `en-gu-api` → *Revisions* → *Manage traffic* → 100% on the previous revision.

**Check**: `curl -s <api-url>/info | jq .build,.revision` shows the old commit and revision;
`python -m en_gu smoke-test --url <api-url> --no-feedback` passes.

**After**: `main` still contains the bad commit, so the next merge would redeploy it. Revert it with a pull
request (`git revert <sha>`), and write down what happened.

**Roll forward again**: the same command with the newer revision's name. Rolling back never deletes anything.

## 2. A deploy failed in CD

The *Deploy to staging/production* job summary shows the new and previous revision and which step failed.

| Failed step | What it means | Users affected? |
|---|---|---|
| Deploy the new revision with no traffic | The revision never became ready: image doesn't start, or the model didn't load within 240 s (startup probe on `/ready`) | No: traffic never moved |
| Smoke test the new revision | It starts but answers wrongly (see the FAIL lines) | No: it had 0% traffic |
| Canary | Errors or slower than the stable revision while serving 10% (see the canary table) | Up to 10% of requests for the canary window |
| Smoke test the public URL | Something broke after promotion | Briefly; already rolled back |
| Deploy the UI | The Streamlit service didn't come up | UI only; the API stays on the new revision |

In every case except the UI, the rollback step has already sent 100% of traffic back to the previous revision.
Read the new revision's logs:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.revision_name="<new-revision>"' \
  --limit 100 --format='value(textPayload)'
```

The failed revision stays in the revision list with 0% traffic (and the `candidate` tag until the next deploy).
Fix forward with a new pull request, or re-run the workflow if the failure was transient.

## 3. Canary details and manual canary

Production rollouts go 0% → smoke test → 10% for 5 minutes → 100%. During the window `python -m en_gu canary`
sends a translation every 5 s to the new revision (tag URL `candidate---…`) and to the old one (`stable---…`) and
fails if the new revision's error rate is above 2% or its p95 latency is above
max(1.5 × stable p95, stable p95 + 250 ms). The repository variables `CANARY_PERCENT`, `CANARY_MINUTES`,
`CANARY_MAX_ERROR_RATE` and `CANARY_MAX_P95_RATIO` change these. The probes are synthetic: request ids start with
`canary-` and the User-Agent is `en-gu-canary`.

By hand (e.g. to give a revision a longer look):

```bash
gcloud run services update-traffic en-gu-api --region us-central1 --to-revisions <new>=10,<old>=90
python -m en_gu canary --url https://candidate---<api-host> --baseline-url https://stable---<api-host> --minutes 30
gcloud run services update-traffic en-gu-api --region us-central1 --to-revisions <new>=100   # or <old>=100
```

## 4. Measure cold starts / compare model delivery

The services scale to zero, so the first request after ~15 idle minutes starts a new instance: container start,
torch import, model download or read, model load. The startup probe holds requests until `/ready`, so a cold
request is slow rather than failed.

```bash
python -m en_gu cold-start --url <staging-api-url> --samples 3          # waits 16 min before each sample
python -m en_gu cold-start --url <staging-api-url> --samples 1 --no-initial-wait   # if it is idle already
```

Each sample reports the client-side time to the first translation, the model-load time inside the instance, and
whether a fresh instance really answered. To compare the three ways of getting the model into the container, set
the repository variable `EN_GU_MODEL_DELIVERY`, re-run the CD workflow (*Actions → CD → Run workflow*), and measure
staging after each:

| `EN_GU_MODEL_DELIVERY` | How the model arrives | Extra setup |
|---|---|---|
| `hub` (default) | Downloaded from the public Hugging Face repo at startup | None |
| `gcs` | Read from a Cloud Storage bucket mounted at `/models` (Cloud Storage FUSE, gen2) | `MODEL_BUCKET=… bash scripts/setup_gcp_cicd.sh`, then `make upload-model-gcs BUCKET=…` |
| `baked` | Inside the image (`api-baked` target, ~120 MB larger) | None: CD downloads it from Hugging Face while building |

Record the results in the README table. Keeping one instance warm (`--min-instances 1`) removes cold starts but is
billed around the clock: only do it if the budget allows.

## 5. Budget alert fired

`scripts/setup_gcp_cicd.sh` (with `BILLING_ACCOUNT`) creates a monthly budget (default 5 USD) that e-mails the
billing admins at 50%, 90% and 100% of actual spend and at 100% of forecast spend. A budget only alerts; it does
not stop anything.

1. Billing → Reports, group by service and SKU: find what costs money.
2. Usual suspects: an instance kept warm (`gcloud run services describe <svc> --format='value(spec.template.metadata.annotations)'`
   shows `minScale`), heavy traffic (the services are capped at 2 instances each), old images piling up in
   Artifact Registry (add a cleanup policy or delete old tags).
3. Emergency stop for the API: `gcloud run services update en-gu-api --region us-central1 --max-instances 1`,
   or remove public access: `gcloud run services remove-iam-policy-binding en-gu-api --region us-central1
   --member allUsers --role roles/run.invoker`.

## 6. Deploy by hand (CD unavailable)

```bash
IMAGE=us-central1-docker.pkg.dev/<project>/en-gu/en-gu-api:<sha>
gcloud run deploy en-gu-api --image "$IMAGE" --region us-central1 --no-traffic --tag candidate
python -m en_gu smoke-test --url https://candidate---<api-host> --no-feedback
gcloud run services update-traffic en-gu-api --region us-central1 --to-revisions <new-revision>=100
```

`gcloud run deploy` keeps the service's other settings (resources, probes, env vars) from the previous revision.
