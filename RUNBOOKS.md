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


## Monitoring

Use the local Compose profile from the [README](README.md#monitoring) before running a drill on staging.
Prometheus shows firing rules at `http://localhost:9090/alerts`. For local demonstrations, export stdout
with `docker compose logs --no-log-prefix api` and run the drift and quality commands from the README.
The file exporter serves the `.prom` outputs from `artifacts/monitoring`.

### Production setup

1. Set the `PRODUCTION_API_URL` repository variable to the deployed API endpoint.
2. Create a dedicated Google Cloud service account with `roles/logging.viewer` and grant the repository's
   GitHub Workload Identity principal `roles/iam.workloadIdentityUser` on that account, using the
   [deployment setup](scripts/setup_gcp_cicd.sh) as the reference. Set the account email in
   `GCP_MONITORING_SA`. The drift job also uses the existing `GCP_PROJECT_ID`, `GCP_WIF_PROVIDER`,
   `GCP_REGION` and `PRODUCTION_SERVICE` settings.
3. Dispatch **Production monitoring** and review the two artifact bundles. The scheduled run starts daily
   at 03:23 UTC. For Grafana Cloud, set `GRAFANA_INFLUX_URL` to your metrics instance's
   `/api/v1/push/influx` endpoint, set `GRAFANA_METRICS_USER` to its instance ID, and add an access policy
   token with metrics write scope as the `GRAFANA_METRICS_TOKEN` repository secret.
   [Grafana Cloud ingest setup](https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-influxdb/push-from-telegraf/)
4. Import the [daily-signal dashboards](monitoring/grafana/dashboards) with your Grafana Cloud Prometheus
   data source. Use the [scheduled rules](monitoring/scheduled-rules.yml) in that same data source's ruler,
   and configure contact routing for `page` and `review`. For fleet service health, use the Cloud Run
   service's Metrics tab. An individual HTTP scrape of a load-balanced Cloud Run URL only samples the
   instance that answered. The [live scrape rules](monitoring/rules.yml) are for the local Prometheus target.
   [Cloud Run metrics](https://cloud.google.com/run/docs/monitoring)

### Input drift

Check sample counts and the named features in the 24-hour and 7-day reports. Compare the affected
revision, input lengths, scripts, language flags, and token coverage against the reference. A missing
score means there was insufficient usable data in that window. If a shift is sustained, record the
model/version, affected windows, feature results, and observed sample counts in a retraining review.
Use the same review path for a significant increase in negative feedback.

### Golden-set failure

Open the latest quality result and distinguish a low score from a failed probe. If a rollout was
in progress, repeat the probe on a single tagged revision to avoid mixing versions. For a low score
or broken recent revision, compare `/info` with the expected build, run the canary on the stable
revision, and shift traffic to stable using the rollback procedure above. Verify `/ready` and rerun the
200-pair canary against production after the traffic change. Refresh the baseline only after approving
a new champion and measuring it on the same fixed sample.

### Service or output regression

Check request rates, 5xx, p95 latency, and activity or startup by revision. For output alerts, inspect
repetition and length-limit hits with the input drift report and the golden-set scores. Use the
rollback procedure when the regression follows a new revision. The service, drift, and feedback
thresholds are initial settings to calibrate with representative traffic.

### Missing monitoring results

Check the scheduled workflow status, Cloud Logging permissions, endpoint readiness, and the Grafana
export step. Re-dispatch **Production monitoring** after correcting the failing dependency. Locally,
verify that both daily `.prom` files exist, the exporter can read them, and Prometheus has scraped
the exporter. An empty traffic window can produce a fresh report without a drift score.

## Retraining and model releases

A monthly schedule, persistent drift report, or manual dispatch opens a retraining review. After a human has reviewed
and approved corrections, export a private JSONL with string fields `source` and `target`. Use this command to normalize,
validate, and append the approved rows.

```bash
python -m en_gu.reviewed private/approved.jsonl
dvc add data/reviewed/accepted.jsonl
```

The validator checks script, length, duplicate sources, and overlap with either side of the frozen evaluation set.
Commit only the DVC pointer, and push correction data to a private DVC remote. Use the printed `new_pairs` count
when dispatching **Review model retraining**. The default threshold is 100 new pairs. A count of zero opens a manual
review. The pipeline incorporates accepted rows after the base corpus cap and frozen-set exclusions.

Run training from a GPU-enabled environment. Bring the same corpus and approved DVC snapshot into the existing
Kaggle runner, or use the CLI directly.

```bash
python -m en_gu all --config configs/base.yaml --config configs/full-corpus.yaml
```

Use the base config alone to reproduce the one-million-pair setting. A candidate needs an exported self-contained
checkpoint and the current evaluation fingerprint. Compare and stage it without changing the live registry alias.

```bash
python -m en_gu promote --candidate artifacts/best.pt --stage-candidate
```

### Production release configuration

- Serve the registry champion from the existing GCS model mount at `/models`, including the checkpoint version
  sidecar. Set `EN_GU_MODEL_DELIVERY=gcs` and `GCP_MODEL_BUCKET`. The release validates the current API model
  version against the registry before changing traffic.
- Provision a private Cloud Run shadow service with the same read-only GCS mount, startup probe, and runtime
  configuration as the API. Grant the production runtime account `roles/run.invoker` on the shadow service.
  Set `EN_GU_SHADOW_SERVICE` to its name and `EN_GU_SHADOW_URL` to its HTTPS origin, then deploy the API
  with shadow sampling enabled. Sampling defaults to 5% and compares the first 512 characters of the outputs.
  Request and response text stays in memory. Only aggregate comparisons and model labels reach logs.
- Set `MLFLOW_TRACKING_URI` to the reachable registry and configure `MLFLOW_TRACKING_TOKEN` if needed. The release
  identity needs permission to deploy Cloud Run revisions, read logs, and read and create model objects in the
  configured bucket. The release uses the production GitHub environment approval.
- Upload the trusted exported checkpoint as `gs://<bucket>/incoming/<sha256>/best.pt`, then dispatch **Release
  translation model** with the SHA-256. The job verifies the digest, runs the full promotion gate, stages the
  model, and records shadow and golden-set evidence before routing 10% to the tagged candidate.
- The canary compares both tagged endpoints and Cloud Run request logs for the two revisions. Insufficient
  samples or guard regressions stop the release. A clean result transfers all traffic and advances the registry
  champion. The job uploads the reports, restores the previous revision after rollout failures, and opens a
  review issue on rejection.

For a manual rollback, restore the previous Cloud Run revision using the rollback procedure, verify the public API
version, and run `python -m en_gu registry --rollback --set "tracking.uri=$MLFLOW_TRACKING_URI"` if the registry
champion had advanced. Each timeline should record the code and data versions, checkpoint digest, and gate and
rollout reports. The release jobs share their Cloud Run service lock with code deployments.

Development checks cover registry transitions and compare fixture traffic locally. The production workflow requires
a configured registry, model bucket, authenticated shadow service, and enough traffic to exercise both windows.
