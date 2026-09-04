# Deployment

The inference service is a container. Everything below assumes a checkpoint at
`models/final.pt`; copy one there or mount it at runtime.

---

## Local, without Docker

```bash
pip install -e ".[api]"
GMAI_CHECKPOINT=runs/<run>/final.pt uvicorn gmai.api:app --port 8000
```

```bash
curl -X POST localhost:8000/move \
  -H 'Content-Type: application/json' \
  -d '{"fen":"4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"}'
```

```json
{
  "uci": "a2c4",
  "san": "Qc4",
  "q_value": 0.75,
  "inference_ms": 6.155,
  "in_scope": true,
  "scope_detail": "KQvK",
  "legal_moves": 26
}
```

`in_scope` is the field worth reading. For a position outside the three trained
endgames it goes false with a reason, and the move — still legal, still
returned — comes with no support from training:

```json
{ "in_scope": false, "scope_detail": "both sides have material: out of scope" }
```

Interactive docs at `/docs`, liveness at `/health`, Prometheus metrics at
`/metrics`.

---

## Docker

Two images, deliberately separate:

| | `Dockerfile` | `Dockerfile.train` |
|---|---|---|
| Purpose | serving | training |
| torch | CPU wheel | CUDA runtime |
| Extras | fastapi, uvicorn, prometheus-client | gymnasium, matplotlib, the solver cache |
| Runs as | non-root (`gmai`, uid 10001) | root (development image) |

Folding them together would put a 2.5 GB CUDA wheel and the whole training
stack into the thing that answers HTTP requests. Single-position inference is
dominated by Python overhead, not matmul, so the CPU wheel loses nothing.

`tests/test_serving_boundary.py` enforces this: it imports `gmai.api` with the
training-only packages blocked, and fails if anything in the serving path
reaches for them. It also checks that the blocker itself works, so the test
cannot pass vacuously.

```bash
docker build -t gmai-api .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" gmai-api
```

To bake a checkpoint in rather than mounting it, place it at `models/final.pt`
before building — `.dockerignore` excludes `runs/` and loose `*.pt`, so nothing
else is copied by accident.

---

## Full stack

```bash
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | `/docs` for the OpenAPI UI |
| Prometheus | http://localhost:9090 | scrapes the API every 15 s |
| Grafana | http://localhost:3000 | anonymous viewer, no login |

The dashboard is provisioned from
`deploy/grafana/provisioning/dashboards/gmai.json` and versioned with the code,
so it is reviewable in a pull request rather than clicked together and lost.

It shows p50/p95/p99 inference latency, request rate by outcome, the Q-value
distribution of selected moves, and the **out-of-scope rate** — the share of
traffic asking about positions the model was never trained on. A high value
there means the service is being used outside its documented domain, which is
worth an alert well before latency is.

---

## Publishing

Tagging pushes a multi-arch image to GHCR:

```bash
git tag v0.2.0 && git push --tags
```

`release.yml` builds for `linux/amd64` and `linux/arm64`, tags by semver and
commit SHA, and attaches the model card to the GitHub release.

```bash
docker pull ghcr.io/<owner>/gmai:0.2.0
```

---

## Cloud Run

Scales to zero, so an idle demo costs nothing.

```bash
PROJECT=your-project
REGION=europe-west1

gcloud builds submit --tag "gcr.io/$PROJECT/gmai"

gcloud run deploy gmai \
  --image "gcr.io/$PROJECT/gmai" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8000
```

Notes worth knowing before the first deploy:

- **Memory.** torch plus a loaded checkpoint sits around 600 MB; 2 GiB leaves
  room and costs nothing while idle.
- **Cold starts.** Importing torch takes several seconds. With
  `--min-instances 0` the first request after idle is slow. Set it to 1 if that
  matters, and accept the standing cost.
- **The checkpoint must be in the image.** Cloud Run has no persistent volume.
  Either bake it in at build time or fetch it from GCS on startup.
- **Health checks.** Cloud Run probes the port, not `/health`. The endpoint is
  still what to point uptime monitoring at, since it reports whether a
  checkpoint actually loaded.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GMAI_CHECKPOINT` | `/app/models/final.pt` | Path to the checkpoint |
| `GMAI_DEVICE` | `cpu` in the image | `cpu` or `cuda` |

If the checkpoint is missing the service still starts, and `/health` reports
`model_loaded: false` while `/move` returns 503. Starting and reporting the
problem is more useful than refusing to start, since it makes the failure
visible to whatever is watching `/health`.
