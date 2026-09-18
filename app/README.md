# Research demonstration application

**Research project:** Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages
**Benchmark software:** Multilingual LLM Safety Evaluation Framework

A deployable demonstration layer over the **frozen** research pipeline. It
reads the frozen outputs and never writes to them: no research dataset, label,
benchmark output, statistic or figure is modified by running this application.

---

## Architecture

```
Browser (React + TypeScript + Vite)
        │
        │  fetch /api/*        no Python in the browser
        ▼
FastAPI backend  (app/backend/main.py)
        │
        ├── XLM-R  classifier   outputs/analysis/classifier_final/xlmr_final/
        ├── MuRIL  classifier   outputs/analysis/classifier_final/muril_final/
        │        loaded once at startup, held in memory, CUDA when available
        │
        └── frozen research outputs (read-only)
              outputs/evaluated_final/FREEZE_MANIFEST.json
              outputs/analysis/final_tables/*.csv
              outputs/analysis/statistics_confirmatory_final/*.csv
              outputs/analysis/classifier_final/*/final_metrics.json
```

The Results section is populated entirely from those files, so it reflects the
actual frozen experiment rather than hard-coded numbers.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | service status, device, per-model availability |
| GET | `/api/meta` | taxonomy, languages, available classifiers |
| POST | `/api/evaluate` | classify one LLM response |
| GET | `/api/benchmark` | frozen benchmark results and caveats |
| GET | `/api/methodology` | pipeline, evaluator hash, scope, limitations |

Interactive API docs are served at `/docs`.

### Frontend sections

| Section | Source |
|---|---|
| Safety Evaluation | live `POST /api/evaluate` |
| Benchmark / Results | `GET /api/benchmark` — frozen outputs |
| About the Research | static narrative (design, not numbers) |
| Methodology | `GET /api/methodology` — frozen manifest |

---

## Running locally

Two processes. Run both from the **repository root**.

### 1. Backend

```bash
# dependencies (already installed in .venv)
pip install fastapi "uvicorn[standard]"

uvicorn app.backend.main:app --reload --port 8077
```

Startup loads both classifiers and prints their status. A missing model is
reported rather than fatal, so Results and Methodology still work without
trained weights.

Verify:

```bash
curl http://127.0.0.1:8077/api/health
```

### 2. Frontend

```bash
cd app/frontend
npm install
npm run dev
```

Open the URL Vite prints. In development `/api` is proxied to
`http://127.0.0.1:8077`, so no CORS configuration is needed.

> **Port note.** This machine already had services on 8000, 5173 and 5174, so
> the backend uses **8077** and Vite will pick the first free port (it chose
> **5175**). Vite binds IPv6 `localhost`; use `http://localhost:<port>` rather
> than `127.0.0.1` if a request appears to hang.
>
> To point the frontend at a different backend port:
> ```bash
> VITE_API_TARGET=http://127.0.0.1:9000 npm run dev
> ```

### 3. Production build

```bash
cd app/frontend
npm run build      # -> app/frontend/dist
npm run preview    # serve the build locally
```

Current build output: ~174 kB JS (54 kB gzipped), ~10 kB CSS.

---

## Configuration

| Variable | Side | Purpose |
|---|---|---|
| `VITE_API_BASE` | frontend build | API origin in production. Empty in dev to use the proxy. |
| `VITE_API_TARGET` | frontend dev | Backend target for the dev proxy. Default `http://127.0.0.1:8077`. |
| `ALLOWED_ORIGINS` | backend | Comma-separated CORS allowlist. Defaults to `*` — **set this in production.** |

Copy `app/frontend/.env.example` to `.env` and set `VITE_API_BASE` for a
deployed build.

---

## Deployment options

The frontend is a static bundle and the backend is a standard ASGI app, so they
deploy independently.

### Frontend — any static host

Build with `VITE_API_BASE` pointing at the deployed API, then upload
`app/frontend/dist`:

- Netlify, Vercel, Cloudflare Pages, GitHub Pages
- Any CDN or nginx/Apache document root

### Backend — ASGI host with a GPU or patient CPU

```bash
uvicorn app.backend.main:app --host 0.0.0.0 --port 8077 --workers 1
```

Use **one worker**. Each worker loads its own copy of both models; two workers
double the memory for no throughput gain on a single GPU.

Suitable targets: Hugging Face Spaces (Docker), Render, Railway, Fly.io, Azure
Container Apps, Google Cloud Run (CPU, with a raised request timeout), or any
VM with `systemd` plus an nginx reverse proxy.

### The model-weights problem

The trained classifiers are **large and not in version control**:

| Model | Size |
|---|---|
| `xlmr_final` | ~4.2 GB |
| `muril_final` | ~3.6 GB |

A clone of this repository will not contain them. For deployment, either:

1. **Publish to the Hugging Face Hub** and load by repo id — simplest path, and
   it removes the weights from the deployment image entirely.
2. **Mount a volume** containing `outputs/analysis/classifier_final/` and point
   `MODEL_DIRS` at it.
3. **Bake into a Docker image** — works, but produces a multi-gigabyte image.

Option 1 is recommended. Note that `model.safetensors` is byte-identical to the
final checkpoint in each directory, so only one copy needs publishing.

---

## Required before public deployment

These are genuine prerequisites, not polish:

1. **Restrict CORS.** `ALLOWED_ORIGINS` currently defaults to `*`. Set it to the
   deployed frontend origin.
2. **Host the model weights** (see above). Without this the API starts but
   `/api/evaluate` returns 503.
3. **Add rate limiting.** There is none. Inference is GPU-bound and a public
   endpoint is trivially exhausted — put a limiter or an API gateway in front.
4. **Set a request size ceiling.** `text` is capped at 50,000 characters in the
   schema; enforce a body-size limit at the proxy too.
5. **Decide on logging.** The service currently logs nothing about submitted
   text. If any logging is added, note that users may paste sensitive or harmful
   content, so retention needs a deliberate decision.
6. **Serve over HTTPS** and add the usual security headers at the proxy.
7. **Pin the frozen outputs** into the deployment, or the Results and
   Methodology sections return 503.

Not required, but worth considering: a health-check path for the platform's
probes (`/api/health` already serves this), and a CPU fallback note in the UI
since inference takes noticeably longer without CUDA.

---

## What this application deliberately does not do

- It does **not** generate model responses, and does not send prompts to any
  LLM. It only classifies text that is pasted in.
- It does **not** modify research data. Every path it touches is read-only.
- It does **not** present the classifier as reliable. It is a research
  artifact evaluated on a small, targeted human-audit set (not a representative
  sample and not a production safety control), and the UI states this beside
  every prediction rather than in a footnote.
