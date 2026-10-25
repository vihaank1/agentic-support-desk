# Deploying to Google Cloud Run (not yet done)

This service has **not** been deployed by the author of this repo yet. The steps below are the standard
Cloud Run source-deploy flow; run them yourself and record what you observe before claiming a deployment.

```bash
# one-time
gcloud auth login && gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# store the key as a secret (do not put it in the image or repo)
printf '%s' "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-

gcloud run deploy support-desk --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars LLM_PROVIDER=gemini,MAX_USD_PER_PROCESS=1.00,PRICE_IN_PER_1K=<real>,PRICE_OUT_PER_1K=<real> \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

Smoke test:

```bash
curl -s "$URL/healthz"
curl -s -X POST "$URL/ask" -H 'content-type: application/json' \
  -d '{"question":"How much is the overage charge if I use 8 GB on the Basic plan?"}'
```

Notes: `--allow-unauthenticated` makes the endpoint public, so keep `MAX_USD_PER_PROCESS` set and consider removing public
access when you are done. Each instance enforces its own spend guard; it is not a global budget (set a billing budget alert too).

Run the real-model evaluation locally before deploying:

```bash
GEMINI_API_KEY=... PYTHONPATH=src python scripts/run_eval.py --llm gemini --max-usd 0.50
GEMINI_API_KEY=... PYTHONPATH=src python scripts/run_eval.py --llm gemini --cases data/eval_cases_independent.json --max-usd 0.50
```
