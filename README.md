# EV Range Lab
An academic full-stack machine learning prototype for electric scooter range and state of health (SOH) prediction. No real telemetry, manufacturer-specific models, maps, or charging station data are included. Do not rely on these predictions for actual travel or battery diagnostics.

## Run locally
Python 3.13 is recommended (the trained model was serialized with scikit-learn 1.8.0).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```
Open http://127.0.0.1:8080 for the website and /docs for interactive API documentation. Model artifacts are already included.

## Retrain and test
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python train.py
pip install pytest httpx
python -m pytest -q
```
The deterministic generator makes 12,000 synthetic examples using assumed battery degradation and energy consumption equations plus noise. The dataset has 11 independent inputs and two targets (range in km and SOH in %). Train/validation/calibration/test counts are 7200/2400/1200/1200. Three models are compared by validation MAE; the winner is refit on train+validation. Calibration residuals determine a nominal 90% marginal prediction interval, then the held-out synthetic test set measures performance. Results are not evidence of real-world generalization. Real measured battery capacity, range, environmental and usage records are necessary for external validation. The 20% margin applied to the lower range bound is an illustrative heuristic, not a safety guarantee. Zero charge returns zero range; below 5% is extrapolated. Inputs are constrained to the simulation domain otherwise.

## Deploy to Google Cloud Run
1. Open https://console.cloud.google.com/ and create or select a project. Enable billing; usage may incur costs. Note its Project ID (not display name).
2. Open Cloud Shell at https://shell.cloud.google.com/ and upload `ev-range-lab.zip` with the Cloud Shell upload menu. Run:
```bash
unzip ev-range-lab.zip
cd ev-range-lab
bash deploy.sh YOUR_GOOGLE_CLOUD_PROJECT_ID
```
The script enables APIs, builds the included Dockerfile with Cloud Build, stores the image in Artifact Registry, and deploys a public Cloud Run service in Mumbai (`asia-south1`). Minimum instances 0, maximum 2, 512 MiB memory; this limits scaling, not all costs. It prints the HTTPS service URL and checks `/health`. If you prefer private access replace `--allow-unauthenticated` with `--no-allow-unauthenticated` and use an authenticated request for the health check.

If you encounter IAM or billing errors, see Google's current FastAPI guide: https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-fastapi-service and source deployment permissions: https://docs.cloud.google.com/run/docs/deploying-source-code. Your deployer and Cloud Build service account need the documented roles; an administrator may need to grant these. No credentials are included. If your organization blocks public access, contact its administrator rather than bypassing policy.

To remove the service when finished:
```bash
gcloud run services delete ev-range-lab --region asia-south1 --project YOUR_GOOGLE_CLOUD_PROJECT_ID
```
Cloud Build and Artifact Registry resources may remain and incur charges; review your project billing and remove unused resources separately. Set a billing budget and alerts; alerts do not cap spend.

## API
- GET `/` — browser UI
- GET `/health` — model readiness
- GET `/api/metrics` — actual benchmark metrics
- POST `/api/predict` — validated JSON inference (see /docs for schema)

Example:
```bash
curl http://localhost:8080/api/predict -H 'Content-Type: application/json' -d '{"capacity_kwh":3.7,"soc_pct":75,"age_years":2,"cycles":450,"fast_charge_pct":20,"avg_battery_temp_c":30,"ambient_temp_c":28,"speed_kmh":35,"payload_kg":80,"traffic":1,"eco_mode":1,"trip_km":30}'
```
Architecture: browser -> FastAPI -> serialized Gradient Boosting models -> JSON -> browser. Cloud Run hosts the same service; no database or paid third-party API is needed. No user data is persisted by the app. Public deployment does not include application-level authentication or rate limiting.

## Verification status
Model training completed and all eight API tests passed in the build environment. Cloud deployment and Docker build were not executed because authenticated Google Cloud access and Docker were not available. Browser interaction was not exercised in a real browser. The application is a deployment-ready academic prototype, not a validated production EV system.
