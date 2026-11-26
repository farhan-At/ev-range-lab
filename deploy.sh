#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ID="${1:?Usage: bash deploy.sh YOUR_GOOGLE_CLOUD_PROJECT_ID}"
REGION="${REGION:-asia-south1}"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$PROJECT_ID"
gcloud run deploy ev-range-lab --project "$PROJECT_ID" --region "$REGION" --source . --allow-unauthenticated --cpu 1 --memory 512Mi --min-instances 0 --max-instances 2 --concurrency 8 --port 8080
URL=$(gcloud run services describe ev-range-lab --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')
echo "Service URL: $URL"
curl --fail --retry 3 --retry-delay 2 "$URL/health"; echo
