# Production Demo Walkthrough

TwinRank AI includes a complete serving ecosystem to demonstrate its capabilities in a production-like environment. The stack features:

- **FastAPI**: Backend service exposing the recommendation endpoints.
- **FAISS**: Approximate Nearest Neighbor (ANN) search for lightning-fast item retrieval.
- **Redis**: Cache-Aside pattern to reduce model load and response latency.
- **Streamlit**: An interactive dashboard covering metrics, embeddings and the pluggable demo.

> **The dashboard does not call the API.** The Streamlit pages read
> `reports/metrics.json` and, on the pluggable demo page, train a small model
> in-process from uploaded CSVs. To exercise the serving path — FAISS retrieval,
> the Redis cache and the popularity fallback — talk to the API directly, as
> described below.

Follow these steps to run the demo locally.

## 1. Start the Environment

Make sure you have Docker and Docker Compose installed. From the root of the repository, start the stack:

```bash
docker compose up --build app redis streamlit
```

This command will build the images and spin up the FastAPI backend, the Redis cache, and the Streamlit frontend.

## 2. Access the Interfaces

Once the containers are running, open:

- [http://localhost:8501](http://localhost:8501) — the TwinRank AI dashboard.
- [http://localhost:8000/docs](http://localhost:8000/docs) — the interactive API documentation.

## 3. Generate Recommendations

Call the API with any visitor id from the training population:

```bash
curl "http://localhost:8000/recommend/172?top_k=5"
```

The response carries the recommended item ids plus a `strategy` field:

```json
{"user_id":172,"item_ids":[465522,48030,10034,242905,292240],"strategy":"two_tower"}
```

`strategy` tells you which path produced the answer. A visitor missing from the
training index falls back to the global popularity ranking:

```bash
curl "http://localhost:8000/recommend/1?top_k=5"
```

```json
{"user_id":1,"item_ids":[461686,5411,257040,187946,309778],"strategy":"popularity_fallback"}
```

If every call returns `"strategy":"unavailable"`, the checkpoint is missing
rather than broken — check `model_loaded` in `http://localhost:8000/model/version`.

## 4. Verify the Cache (Hit/Miss)

To observe the Cache-Aside pattern in action, repeat the same request and read the logs with `docker compose logs app`:

1. **First request:** the backend queries the FAISS index and logs a `cache_miss`. The result is then stored in Redis.
2. **Subsequent requests:** the same user id and `top_k` are served straight from Redis, logging a `cache_hit`.

```text
level=INFO logger=reco.serving.service msg=cache_miss user_id=172 top_k=5
level=INFO logger=reco.serving.service msg=cache_hit user_id=172 top_k=5
```

Every response also carries an `X-Response-Time-ms` header with the measured latency.

This demonstrates how TwinRank AI is architected not just for offline metrics, but for real-world, low-latency performance.

## 5. Running on Kubernetes

The same API runs on a cluster through the manifests in [`k8s/`](../k8s/), with
Redis, health probes and an HPA scaling from 2 to 6 replicas. See
[`k8s/README.md`](../k8s/README.md) for the walkthrough.
