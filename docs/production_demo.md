# Production Demo Walkthrough

TwinRank AI includes a complete serving ecosystem to demonstrate its capabilities in a production-like environment. The stack features:

- **FastAPI**: Backend service exposing the recommendation endpoint.
- **FAISS**: Approximate Nearest Neighbor (ANN) search for lightning-fast item retrieval.
- **Redis**: Cache-Aside pattern to reduce model load and response latency.
- **Streamlit**: An interactive dashboard to visualize recommendations.

Follow these steps to run the demo locally.

## 1. Start the Environment

Make sure you have Docker and Docker Compose installed. From the root of the repository, start the stack:

```bash
docker-compose up --build app redis streamlit
```

This command will build the images and spin up the FastAPI backend, the Redis cache, and the Streamlit frontend.

## 2. Access the Dashboard

Once the containers are running, open your web browser and navigate to:

[http://localhost:8501](http://localhost:8501)

You will see the TwinRank AI control panel.

## 3. Generate Recommendations

1. In the sidebar, type a **User ID** (e.g., `12345`).
2. Adjust the **Top K** slider to choose how many recommendations you want.
3. Click **Generate recommendations**.

The dashboard will display the recommended items in an interactive grid. Behind the scenes, the FastAPI backend routes the request to our pre-trained Two-Tower model, using the FAISS index to rapidly retrieve the best matching items.

## 4. Verify the Cache (Hit/Miss)

To observe the Cache-Aside pattern in action, check your terminal logs:

1. **First request:** The backend queries the FAISS index. You will see a log entry indicating a `cache_miss`. The results are then stored in Redis.
2. **Subsequent requests:** If you click "Generate recommendations" again for the same User ID and Top K, the backend will retrieve the results instantly from Redis. You will see a `cache_hit` in the terminal logs.

```text
[INFO] reco.serving.service: cache_miss user_id=12345 top_k=10
[INFO] reco.serving.service: cache_hit user_id=12345 top_k=10
```

This demonstrates how TwinRank AI is architected not just for offline metrics, but for real-world, low-latency performance.
