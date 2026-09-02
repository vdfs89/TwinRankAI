# TwinRank AI

[🇧🇷 Português](README.md) · [🇬🇧 English](README.en.md)

**Deep Learning Recommendation Engine**

**Vitor Diogo Fonseca da Silva — RM375157**
Tech Challenge · Phase 2 · Machine Learning Engineering · FIAP

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white)](https://dvc.org/)
[![Ruff](https://img.shields.io/badge/Ruff-000000?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)

> **Learning Intent. Ranking Experiences.**
> Every interaction tells a story. TwinRank AI learns from it.

TwinRank AI is a production‑oriented recommendation engine for e‑commerce. It learns user intent from implicit signals such as clicks, views, cart events and purchases, and projects users and items into a shared embedding space using a **Two‑Tower** neural architecture.

The project combines **Deep Learning, Machine Learning Engineering and MLOps** into a reproducible pipeline for experimentation and deployment. Instead of relying only on popularity or static rules, it learns behavioral patterns with traceability, versioning and operational rigor.

> In our reference run, the **Two-Tower model reached Recall@10 = 0.123 and NDCG@10 = 0.101**, against 0.00342 and 0.00408 for the popularity baseline. Part of that gain comes from re-recommending items the user had already seen; controlling for pure discovery, the Two-Tower still leads. See [Metrics](#metrics) for the full numbers and methodological caveats.

---

## Quick links

- [Architecture](docs/architecture.md)
- [Model Card](docs/model_card.md)
- [Kubernetes deployment](k8s/README.md)
- [Dataset](#dataset)
- [Local Setup](#local-setup)
- [Production Demo](#production-demo)
- [Metrics](#metrics)
- [Roadmap](#roadmap)

---

## Product vision

Imagine an online store with millions of products.

A few page views, one item added to cart, another removed, and a user coming back days later already carry a trail of intent. Without saying a single word, users reveal interests, curiosity and decisions.

**TwinRank AI** was built to interpret these signals. It turns raw events into structured representations that power personalized recommendations at scale, following modern **ML Engineering** and **MLOps** practices.

More than a single model, TwinRank AI is a **compact blueprint for a recommendation platform**: clean code, reproducible pipelines, experiment tracking and lifecycle management close to what real‑world ML systems require.

---

## Why TwinRank AI

- Learns from behavioral signals instead of relying solely on popularity.
- Uses neural embeddings for personalized ranking.
- Supports scalable *retrieval* and *re‑ranking* flows in a shared latent space.
- Organizes data, experiments and model lifecycle with reproducibility in mind.
- Aligns software engineering and MLOps to the development of recommender systems.

---

## Core architecture

TwinRank AI follows a **Two‑Tower + re‑ranking** recommendation architecture:

- **User Tower**
  - Learns user embeddings from interaction history and contextual behavioral signals.

- **Item Tower**
  - Learns item embeddings from product identity and optional metadata.

- **Retrieval**
  - Generates candidate items by computing similarity (dot product or equivalent score function) between user and item embeddings.

- **Re‑ranking**
  - Applies additional signals (recency, diversity, business rules) to order candidates before presenting them to the user.

This separation mirrors large‑scale recommendation systems: a retrieval stage narrows down a huge catalog, followed by scoring and re‑ranking to produce the final ranked list.

---

## Dataset

TwinRank AI uses the **RetailRocket E‑commerce Dataset** as a realistic source of user–item interactions:

- `events.csv`
- `item_properties.csv`
- `category_tree.csv`

Suggested download:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
```

---

## Repository structure

```text
TwinRank-AI/
├── src/reco/        # source code (data, models, pipelines, serving, training, utils)
├── tests/           # unit and integration tests
├── scripts/         # utility scripts (train, eval, serve)
├── configs/         # experiment configs
├── data/            # raw and processed data (tracked with DVC)
├── models/          # saved model artifacts
├── docs/            # architecture, model card, etc.
├── k8s/             # Kubernetes manifests for the serving API
├── dvc.yaml
├── pyproject.toml
├── docker-compose.yml
└── Dockerfile
```

Responsibilities are separated across data processing, feature generation, training, evaluation, serving and infrastructure. This layout favors clean code, testability and a reproducible flow from raw events to recommendation endpoints.

---

## Expected pipeline

TwinRank AI is designed as a reproducible ML pipeline with explicit data and experiment lineage:

1. Preprocess raw interaction logs and build user–item events.
2. Engineer features and build indexed representations for users and items.
3. Generate training pairs with negative sampling.
4. Train the neural Two‑Tower model in PyTorch.
5. Evaluate ranking quality using recommendation metrics.
6. Track runs, metrics and artifacts in MLflow.
7. Register the best model version and promote it through the lifecycle.
8. Serve recommendations through an API layer.

This mirrors multi‑stage pipelines in real recommendation systems, where reproducibility, observability and controlled promotion matter as much as offline metrics.

---

## Tech stack

| Layer                    | Tools                         |
|--------------------------|-------------------------------|
| Deep Learning            | PyTorch                       |
| Baselines / Preprocessing| Scikit‑Learn                  |
| API                      | FastAPI                       |
| Experiment Tracking      | MLflow                        |
| Data & Pipeline Versioning| DVC                          |
| Containerization         | Docker, Docker Compose        |
| Orchestration            | Kubernetes (manifests in `k8s/`) |
| Dependency Management    | Poetry                        |
| Quality                  | Pytest, Ruff, pre‑commit      |
| CI/CD                    | GitHub Actions                |

---

## Local setup

```bash
make install
make validate
make lint
make test
make mlflow-ui
```

Run the API locally:

```bash
python -m uvicorn reco.serving.api:app --reload
```

### API routes

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/model/version` | GET | Path, registered name, stage and `model_loaded` for the served model |
| `/recommend/{user_id}` | GET | Top-k recommendations (`top_k` between 1 and 100) |
| `/predict` | POST | Top-k restricted to a candidate list |
| `/train` | POST | Kicks off the training pipeline in the background (202) |
| `/preprocess` | POST | Runs preprocessing |
| `/feature-eng` | POST | Runs feature engineering |

The model is loaded at application startup (~5 s), not on the first request. Every response carries an `X-Response-Time-ms` header, and latency is recorded in the structured log.

`/recommend` and `/predict` return a `strategy` field indicating how the response was produced: `two_tower` (personalized), `popularity_fallback` (visitor missing from the index — cold start, falls back to the global popularity ranking) or `unavailable` (no model loaded).

A constant `strategy: "unavailable"` means the checkpoint is missing, not that the model failed. The API boots without a checkpoint on purpose, so training can be triggered through `POST /train`, and `/health` still answers `ok` in that state. Confirm through the `model_loaded` field of `/model/version` and the `checkpoint_indisponivel` warning in the startup log.

> **Known limitation — `POST /train`.** The route returns `202 training_started` immediately and runs the pipeline through FastAPI's `BackgroundTasks`, because real training takes ~12.5 min and would block the connection until timeout. `BackgroundTasks` runs in the server process: it does not survive a restart, has no retries, no progress visibility, and competes for CPU with serving. **It is not production-ready without a dedicated job queue** (Celery, RQ, Arq or equivalent). Track progress in MLflow.

### Docker

```bash
docker compose up -d
```

Starts `app` (FastAPI on 8000), `mlflow` (5000), `redis` (6379) and
`streamlit` (8501).

> **The training service does NOT start with `docker compose up`.** It sits
> behind the `training` profile on purpose: its command runs the full pipeline
> (~18 min) and would overwrite `models/` through the mounted volume,
> destroying the checkpoint this documentation references. To train inside
> Docker, invoke it explicitly:
>
> ```bash
> docker compose --profile training run --rm train
> ```

The image is multi-stage and installs the CPU wheel of PyTorch, declared as an
explicit source in `pyproject.toml`. The default PyPI wheel bundles the CUDA
libraries and pushed the image to 9.02 GB; with the CPU wheel it sits at
2.43 GB, with no GPU involved in serving.

### Kubernetes

The manifests under [`k8s/`](k8s/) run the same API on a cluster, with Redis,
probes, CPU requests and an HPA scaling from 2 to 6 replicas. They were applied
and verified on a kind cluster (Kubernetes v1.34). The walkthrough, including
how to load the checkpoint into the volume, lives in
[`k8s/README.md`](k8s/README.md).

### Reproducing the project from scratch

Full flow, validated on a clean clone on a separate machine:

```bash
git clone https://github.com/vdfs89/TwinRankAI.git
cd TwinRankAI
poetry install --only main
dvc pull
poetry run dvc repro
```

**Reference time:** the 4 stages take **~18 minutes** (measured: 1082 s),
dominated by training. `dvc pull` fetches 942 MB in a few seconds.

> **Use `poetry run` (or activate the environment first).** Every `dvc.yaml`
> stage runs `python -m reco.pipelines...`; without the project environment
> active that `python` is the system one, and the pipeline fails within
> seconds with `ModuleNotFoundError: No module named 'reco'`.

> **No MLflow server required.** The default is the local file store
> (`MLFLOW_TRACKING_URI=file:./mlruns`), so the pipeline runs offline. To
> inspect runs afterwards: `make mlflow-ui`. A remote/shared server is
> optional — just override `MLFLOW_TRACKING_URI` in `.env`.

---

## Production Demo

TwinRank AI includes a complete serving stack with sub-millisecond retrieval using **FAISS (ANN)**, low-latency caching with **Redis**, and an interactive visualization dashboard built with **Streamlit**.

Check out the [Production Demo Walkthrough](docs/production_demo.md) to see how to spin up the entire ecosystem locally with a single `docker-compose` command, generate real-time recommendations, and observe Cache Hits/Misses in action.

---

## Pluggable E-Commerce Demo

If you own a small e-commerce or want to see TwinRank working on your own data instantly, we built a pluggable demo that trains the Two-Tower neural network **on-the-fly**.

Just provide two CSV files:
- `products.csv`: (product_id, name, category, price)
- `orders.csv`: (order_id, user_id, product_id, timestamp)

Run the standalone dashboard:
```bash
poetry run streamlit run src/reco/frontend/app.py
```

Upload your CSVs (or use the built-in dummy data), and the app will train a custom TwinRank model + FAISS index in memory in just a few seconds, unlocking state-of-the-art recommendations for your specific catalog.

---

## Metrics

TwinRank AI focuses on **ranking metrics** rather than plain classification accuracy. For recommender systems, metrics such as **Recall@K**, **MAP@K**, **MRR@K** and **NDCG@K** provide a more useful view of how well the model surfaces relevant items in top positions.

| Model                           | Recall@10 | Precision@10 | MAP@10 | MRR@10 | NDCG@10 |
|---------------------------------|-----------|--------------|--------|--------|---------|
| Popularity baseline             | 0.00342   | 0.00147      | 0.00310 | 0.00965 | 0.00408 |
| Matrix Factorization baseline   | 0.02265   | 0.00870      | 0.01386 | 0.03086 | 0.01958 |
| Neural Two‑Tower model          | **0.12311** | **0.03205** | **0.07604** | **0.13286** | **0.10078** |

The Two-Tower reaches 36.0x the popularity baseline's Recall@10, 24.7x its NDCG@10, and 5.4x the Matrix Factorization's Recall@10.

> **Evaluation population changed.** These numbers are not comparable with earlier versions of this README: evaluation is now restricted to visitors with at least 5 interactions in the training set (`eval_min_train_interactions=5`), which reduced the evaluated population from 23,476 to 2,920 users. The filter was necessary because 57% of the original visitors had a single training interaction and dominated the average, making the previous metric statistically meaningless. An apparent drop in the baseline between versions reflects this population change, not a model regression.

> **Repetition vs. discovery.** A substantial share of the Two-Tower's Recall@10 comes from recommending items the user had already seen in training, not novel ones. `Recall@10 (novel)` measures discovery alone: it excludes from each user's relevant set everything already present in their training history.
>
> | Model | Recall@10 (overall) | Recall@10 (novel) | Share that is repetition |
> |---|---|---|---|
> | Popularity baseline | 0.00342 | 0.00202 | 40.8% |
> | Matrix Factorization | 0.02265 | 0.00359 | 84.1% |
> | Two-Tower | 0.12311 | 0.01168 | 90.5% |
>
> In other words: only ~9.5% of the Two-Tower's Recall@10 are hits on items never interacted with before. The repetition rate rises with personalization — Popularity does not personalize, so it hits novel items at a relatively higher rate. This is expected in pure ID-embedding models, where the only possible generalization is via learned co-occurrence, and is not an implementation bug. Even controlling for pure discovery the Two-Tower still leads, at 3.3x the Matrix Factorization's `Recall@10 (novel)` and 5.8x Popularity's — but that margin is narrower than the 5.4x of the overall Recall@10 suggests. A fair comparison cites both metrics, never the overall Recall@10 alone. See the [Model Card](docs/model_card.md) for the full analysis.

> **Memorization ceiling.** Since much of Recall@10 comes from repetition, it
> helps to know how much was available: an oracle returning only the
> highest-relevance items from each user's own training history reaches
> Recall@10 = 0.16158 on the same 2,920-visitor population. The Two-Tower
> reaches **76.2%** of that ceiling (0.12311 / 0.16158). Reproducible with
> `poetry run python scripts/memorization_ceiling.py`.

*Results from the reference run tracked in MLflow under run
`6e55cf97abb34251b862369e7c725770`, registered as `twinrank-ai-two-tower`
version 3 (Production stage), produced by `dvc repro` into
`reports/metrics.json`. Version 2 comes from run
`a9e7c00368df4b93acc655a6493e9b69` and is archived; both produced identical
numbers, and that match is what serves as determinism evidence. See the
[Model Card](docs/model_card.md) for details.*

> **TODO — manual sync.** The tables above are generated by `python scripts/export_metrics_for_readme.py`, which only prints markdown to stdout. After any `dvc repro` that changes `reports/metrics.json`, the output must be pasted here and into the [Model Card](docs/model_card.md) by hand, otherwise the numbers drift from the pipeline silently. The Streamlit metrics page already reads the JSON directly and needs no such step.

---

## Mission, vision and values

**Mission**

> Democratize modern recommendation systems through a reproducible, scalable, deep‑learning‑oriented architecture that turns behavioral data into high‑quality personalized experiences.

**Vision**

> Become an open reference for recommendation engineering, showing how Deep Learning, MLOps and software engineering can converge into systems close to those used by large e‑commerce platforms.

**Values**

- Data‑driven intelligence
- Production‑grade engineering
- Reproducibility
- Continuous learning
- Transparency and traceability
- Scalability
- Clean and collaborative code

---

## Manifesto

Every click represents intent.
Every abandoned cart tells part of a story.
Every purchase confirms a need.

In digital commerce, users rarely state explicitly what they want; they reveal it through behavior.

**TwinRank AI** was created to interpret these hidden signals and to continuously learn how to connect people with the most relevant products. More than a recommendation algorithm, it sits at the intersection of **Deep Learning, software engineering and MLOps** to build intelligent, scalable and reproducible systems.

Recommending products is not just about predicting the next click.
It is about understanding the intent behind every interaction.

---

## Roadmap

- [x] Product positioning and repository narrative
- [x] Architecture and Model Card documentation
- [x] Data preprocessing pipeline
- [x] Feature engineering for RetailRocket interactions
- [x] Negative sampling strategy
- [x] Popularity baseline
- [x] Matrix Factorization / classical baseline
- [x] Two‑Tower neural recommender
- [x] Experiment tracking with MLflow
- [x] Reproducible pipeline with DVC
- [x] Multi‑stage Docker environment
- [x] Model Registry promotion flow
- [x] Recommendation service with FastAPI
- [x] Kubernetes manifests for the API, verified on a local cluster
- [ ] API deployed behind a public URL
- [x] GitHub Actions CI
- [x] FAISS retrieval layer
- [x] Redis recommendation cache
- [x] Streamlit dashboard

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
