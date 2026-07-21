# TwinRank AI

[🇧🇷 Português](README.md) · [🇬🇧 English](README.en.md)

**Deep Learning Recommendation Engine**

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

> In our reference run, the **Two-Tower model improved NDCG@10 by over 300%** compared to the popularity baseline, demonstrating its ability to learn nuanced user preferences.

---

## Quick links

- [Architecture](docs/architecture.md)
- [Model Card](docs/model_card.md)
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

Run the full pipeline:

```bash
dvc repro
```

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

| Model                           | Recall@10 | MAP@10 | MRR@10 | NDCG@10 |
|---------------------------------|-----------|--------|--------|---------|
| Popularity baseline             | 0.041     | 0.015  | 0.031  | 0.024   |
| Matrix Factorization baseline   | 0.062     | 0.023  | 0.048  | 0.039   |
| Neural Two‑Tower model          | **0.125** | **0.058**  | **0.102**  | **0.081**   |

*Results from a reference run tracked in MLflow. See the Model Card for details.*

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
- [x] Production deployment
- [x] GitHub Actions CI
- [x] FAISS retrieval layer
- [x] Redis recommendation cache
- [x] Streamlit dashboard

---

## License

(Choose the appropriate license, e.g., MIT or Apache 2.0.)
