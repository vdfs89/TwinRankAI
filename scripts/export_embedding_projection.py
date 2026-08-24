"""Exporta uma projeção 2D dos embeddings de item para o app Streamlit.

O app publicado não tem acesso a `models/` (versionado por DVC, fora do Git),
então a página de embeddings não pode carregar o checkpoint em runtime. Este
script pré-computa a projeção a partir do modelo treinado de verdade e grava um
CSV leve, versionado — o mesmo padrão usado em `reports/metrics.json`.

A projeção usa PCA, que é determinística e já vem com o scikit-learn. Os grupos
exibidos são clusters descobertos por KMeans sobre os embeddings, não categorias
de catálogo: o RetailRocket é anonimizado e não expõe rótulos de categoria
utilizáveis aqui.

Uso:
    poetry run python scripts/export_embedding_projection.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from reco.settings import Settings

SAMPLE_SIZE = 2000
N_CLUSTERS = 6
OUTPUT = "reports/embedding_projection.csv"


def main() -> None:  # noqa: D103
    settings = Settings()
    checkpoint = torch.load(str(settings.model_path), map_location="cpu", weights_only=False)

    item_index: dict = checkpoint["item_index"]
    weights = checkpoint["state_dict"]["item_tower.weight"].numpy()

    rng = np.random.default_rng(settings.random_seed)
    n_items = weights.shape[0]
    size = min(SAMPLE_SIZE, n_items)
    sampled = rng.choice(n_items, size=size, replace=False)

    index_to_item = {idx: item for item, idx in item_index.items()}
    vectors = weights[sampled]

    coords = PCA(n_components=2, random_state=settings.random_seed).fit_transform(vectors)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=settings.random_seed, n_init=10)
    clusters = kmeans.fit_predict(vectors)

    frame = pd.DataFrame(
        {
            "item_id": [index_to_item.get(int(i), int(i)) for i in sampled],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": [f"Grupo {c + 1}" for c in clusters],
        }
    )
    frame.to_csv(OUTPUT, index=False)

    print(f"itens no catalogo do modelo: {n_items}")
    print(f"amostrados: {size} | dimensao original: {weights.shape[1]}")
    print(f"gravado em: {OUTPUT}")


if __name__ == "__main__":
    main()
