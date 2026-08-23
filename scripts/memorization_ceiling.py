"""Teto de memorização do Recall@10 na população filtrada de avaliação.

Responde: "qual seria o Recall@10 de um oráculo que apenas devolvesse os
próprios itens de treino do usuário?". Serve de referência superior para o
componente de memorização do modelo — o Two-Tower não deveria ultrapassá-lo
por repetição, e a distância até ele indica quanto do comportamento
memorizável a rede de fato capturou.

Roda sobre a MESMA população usada nas métricas oficiais (visitantes com pelo
menos `MIN_TRAIN_INTERACTIONS` interações de treino), para que o número seja
comparável ao `recall_at_10` de `reports/metrics.json`. Não carrega nem treina
modelo algum.

Uso:
    python scripts/memorization_ceiling.py
"""

import json
from pathlib import Path

import pandas as pd

from reco.settings import Settings
from reco.training.evaluate import (
    build_relevance_lookup,
    filter_lookup_by_history,
    recall_at_k,
)


def main() -> None:  # noqa: D103
    settings = Settings()
    k = settings.top_k
    processed = settings.processed_data_dir

    train_events = pd.read_csv(processed / "train_features.csv", parse_dates=["timestamp"])
    test_events = pd.read_csv(processed / "test_features.csv", parse_dates=["timestamp"])

    lookup, audit = filter_lookup_by_history(build_relevance_lookup(test_events), train_events)

    # O oráculo escolhe os k itens de maior relevância do histórico do próprio
    # usuário — o melhor que uma estratégia de pura memorização poderia fazer.
    train_ranked = (
        train_events.groupby(["visitorid", "itemid"])["relevance"]
        .max()
        .reset_index()
        .sort_values(["visitorid", "relevance"], ascending=[True, False])
    )
    history = train_ranked.groupby("visitorid")["itemid"].apply(list).to_dict()

    recalls = [
        recall_at_k(
            [int(item) for item in history.get(visitor_id, [])[:k]],
            set(relevance_map.keys()),
            k,
        )
        for visitor_id, relevance_map in lookup.items()
    ]
    ceiling = sum(recalls) / len(recalls)

    metrics_path = Path("reports/metrics.json")
    actual = None
    if metrics_path.exists():
        actual = json.loads(metrics_path.read_text(encoding="utf-8"))["two_tower"][f"recall_at_{k}"]

    print(f"populacao avaliada: {audit['eval_visitors_after_filter']} visitantes")
    print(f"teto de memorizacao recall@{k}: {ceiling:.5f}")
    if actual is not None:
        print(f"two-tower real recall@{k}:      {actual:.5f}")
        print(f"fracao do teto atingida:        {actual / ceiling * 100:.1f}%")


if __name__ == "__main__":
    main()
