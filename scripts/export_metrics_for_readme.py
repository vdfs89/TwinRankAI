import json  # noqa: D100
from pathlib import Path

# 5 casas: com 3, Popularity colapsa (0,00342 e 0,00310 viram ambos "0.003")
# e a distância para os demais modelos deixa de ser legível.
DECIMALS = 5

COLUMNS = [
    ("Recall@10", "recall_at_10"),
    ("Precision@10", "precision_at_10"),
    ("MAP@10", "map_at_10"),
    ("MRR@10", "mrr_at_10"),
    ("NDCG@10", "ndcg_at_10"),
]

NOVEL_COLUMNS = [
    ("Recall@10 (geral)", "recall_at_10"),
    ("Recall@10 (novel)", "recall_at_10_novel"),
]

MODELS_DISPLAY = {
    "popularity": "Baseline de Popularidade",
    "matrix_factorization": "Matrix Factorization / Baseline",
    "two_tower": "Modelo Neural Two-Tower",
}


def _fmt(value: float, bold: bool) -> str:
    text = f"{value:.{DECIMALS}f}".replace(".", ",")
    return f"**{text}**" if bold else text


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:  # noqa: D103
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        print(f"File {metrics_path} not found. Please run 'dvc repro evaluate' first.")
        return

    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    keys = [k for k in MODELS_DISPLAY if k in metrics_data]

    rows = [
        [MODELS_DISPLAY[k]]
        + [_fmt(metrics_data[k][field], k == "two_tower") for _, field in COLUMNS]
        for k in keys
    ]
    _print_table(["Modelo"] + [label for label, _ in COLUMNS], rows)

    if not all("recall_at_10_novel" in metrics_data[k] for k in keys):
        return

    print()
    novel_rows = []
    for k in keys:
        general = metrics_data[k]["recall_at_10"]
        novel = metrics_data[k]["recall_at_10_novel"]
        repetition = f"{(1 - novel / general) * 100:.1f}%".replace(".", ",")
        novel_rows.append(
            [
                MODELS_DISPLAY[k],
                _fmt(general, False),
                _fmt(novel, False),
                repetition,
            ]
        )
    _print_table(
        ["Modelo"] + [label for label, _ in NOVEL_COLUMNS] + ["Fração do recall que é repetição"],
        novel_rows,
    )


if __name__ == "__main__":
    main()
