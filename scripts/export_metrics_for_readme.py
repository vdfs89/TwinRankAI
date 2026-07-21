import json
from pathlib import Path


def main():
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        print(f"File {metrics_path} not found. Please run 'dvc repro evaluate' first.")
        return

    with open(metrics_path, encoding="utf-8") as f:
        metrics_data = json.load(f)

    # Required columns
    columns = ["Recall@10", "MAP@10", "MRR@10", "NDCG@10"]
    # Model display names
    models_display = {
        "popularity": "Baseline de Popularidade",
        "matrix_factorization": "Matrix Factorization / Baseline",
        "two_tower": "Modelo Neural Two-Tower",
    }

    print("| Modelo | " + " | ".join(columns) + " |")
    print("|" + "-" * 33 + "|" + "|".join(["-" * 11 for _ in columns]) + "|")

    for model_key in ["popularity", "matrix_factorization", "two_tower"]:
        if model_key in metrics_data:
            m = metrics_data[model_key]
            name = models_display.get(model_key, model_key)
            recall = f"{m.get('recall_at_10', 0):.3f}"
            map_score = f"{m.get('map_at_10', 0):.3f}"
            mrr = f"{m.get('mrr_at_10', 0):.3f}"
            ndcg = f"{m.get('ndcg_at_10', 0):.3f}"

            # Bold for two_tower as in README
            if model_key == "two_tower":
                recall = f"**{recall}**"
                map_score = f"**{map_score}**"
                mrr = f"**{mrr}**"
                ndcg = f"**{ndcg}**"

            print(
                f"| {name.ljust(31)} | {recall.ljust(9)} | {map_score.ljust(9)} | {mrr.ljust(9)} | {ndcg.ljust(9)} |"
            )


if __name__ == "__main__":
    main()
