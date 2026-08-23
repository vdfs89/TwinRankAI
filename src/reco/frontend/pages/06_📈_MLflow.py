import os  # noqa: D100
import sys

import pandas as pd
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from reco.frontend.utils import (
    MODEL_LABELS,
    inject_custom_css,
    load_metrics,
    metrics_missing_warning,
    model_keys,
)

st.set_page_config(page_title="MLOps & MLflow - TwinRank AI", page_icon="📈", layout="wide")
inject_custom_css()

st.title("📈 MLOps & Tracking")
st.markdown(
    "Monitoramento de experimentos, versionamento de hiperparâmetros e registro de modelos utilizando **MLflow**."  # noqa: E501
)

st.write(
    "O TwinRank AI foi construído com as melhores práticas de MLOps. Todas as execuções de treinamento "  # noqa: E501
    "são rastreadas. Registramos automaticamente as métricas de validação, os hiperparâmetros (como `learning_rate`, "  # noqa: E501
    "`batch_size`, `embedding_dim`) e persistimos os pesos do modelo (Artifacts)."
)

st.markdown("### Resultados dos Modelos Avaliados")

metrics_data = load_metrics()

if metrics_data is None:
    metrics_missing_warning()
else:
    meta = metrics_data.get("_meta", {})
    top_k = meta.get("top_k", 10)

    rows = [
        {
            "Modelo": MODEL_LABELS[key],
            "Embed Dim": meta.get("embedding_dim", "—") if key == "two_tower" else "—",
            f"Recall@{top_k}": metrics_data[key][f"recall_at_{top_k}"],
            f"NDCG@{top_k}": metrics_data[key][f"ndcg_at_{top_k}"],
            f"MAP@{top_k}": metrics_data[key][f"map_at_{top_k}"],
            f"Recall@{top_k} novel": metrics_data[key][f"recall_at_{top_k}_novel"],
        }
        for key in model_keys(metrics_data)
    ]

    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                f"Recall@{top_k}": "{:.5f}",
                f"NDCG@{top_k}": "{:.5f}",
                f"MAP@{top_k}": "{:.5f}",
                f"Recall@{top_k} novel": "{:.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    run_id = meta.get("run_id")
    st.caption(
        f"Run registrado em produção: `{run_id}` · "
        f"avaliação sobre {meta.get('eval_visitors', 0):,} visitantes "
        f"com histórico mínimo de treino.".replace(",", ".")
        if run_id
        else f"Avaliação sobre {meta.get('eval_visitors', 0):,} visitantes.".replace(",", ".")
    )

st.info(
    "Na infraestrutura em Cloud ou Local, acessar http://localhost:5000 exibe o painel oficial do MLflow com os gráficos interativos de cada step das épocas (Loss vs Epoch)."  # noqa: E501
)
