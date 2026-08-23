"""Testes de integração da API de serving.

Cobertura mínima intencional: o objetivo é garantir que a API sobe e serve o
modelo real de ponta a ponta, não exercitar todas as rotas.
"""

from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from reco.serving.api import app
from reco.settings import get_settings


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Cliente da API com o ciclo de startup executado (carrega o modelo)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def eligible_visitor_id() -> int:
    """Um visitorid real presente no índice do checkpoint treinado."""
    model_path = get_settings().model_path
    if not model_path.exists():
        pytest.skip(f"checkpoint ausente em {model_path}; rode 'dvc repro train'")
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    visitor_index = checkpoint["visitor_index"]
    if not visitor_index:
        pytest.skip("checkpoint sem visitantes indexados")
    return int(next(iter(visitor_index)))


@pytest.fixture(scope="module")
def popularity_checkpoint() -> None:
    """Pula o teste quando o baseline de popularidade não está em disco.

    Mesmo critério de `eligible_visitor_id`: o CI roda sem artefatos versionados
    (só o ponteiro DVC), então testes que dependem de um modelo treinado são
    pulados em vez de falharem.
    """
    settings = get_settings()
    fallback_path = settings.models_dir / "popularity" / "model.joblib"
    if not fallback_path.exists():
        pytest.skip(f"baseline ausente em {fallback_path}; rode 'dvc repro train'")


def test_health_retorna_ok(client: TestClient) -> None:
    """/health responde 200 com status ok."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_visitante_real_retorna_itens(
    client: TestClient, eligible_visitor_id: int
) -> None:
    """/recommend serve recomendações personalizadas para um visitante do índice."""
    response = client.get(f"/recommend/{eligible_visitor_id}", params={"top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == eligible_visitor_id
    assert body["strategy"] == "two_tower"
    assert len(body["item_ids"]) == 5
    assert all(isinstance(item_id, int) for item_id in body["item_ids"])


def test_recommend_cold_start_cai_no_fallback(
    client: TestClient, popularity_checkpoint: None
) -> None:
    """Visitante fora do índice recebe o ranking de popularidade, não um erro."""
    response = client.get("/recommend/999999999", params={"top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "popularity_fallback"
    assert len(body["item_ids"]) == 5


def test_recommend_cold_start_nunca_falha_sem_checkpoint(client: TestClient) -> None:
    """Sem nenhum checkpoint, o cold-start ainda responde 200 com `unavailable`.

    Complementa o teste acima: aquele exige o baseline em disco e é pulado no
    CI, onde não há artefatos. Este roda em qualquer ambiente e garante o que
    de fato não pode regredir — a rota nunca devolve 500 para um visitante
    desconhecido.
    """
    response = client.get("/recommend/999999999", params={"top_k": 5})

    assert response.status_code == 200
    assert response.json()["strategy"] in {"popularity_fallback", "unavailable"}


@pytest.mark.parametrize("top_k", [0, 101])
def test_recommend_rejeita_top_k_fora_do_range(client: TestClient, top_k: int) -> None:
    """top_k fora de [1, 100] é rejeitado pela validação do Pydantic."""
    response = client.get("/recommend/1", params={"top_k": top_k})

    assert response.status_code == 422


def test_recomendacoes_sao_json_serializaveis(client: TestClient, eligible_visitor_id: int) -> None:
    """Os ids devem ser int nativos, não numpy.int64.

    Regressão: o two-tower devolvia numpy.int64, que o Pydantic aceita mas
    `json.dumps` rejeita. Com Redis indisponível o caminho de escrita no cache
    nunca era exercitado, então a falha só aparecia em ambiente com Redis
    ativo (container), com 500 na rota.
    """
    response = client.get(f"/recommend/{eligible_visitor_id}", params={"top_k": 5})

    item_ids = response.json()["item_ids"]
    assert json.dumps(item_ids)
    assert all(type(item_id) is int for item_id in item_ids)
