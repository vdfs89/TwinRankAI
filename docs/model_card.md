# Model Card — TwinRank AI Two-Tower

## Identificação

- **Nome do modelo**: TwinRank AI Two-Tower
- **Tipo**: Neural recommender de duas torres
- **Framework**: PyTorch
- **Finalidade**: recomendação top-K para e-commerce

## Arquitetura

O modelo usa duas torres independentes:

- torre de visitante com embedding de `visitorid`
- torre de item com embedding de `itemid`
- score final calculado por produto interno entre os vetores latentes

O treinamento utiliza Binary Cross-Entropy com negative sampling e early
stopping.

## Dados de treino

- Fonte: RetailRocket E-commerce Dataset
- Sinais: `view`, `addtocart`, `transaction`
- Relevância ponderada para refletir a força do comportamento implícito

## Hiperparâmetros

| Parâmetro | Valor |
|---|---:|
| Embedding dimension | 64 |
| Negative samples por positivo | 4 |
| Learning rate | 0.001 |
| Batch size | 1024 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Top-K padrão | 10 |

## Métricas

| Métrica | Objetivo |
|---|---|
| Recall@K | Cobertura dos itens relevantes recuperados |
| Precision@K | Qualidade do top-K retornado |
| MRR@K | Posição do primeiro item relevante |
| MAP@K | Precisão média ao longo do ranking |
| NDCG@K | Qualidade de ordenação com relevância ponderada |

## Limitações conhecidas

- Cold start para visitantes e itens novos.
- Dependência de interações históricas suficientes.
- Sensibilidade à qualidade da amostragem negativa.
- Métricas ainda precisam ser preenchidas com resultados finais rastreados no MLflow.

## Riscos e vieses

- Popularidade pode enviesar a exposição dos itens mais vistos.
- Eventos implícitos não representam intenção de compra com perfeição.
- Dados de navegação podem refletir sazonalidade e campanhas externas.

## Casos de falha esperados

- Usuário novo sem histórico.
- Item recém-catalogado sem embeddings treinados.
- Sessões muito curtas com sinal insuficiente.

## Deploy e observabilidade

- Modelo deve ser promovido no MLflow Registry antes de ser servido.
- A API deve expor recomendações e health check.
- Latência, throughput e taxa de erro devem ser monitorados nas próximas fases.

## Responsável

Projeto acadêmico para o Tech Challenge da FIAP.
