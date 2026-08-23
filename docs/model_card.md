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

O treinamento utiliza Binary Cross-Entropy ponderada pela relevância do
feedback implícito (view = 1,0; addtocart = 3,0; transaction = 5,0), com
negative sampling e early stopping. As camadas de embedding são esparsas
(`nn.Embedding(sparse=True)` + `SparseAdam`), de modo que cada passo do
otimizador atualiza apenas as linhas presentes no batch.

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
| Otimizador | SparseAdam |
| Mínimo de interações de treino por visitante | 5 |

## Métricas

| Métrica | Objetivo |
|---|---|
| Recall@K | Cobertura dos itens relevantes recuperados |
| Precision@K | Qualidade do top-K retornado |
| MRR@K | Posição do primeiro item relevante |
| MAP@K | Precisão média ao longo do ranking |
| NDCG@K | Qualidade de ordenação com relevância ponderada |
| Recall@K_novel | Recall restrito a itens ausentes do histórico de treino do usuário (descoberta pura) |

### Resultados

Avaliação sobre 2.920 visitantes com pelo menos 5 interações no conjunto de
treino (`eval_min_train_interactions=5`). Run do Two-Tower rastreado no MLflow
sob `a9e7c00368df4b93acc655a6493e9b69`, registrado como `twinrank-ai-two-tower`
v2 em stage Production. Números gerados por `dvc repro` em
`reports/metrics.json`.

| Modelo | Recall@10 | Precision@10 | MAP@10 | MRR@10 | NDCG@10 |
|---|---|---|---|---|---|
| Popularity | 0,00342 | 0,00147 | 0,00310 | 0,00965 | 0,00408 |
| Matrix Factorization | 0,02265 | 0,00870 | 0,01386 | 0,03086 | 0,01958 |
| Two-Tower | **0,12311** | **0,03205** | **0,07604** | **0,13286** | **0,10078** |

O Two-Tower atinge 36,0x o Recall@10 da popularidade, 24,7x o NDCG@10, e
5,4x o Recall@10 do Matrix Factorization.

### Descoberta vs. repetição

`Recall@10_novel` exclui do conjunto de relevantes de cada usuário qualquer
item já presente em seu histórico de treino, isolando descoberta real de
repetição de itens conhecidos.

| Modelo | Recall@10 (geral) | Recall@10 (novel) | Fração do recall que é repetição |
|---|---|---|---|
| Popularity | 0,00342 | 0,00202 | 40,8% |
| Matrix Factorization | 0,02265 | 0,00359 | 84,1% |
| Two-Tower | 0,12311 | 0,01168 | 90,5% |

A taxa de repetição cresce com o grau de personalização: a popularidade não
personaliza e por isso acerta itens novos com frequência relativa maior,
enquanto modelos de embedding de ID aprendem fortemente a afinidade
usuário-item — inclusive para itens que o usuário já consumiu.

### Teto de memorização

Para dimensionar o componente de repetição, calculamos o Recall@10 de um
oráculo que apenas devolve os itens de maior relevância do próprio histórico de
treino de cada usuário — o melhor que uma estratégia de pura memorização
poderia atingir.

| Referência | Recall@10 |
|---|---|
| Teto de memorização (oráculo) | 0,16158 |
| Two-Tower (real) | 0,12311 |
| **Fração do teto atingida** | **76,2%** |

Reproduzível por `python scripts/memorization_ceiling.py`, sobre a mesma
população filtrada de 2.920 visitantes usada nas métricas oficiais — a
comparação só é válida sob essa condição. Estimativas anteriores que dividiam o
Recall@10 filtrado por um teto medido na população não filtrada misturavam duas
populações distintas e foram descartadas.

## Reprodutibilidade

- Seeds fixadas em `TwoTowerRecommender.fit`: `torch.manual_seed` (inicialização
  dos embeddings e ordem do DataLoader) e `np.random.seed` (amostragem negativa
  em `_InteractionDataset`). Ambas derivam de `settings.random_seed`.
- Pipeline versionado com DVC (`dvc.yaml` / `dvc.lock`): `dvc repro` reexecuta
  preprocess, feature engineering, treino e avaliação de forma determinística.
- Métricas persistidas em `reports/metrics.json` e registradas no MLflow junto
  aos parâmetros de auditoria da população avaliada
  (`eval_min_train_interactions`, `eval_visitors_before_filter`,
  `eval_visitors_after_filter`).

## Limitações conhecidas

- **Mudança da população de avaliação.** A partir desta versão a avaliação
  restringe-se a visitantes com pelo menos 5 interações de treino, reduzindo a
  população de 23.476 para 2.920 usuários. O filtro foi necessário porque 57%
  dos visitantes originais tinham uma única interação de treino e dominavam a
  média, tornando a métrica anterior sem sentido estatístico. Consequência
  prática: os números desta versão **não são comparáveis** com os de versões
  anteriores — uma queda aparente do baseline reflete a troca de população, não
  uma regressão de modelo. Qualquer tabela de métricas deve trazer essa nota
  explicitamente.
- **Parte substancial do ganho é repetição, não descoberta.** Do Recall@10 =
  0,12311 do Two-Tower, apenas ~9,5% (Recall@10_novel = 0,01168) representa
  acerto em item que o usuário nunca interagiu antes; os ~90,5% restantes são
  casos em que o modelo recomenda de volta algo que o próprio histórico já
  continha. Isso é esperado em modelos de embedding de ID puro — sem features
  de conteúdo, a única generalização possível é por co-ocorrência aprendida — e
  é uma limitação conhecida da arquitetura, não um erro de implementação. Mesmo
  controlando por descoberta pura o Two-Tower ainda lidera, com 3,3x o
  Recall@10_novel do Matrix Factorization e 5,8x o da Popularidade, mas essa
  margem é menor que o Recall@10 geral sugere. A comparação justa deve citar as
  duas métricas, nunca o Recall@10 geral como headline isolado.
- Cold start para visitantes e itens novos.
- Dependência de interações históricas suficientes: visitantes abaixo do limiar
  de 5 interações não são modelados nem avaliados.
- Sensibilidade à qualidade da amostragem negativa.

## Riscos e vieses

- Popularidade pode enviesar a exposição dos itens mais vistos.
- Eventos implícitos não representam intenção de compra com perfeição.
- Dados de navegação podem refletir sazonalidade e campanhas externas.

## Casos de falha esperados

- Usuário novo sem histórico: o two-tower não tem embedding para ele e a API
  degrada para o ranking global de popularidade, sinalizado no campo `strategy`
  da resposta como `popularity_fallback`.
- Item recém-catalogado sem embeddings treinados: não é recuperável até o
  próximo ciclo de treino.
- Sessões muito curtas com sinal insuficiente.

## Deploy e observabilidade

- Modelo deve ser promovido no MLflow Registry antes de ser servido.
- A API expõe `/recommend`, `/predict`, `/health` e `/model/version`; o modelo é
  carregado no startup (~8 s) para não penalizar o primeiro request.
- Latência de cada request é medida por middleware, registrada no log
  estruturado e devolvida no header `X-Response-Time-ms`. Throughput e taxa de
  erro seguem pendentes de instrumentação.

## Dívida técnica conhecida

- `Settings` usa `BaseSettings` do pydantic v1 e não é hashable. Isso já
  derrubou o serving uma vez, via `lru_cache` sobre `get_recommendation_service`
  (`TypeError` a cada `/recommend`, resultando em 500). A correção aplicada foi
  um singleton de módulo; a migração de `Settings` para pydantic v2 continua
  pendente.
- `POST /train` executa via `BackgroundTasks` no mesmo processo do servidor: sem
  fila dedicada, não há retry, persistência entre restarts nem visibilidade de
  progresso.
- As tabelas de métricas do README e deste Model Card são sincronizadas à mão a
  partir de `scripts/export_metrics_for_readme.py`, que só imprime em stdout.

## Responsável

Projeto acadêmico para o Tech Challenge da FIAP.
