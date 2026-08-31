# Deploy em Kubernetes

Manifests mínimos para servir a API do TwinRank AI em um cluster
(testado com o alvo `minikube`/Docker Desktop, nó único).

## O que sobe

| Arquivo | Recurso | Papel |
| --- | --- | --- |
| `00-configmap.yaml` | ConfigMap | Variáveis de ambiente da API (caminhos do modelo, Redis, MLflow). |
| `01-models-pvc.yaml` | PersistentVolumeClaim | Volume de 1 Gi com o checkpoint e o índice FAISS. |
| `02-redis.yaml` | Deployment + Service | Cache-aside das recomendações. |
| `03-api.yaml` | Deployment + Service | API FastAPI, com probes e requests de CPU. |
| `04-hpa.yaml` | HorizontalPodAutoscaler | Escala de 2 a 6 réplicas por utilização de CPU. |
| `05-models-loader.yaml` | Pod | Auxiliar temporário para copiar os artefatos para o PVC. |

## Por que os artefatos não vêm na imagem

O diretório `models/` está no `.dockerignore`, então a imagem não carrega o
checkpoint (~35 MB) nem o índice FAISS (~21 MB). No `docker-compose.yml` esses
arquivos chegam por bind mount do host; no cluster, não há host para montar. O
remote do DVC também não ajuda: é um caminho local da máquina de treino, então
um `dvc pull` de dentro do Pod não encontraria nada.

Por isso os artefatos vão para um PVC, copiados uma vez pelo Pod auxiliar.

## Passo a passo

Construa a imagem e disponibilize-a para o cluster:

```bash
docker build -t twinrank-ai:local .
minikube image load twinrank-ai:local
```

Crie a infraestrutura e popule o volume com o modelo:

```bash
kubectl apply -f k8s/00-configmap.yaml -f k8s/01-models-pvc.yaml -f k8s/02-redis.yaml
kubectl apply -f k8s/05-models-loader.yaml
kubectl wait --for=condition=Ready pod/models-loader --timeout=120s
tar cf - -C models two_tower popularity | kubectl exec -i models-loader -- tar xf - -C /models
kubectl exec models-loader -- ls -lR /models
kubectl delete pod models-loader
```

O comando acima assume Git Bash, Linux ou macOS. Duas armadilhas nesse passo,
ambas verificadas em cluster:

- **`kubectl cp` falha no Git Bash.** Ele monta o caminho remoto com o
  separador do host e o tar do container responde
  `tar: can't change directory to '/models/two_tower'`. O `tar | kubectl exec`
  acima contorna isso. No PowerShell o `kubectl cp` funciona normalmente:
  `kubectl cp models/two_tower/model.joblib models-loader:/models/two_tower/model.joblib`,
  um arquivo por vez.
- **Exporte `MSYS_NO_PATHCONV=1` antes desses comandos no Git Bash.** Sem isso
  o MSYS reescreve `/models` como `C:/Program Files/Git/models` e o tar extrai
  para um caminho inexistente *dentro do container*, sem erro nenhum: o
  `kubectl exec` sai com código 0 e o PVC continua vazio.

No PowerShell, **não** use o `tar | kubectl exec`: o pipeline decodifica o
fluxo como texto e corrompe os artefatos, com o container respondendo
`tar: invalid tar magic`. Use o `kubectl cp` arquivo a arquivo.

Confira a saída do `ls -lR` antes de seguir. O `popularity/` é obrigatório: sem
ele, todo visitante fora da população de treino recebe
`{"item_ids": [], "strategy": "unavailable"}`.

Suba a API e o autoscaler:

```bash
kubectl apply -f k8s/03-api.yaml -f k8s/04-hpa.yaml
kubectl rollout status deploy/twinrank-api
```

Acesse e valide:

```bash
kubectl port-forward svc/twinrank-api 8000:8000
```

Com o port-forward ativo, `curl localhost:8000/health` responde
`{"status":"ok"}` e `curl "localhost:8000/recommend/1?top_k=5"` devolve as
recomendações.

## Limitações conhecidas

- O HPA depende do **metrics-server**. Em `minikube`, habilite com
  `minikube addons enable metrics-server`, senão as métricas de CPU aparecem
  como `<unknown>` e não há escala.
- O PVC usa `ReadWriteOnce`. Todas as réplicas precisam cair no mesmo nó. Em
  cluster multi-nó, troque por `ReadWriteMany` (NFS/EFS) ou embuta os
  artefatos na imagem.
- A rota `POST /train` continua inadequada em produção, como já documentado no
  README principal: sem fila de jobs dedicada, o treino roda dentro do Pod da
  API e disputa CPU com o tráfego de inferência.
- Não há Ingress. O acesso externo é por `port-forward` ou por trocar o
  Service para `NodePort`/`LoadBalancer`. O `port-forward` cai a cada
  `rollout restart`; para validar sem ele, use
  `kubectl exec <pod> -- python -c "import urllib.request; ..."` contra
  `http://localhost:8000` dentro do próprio Pod.
- A API **não falha** quando o checkpoint está ausente: `_load_model_if_available`
  simplesmente não carrega nada, `/health` responde `ok` e o log ainda diz
  `modelo_carregado_no_startup`. O sintoma é `strategy: "unavailable"` em toda
  requisição. Se o `/recommend` vier vazio, confira o conteúdo do PVC antes de
  suspeitar do modelo.
