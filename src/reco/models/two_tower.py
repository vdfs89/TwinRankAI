"""Modelo Two-Tower (Dual Encoder) em PyTorch para feedback implícito.

User Tower e Item Tower são embeddings independentes; o score é o produto
interno entre os dois vetores. Treinado com Binary Cross-Entropy e
negative sampling, conforme referência da Fase 2.
"""

import faiss
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reco.models.base import native_id
from reco.settings import Settings
from reco.training.evaluate import MIN_TRAIN_INTERACTIONS, eligible_visitors
from reco.utils.logging import get_logger

logger = get_logger(__name__)


class _InteractionDataset(Dataset):
    """Dataset de pares (visitor, item, label, peso) com negative sampling on-the-fly."""

    def __init__(
        self,
        visitor_ids: np.ndarray,
        item_ids: np.ndarray,
        relevances: np.ndarray,
        n_items: int,
        n_negatives: int,
    ) -> None:
        self._visitor_ids = visitor_ids
        self._item_ids = item_ids
        self._relevances = relevances
        self._n_items = n_items
        self._n_negatives = n_negatives

    def __len__(self) -> int:
        return len(self._visitor_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visitor = self._visitor_ids[idx]
        pos_item = self._item_ids[idx]
        neg_items = np.random.randint(0, self._n_items, size=self._n_negatives)

        visitors = np.full(1 + self._n_negatives, visitor)
        items = np.concatenate([[pos_item], neg_items])
        labels = np.concatenate([[1.0], np.zeros(self._n_negatives)])
        # C3: o positivo carrega o peso do feedback implícito
        # (view=1.0 / addtocart=3.0 / transaction=5.0); negativos amostrados
        # valem 1.0. Aplicado como peso por amostra na BCE — abordagem mais
        # simples e correta que pos_weight, que é global por classe e não
        # distinguiria uma compra de uma visualização.
        weights = np.concatenate([[self._relevances[idx]], np.ones(self._n_negatives)])

        return (
            torch.tensor(visitors, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )


class _TwoTowerNet(nn.Module):
    """Duas torres de embedding (user/item) com score via produto interno."""

    def __init__(self, n_visitors: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        # sparse=True: o gradiente de cada batch toca só as linhas usadas.
        # Combinado com SparseAdam, evita que o otimizador atualize densamente
        # os ~85M parâmetros a cada step (causa raiz das 11,5h de treino).
        self.user_tower = nn.Embedding(n_visitors, embedding_dim, sparse=True)
        self.item_tower = nn.Embedding(n_items, embedding_dim, sparse=True)
        nn.init.normal_(self.user_tower.weight, std=0.01)
        nn.init.normal_(self.item_tower.weight, std=0.01)

    def forward(self, visitor_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Retorna logits (produto interno) para pares (visitor, item)."""
        user_emb = self.user_tower(visitor_idx)
        item_emb = self.item_tower(item_idx)
        return (user_emb * item_emb).sum(dim=-1)


class TwoTowerRecommender:
    """Wrapper de treino/inferência do modelo two-tower, com early stopping."""

    def __init__(self, settings: Settings) -> None:
        """Initialize TwoTower model parameters."""
        self._settings = settings
        self._visitor_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._index_to_item: dict[int, int] = {}
        self._net: _TwoTowerNet | None = None
        self._faiss_index = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_events: pd.DataFrame) -> None:
        """Treina o two-tower com BCE ponderada + negative sampling e early stopping."""
        torch.manual_seed(self._settings.random_seed)
        np.random.seed(self._settings.random_seed)

        # C5: treina apenas visitantes com histórico mínimo. Um embedding de ID
        # aprendido a partir de 1 única interação fica praticamente na
        # inicialização — só infla a matriz de parâmetros e o tempo de treino.
        # Mesmo critério aplicado na avaliação (reco.training.evaluate).
        # O corte padrão é MIN_TRAIN_INTERACTIONS (5), o mesmo da avaliação.
        # Configurações menores — como a do demo plugável, que treina sobre uma
        # planilha de dezenas de linhas — podem baixá-lo sem afetar o pipeline.
        min_interactions = getattr(self._settings, "min_train_interactions", MIN_TRAIN_INTERACTIONS)
        eligible = eligible_visitors(train_events, min_interactions)
        filtered_events = train_events[train_events["visitorid"].isin(eligible)]
        logger.info(
            "populacao_de_treino_filtrada",
            min_interacoes=min_interactions,
            visitors_antes=int(train_events["visitorid"].nunique()),
            visitors_depois=int(filtered_events["visitorid"].nunique()),
            eventos_antes=len(train_events),
            eventos_depois=len(filtered_events),
        )

        visitors = filtered_events["visitorid"].unique()
        items = filtered_events["itemid"].unique()
        self._visitor_index = {v: i for i, v in enumerate(visitors)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._index_to_item = {i: native_id(it) for it, i in self._item_index.items()}

        visitor_ids = filtered_events["visitorid"].map(self._visitor_index).to_numpy()
        item_ids = filtered_events["itemid"].map(self._item_index).to_numpy()
        relevances = filtered_events["relevance"].to_numpy(dtype=np.float32)

        dataset = _InteractionDataset(
            visitor_ids=visitor_ids,
            item_ids=item_ids,
            relevances=relevances,
            n_items=len(items),
            n_negatives=self._settings.negative_samples_per_positive,
        )
        loader = DataLoader(
            dataset,
            batch_size=self._settings.batch_size,
            shuffle=True,
            num_workers=self._settings.dataloader_workers,
            persistent_workers=self._settings.dataloader_workers > 0,
        )

        self._net = _TwoTowerNet(
            n_visitors=len(visitors),
            n_items=len(items),
            embedding_dim=self._settings.embedding_dim,
        ).to(self._device)

        # SparseAdam: único otimizador Adam-like que aceita gradiente esparso.
        # Não suporta weight_decay nem amsgrad — a regularização aqui vem do
        # negative sampling e do early stopping.
        optimizer = torch.optim.SparseAdam(
            list(self._net.parameters()), lr=self._settings.learning_rate
        )
        # reduction="none" para poder ponderar cada amostra pela relevância (C3).
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self._settings.max_epochs):
            epoch_loss = self._run_epoch(loader, optimizer, criterion)
            logger.info("epoch_concluida", epoch=epoch, loss=round(epoch_loss, 5))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._settings.early_stopping_patience:
                    logger.info("early_stopping_acionado", epoch=epoch, best_loss=best_loss)
                    break

        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """Constrói o índice FAISS com os embeddings dos itens."""
        if self._net is None:
            return

        self._net.eval()
        with torch.no_grad():
            item_indices = torch.arange(len(self._item_index), device=self._device)
            item_emb = self._net.item_tower(item_indices).cpu().numpy().astype(np.float32)

        dim = item_emb.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(item_emb)
        logger.info("faiss_index_construido", num_items=self._faiss_index.ntotal)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """Executa uma época de treino e retorna a loss média."""
        assert self._net is not None
        self._net.train()
        total_loss = 0.0
        n_batches = 0

        for visitors_batch, items_batch, labels_batch, weights_batch in loader:
            visitors_batch = visitors_batch.view(-1).to(self._device)
            items_batch = items_batch.view(-1).to(self._device)
            labels_batch = labels_batch.view(-1).to(self._device)
            weights_batch = weights_batch.view(-1).to(self._device)

            optimizer.zero_grad()
            logits = self._net(visitors_batch, items_batch)
            # C3: média ponderada — uma transaction pesa 5x uma view.
            per_sample = criterion(logits, labels_batch)
            loss = (per_sample * weights_batch).sum() / weights_batch.sum()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict_top_k(self, visitor_id: int, k: int) -> list[int]:
        """Calcula score do visitante contra todos os itens e retorna o top-k."""
        if self._net is None or visitor_id not in self._visitor_index:
            return []

        self._net.eval()
        v_idx = torch.tensor([self._visitor_index[visitor_id]], device=self._device)

        if self._faiss_index is not None:
            with torch.no_grad():
                user_emb = self._net.user_tower(v_idx).cpu().numpy().astype(np.float32)
            scores, indices = self._faiss_index.search(user_emb, k)
            top_indices = indices[0]
        else:
            all_item_indices = torch.arange(len(self._item_index), device=self._device)
            with torch.no_grad():
                user_emb = self._net.user_tower(v_idx)
                item_emb = self._net.item_tower(all_item_indices)
                scores = (item_emb @ user_emb.T).squeeze(-1)
            top_indices = torch.topk(scores, k=min(k, len(scores))).indices.cpu().numpy()

        return [self._index_to_item[i] for i in top_indices if i in self._index_to_item]

    def save(self, path: str) -> None:
        """Save model weights and index structures."""
        assert self._net is not None
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "visitor_index": self._visitor_index,
                "item_index": self._item_index,
            },
            path,
        )
        if self._faiss_index is not None:
            faiss_path = self._settings.faiss_index_path
            faiss_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._faiss_index, str(faiss_path))

    def load(self, path: str) -> None:
        """Carrega pesos do modelo e os índices visitor/item."""
        # weights_only=False: o checkpoint carrega também os dicionários de
        # índice visitor/item (não só tensores), e o default do torch >= 2.6
        # os rejeitaria. O artefato é produzido pelo próprio pipeline.
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._visitor_index = checkpoint["visitor_index"]
        self._item_index = checkpoint["item_index"]
        self._index_to_item = {i: native_id(it) for it, i in self._item_index.items()}

        self._net = _TwoTowerNet(
            n_visitors=len(self._visitor_index),
            n_items=len(self._item_index),
            embedding_dim=self._settings.embedding_dim,
        ).to(self._device)
        self._net.load_state_dict(checkpoint["state_dict"])

        faiss_path = self._settings.faiss_index_path
        if faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
