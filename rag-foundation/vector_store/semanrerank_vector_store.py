# autoflake: off
# flake8: noqa: F841
from sentence_transformers import CrossEncoder
import sys
from typing import Dict, List, cast

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult

logger.add(
    sink=sys.stdout,
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
)


class SemanticVectorStore(BaseVectorStore):
    """Semantic Vector Store using SentenceTransformer embeddings with re-ranking."""

    saved_file: str = "rag-foundation/data/test_db_00.csv"
    embed_model_name: str = "all-MiniLM-L6-v2"
    re_rank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    embed_model: SentenceTransformer = SentenceTransformer(embed_model_name)
    re_rank_model: CrossEncoder = CrossEncoder(re_rank_model_name)
    re_rank_models: Dict[str, CrossEncoder] = {re_rank_model_name: re_rank_model}

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_store()

    def get(self, text_id: str) -> TextNode:
        """Get node."""
        try:
            return self.node_dict[text_id]
        except KeyError:
            logger.error(f"Node with id `{text_id}` not found.")
            return None

    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            if node.embedding is None:
                logger.info(
                    "Found node without embedding, calculating "
                    f"embedding with model {self.embed_model_name}"
                )
                node.embedding = self._get_text_embedding(node.text)
            self.node_dict[node.id_] = node
        self._update_csv()  # Update CSV after adding nodes
        return [node.id_ for node in nodes]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Calculate embedding."""
        return self.embed_model.encode(text).tolist()

    def delete(self, node_id: str, **delete_kwargs: Dict) -> None:
        """Delete nodes using node_id."""
        if node_id in self.node_dict:
            del self.node_dict[node_id]
            self._update_csv()  # Update CSV after deleting nodes
        else:
            logger.error(f"Node with id `{node_id}` not found.")

    def _calculate_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        doc_ids: List[str],
        similarity_top_k: int = 3,
    ) -> tuple[List[float], List[str]]:
        """Get top nodes by similarity to the query."""
        qembed_np = np.array(query_embedding)
        dembed_np = np.array(doc_embeddings)

        # calculate the dot product of the query embedding with the document embeddings
        dproduct_arr = np.dot(qembed_np, dembed_np.T)

        # calculate the cosine similarity by dividing the dot product by the norm
        qnorm = np.linalg.norm(qembed_np)
        dnorm = np.linalg.norm(dembed_np, axis=1)
        cos_sim_arr = dproduct_arr / (qnorm * dnorm)

        # get the indices of the top k similarities
        top_k_indices = np.argsort(cos_sim_arr)[-similarity_top_k:][::-1]
        similarities = cos_sim_arr[top_k_indices].tolist()
        node_ids = [doc_ids[idx] for idx in top_k_indices]

        return similarities, node_ids

    def query(self, query: str, top_k: int = 3, re_rank_model_name: str = None) -> VectorStoreQueryResult:
        """Query similar nodes with re-ranking."""
        query_embedding = cast(List[float], self._get_text_embedding(query))
        doc_embeddings = [node.embedding for node in self.node_dict.values()]
        doc_ids = list(self.node_dict.keys())

        if len(doc_embeddings) == 0:
            logger.error("No documents found in the index.")
            result_nodes, similarities, node_ids = [], [], []
        else:
            similarities, node_ids = self._calculate_similarity(
                query_embedding, doc_embeddings, doc_ids, top_k
            )
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]

            logger.debug(f"Initial similarities: {similarities}")
            logger.debug(f"Initial result nodes: {[node.id_ for node in result_nodes]}")

            # Re-rank using Cross-Encoder
            if re_rank_model_name:
                if re_rank_model_name not in self.re_rank_models:
                    self.re_rank_models[re_rank_model_name] = CrossEncoder(re_rank_model_name)
                re_rank_model = self.re_rank_models[re_rank_model_name]
            else:
                re_rank_model = self.re_rank_model

            pairs = [[query, node.text] for node in result_nodes]
            re_rank_scores = re_rank_model.predict(pairs)
            logger.debug(f"Re-rank scores: {re_rank_scores}")

            re_rank_indices = np.argsort(re_rank_scores)[::-1]
            result_nodes = [result_nodes[i] for i in re_rank_indices]
            similarities = [re_rank_scores[i] for i in re_rank_indices]
            node_ids = [node_ids[i] for i in re_rank_indices]

            logger.debug(f"Re-ranked result nodes: {[node.id_ for node in result_nodes]}")
            logger.debug(f"Re-ranked similarities: {similarities}")

        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )

    def batch_query(self, query: List[str], top_k: int = 3, re_rank_model_name: str = None) -> List[VectorStoreQueryResult]:
        """Batch query similar nodes with optional re-ranking."""
        return [self.query(q, top_k, re_rank_model_name) for q in query]
