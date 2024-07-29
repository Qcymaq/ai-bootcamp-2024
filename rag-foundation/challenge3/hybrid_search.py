from typing import List
from collections import defaultdict
from vector_store.node import VectorStoreQueryResult

class HybridSearch:
    
    @staticmethod
    def rrf(
        query_results: List[VectorStoreQueryResult], 
        top_k: int = 5,
        k: float = 50.0
    ) -> VectorStoreQueryResult:
        """
        Re-rank the query results using the Ranked Reciprocal Fusion (RRF) method.

        Args:
            query_results (VectorStoreQueryResult): The initial query results to be re-ranked.
            top_k (int): The number of top results to return.
            k (float): The parameter used in the RRF formula to adjust ranking.

        Returns:
            VectorStoreQueryResult: The re-ranked query results.
        """
        node_rrf_dict = defaultdict(float)
        node_dict = {}

        # Accumulate RRF scores for each node
        for i, node in enumerate(query_results.nodes):
            doc_rank = i + 1
            node_dict[node.id_] = node
            node_rrf_dict[node.id_] += 1.0 / (k + doc_rank)

        # Sort nodes by their RRF score in descending order and get the top_k nodes
        sorted_node_list = sorted(node_rrf_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Construct the re-ranked results
        return VectorStoreQueryResult(
            nodes=[node_dict[node_id] for node_id, _ in sorted_node_list],
            similarities=[score for _, score in sorted_node_list],
            ids=[node_id for node_id, _ in sorted_node_list]
        )
