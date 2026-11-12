from app import config
from app.retrieval.dense import DenseRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import CrossEncoderReranker


class RetrievalPipeline:
    def __init__(self, use_reranker: bool = True):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def embed_query(self, query: str) -> list[float]:
        return self.dense.embed(query)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid_rerank",
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        if mode == "dense":
            return self.dense.search(query, top_k=top_k, query_vec=query_vec)

        if mode == "bm25":
            return self.bm25.search(query, top_k=top_k)

        pool = max(config.CANDIDATE_POOL, top_k * 3)
        dense_res = self.dense.search(query, top_k=pool, query_vec=query_vec)
        bm25_res = self.bm25.search(query, top_k=pool)
        fused = reciprocal_rank_fusion(dense_res, bm25_res, top_k=pool)

        if mode == "hybrid":
            return fused[:top_k]

        if mode == "hybrid_rerank":
            if self.reranker is None:
                return fused[:top_k]
            return self.reranker.rerank(query, fused, top_k=top_k)

        raise ValueError(f"Unsupported retrieval mode: {mode}")