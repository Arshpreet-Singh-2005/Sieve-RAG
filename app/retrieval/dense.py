from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app import config


class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.client = QdrantClient(config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=30)

    def embed(self, query: str) -> list[float]:
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def search(self, query: str, top_k: int = 10, query_vec: list[float] | None = None) -> list[dict]:
        if query_vec is None:
            query_vec = self.embed(query)

        points = self.client.query_points(
            collection_name=config.COLLECTION,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "doc_id": str(pt.payload["doc_id"]),
                "title": pt.payload["title"],
                "text": pt.payload["text"],
                "score": round(float(pt.score), 6),
            }
            for pt in points
        ]