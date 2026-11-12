import pickle
import numpy as np
from app import config
from app.ingest import tokenize


class BM25Retriever:
    def __init__(self):
        if not config.BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {config.BM25_INDEX_PATH}. Run `python -m app.ingest` first."
            )
        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.docs = data["docs"]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.docs[idx]
            results.append(
                {
                    "doc_id": str(doc["doc_id"]),
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": round(float(scores[idx]), 6),
                }
            )
        return results