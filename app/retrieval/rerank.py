from sentence_transformers import CrossEncoder
from app import config


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(config.RERANK_MODEL)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, f"{c['title']} {c['text']}"] for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for cand, sc in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = round(float(sc), 4)
            item["score"] = item["rerank_score"]
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]