import hashlib
import json
import numpy as np
import redis
from app import config


class SemanticCache:
    def __init__(self):
        self.client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True,
        )
        self.stats_key = "sieve:stats"

    @staticmethod
    def _exact_key(query: str, mode: str, top_k: int) -> str:
        norm = query.strip().lower()
        digest = hashlib.sha256(f"{norm}|{mode}|{top_k}".encode()).hexdigest()[:16]
        return f"sieve:exact:{digest}"

    def get(
        self, query: str, query_vec: list[float], mode: str, top_k: int
    ) -> tuple[list[dict] | None, str]:
        exact_key = self._exact_key(query, mode, top_k)
        cached = self.client.get(exact_key)
        if cached:
            self.client.hincrby(self.stats_key, "exact_hits", 1)
            return json.loads(cached), "exact_hit"

        q_vec = np.array(query_vec, dtype=np.float32)
        best_sim = -1.0
        best_results = None

        for key in self.client.scan_iter(f"sieve:sem:{mode}:{top_k}:*"):
            raw = self.client.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            c_vec = np.array(entry["vector"], dtype=np.float32)
            sim = float(np.dot(q_vec, c_vec))
            if sim >= config.SEMANTIC_THRESHOLD and sim > best_sim:
                best_sim = sim
                best_results = entry["results"]

        if best_results is not None:
            self.client.hincrby(self.stats_key, "semantic_hits", 1)
            return best_results, f"semantic_hit (sim={best_sim:.3f})"

        self.client.hincrby(self.stats_key, "misses", 1)
        return None, "miss"

    def set(
        self, query: str, query_vec: list[float], mode: str, top_k: int, results: list[dict]
    ) -> None:
        exact_key = self._exact_key(query, mode, top_k)
        self.client.set(exact_key, json.dumps(results), ex=config.CACHE_TTL_SECONDS)

        sem_key = f"sieve:sem:{mode}:{top_k}:{exact_key.split(':')[-1]}"
        payload = {"query": query, "vector": query_vec, "results": results}
        self.client.set(sem_key, json.dumps(payload), ex=config.CACHE_TTL_SECONDS)

    def stats(self) -> dict:
        raw = self.client.hgetall(self.stats_key)
        exact = int(raw.get("exact_hits", 0))
        sem = int(raw.get("semantic_hits", 0))
        misses = int(raw.get("misses", 0))
        total = exact + sem + misses
        return {
            "exact_hits": exact,
            "semantic_hits": sem,
            "misses": misses,
            "total_requests": total,
            "hit_rate": round((exact + sem) / total, 4) if total > 0 else 0.0,
        }