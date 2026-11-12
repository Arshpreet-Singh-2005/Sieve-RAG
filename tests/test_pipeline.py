from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_fusion_ordering():
    dense = [{"doc_id": "1", "title": "A", "text": "x"}, {"doc_id": "2", "title": "B", "text": "y"}]
    bm25 = [{"doc_id": "2", "title": "B", "text": "y"}, {"doc_id": "3", "title": "C", "text": "z"}]
    fused = reciprocal_rank_fusion(dense, bm25, k=60, top_k=3)
    # Doc "2" appears in both lists, so it must rank #1
    assert fused[0]["doc_id"] == "2"


def test_search_and_redis_cache_speedup():
    with TestClient(app) as client:
        q = "0-dimensional biomaterials lack inductive properties."
        r1 = client.get("/search", params={"q": q, "top_k": 5, "mode": "hybrid_rerank"})
        assert r1.status_code == 200
        d1 = r1.json()
        assert len(d1["results"]) == 5

        # Second identical call hits Redis exact cache
        r2 = client.get("/search", params={"q": q, "top_k": 5, "mode": "hybrid_rerank"})
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["cache_status"] == "exact_hit"
        assert d2["latency_ms"] < d1["latency_ms"]