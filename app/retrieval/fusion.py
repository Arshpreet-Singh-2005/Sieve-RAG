from app import config


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = config.RRF_K,
    top_k: int = config.CANDIDATE_POOL,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["doc_id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["doc_id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    fused = []
    for doc_id in sorted_doc_ids[:top_k]:
        item = dict(doc_map[doc_id])
        item["rrf_score"] = round(rrf_scores[doc_id], 6)
        item["score"] = item["rrf_score"]
        fused.append(item)
    return fused