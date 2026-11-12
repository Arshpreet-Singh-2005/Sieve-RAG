import time
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, Query
from app.retrieval import RetrievalPipeline
from app.cache import SemanticCache

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = RetrievalPipeline(use_reranker=True)
    state["cache"] = SemanticCache()
    yield
    state.clear()


app = FastAPI(
    title="Sieve-RAG",
    description="Low-latency hybrid RAG gateway with semantic caching, BM25 + dense retrieval fusion (RRF), and cross-encoder reranking.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Sieve-RAG"}


@app.get("/search")
async def search(
    q: str = Query(..., min_length=3, description="Scientific claim or query"),
    top_k: int = Query(5, ge=1, le=20),
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    use_cache: bool = True,
):
    t0 = time.perf_counter()
    pipeline: RetrievalPipeline = state["pipeline"]
    cache: SemanticCache = state["cache"]

    query_vec = pipeline.embed_query(q)
    cache_status = "bypassed"

    if use_cache:
        cached_res, cache_status = cache.get(q, query_vec, mode, top_k)
        if cached_res is not None:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "query": q,
                "mode": mode,
                "cache_status": cache_status,
                "latency_ms": latency_ms,
                "results": cached_res,
            }

    results = pipeline.search(q, top_k=top_k, mode=mode, query_vec=query_vec)
    if use_cache:
        cache.set(q, query_vec, mode, top_k, results)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "query": q,
        "mode": mode,
        "cache_status": cache_status,
        "latency_ms": latency_ms,
        "results": results,
    }


@app.get("/metrics")
async def metrics():
    cache: SemanticCache = state["cache"]
    return cache.stats()