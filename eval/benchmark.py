import math
import time
from collections import defaultdict
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from app.retrieval import RetrievalPipeline
from app.retrieval.fusion import reciprocal_rank_fusion


def evaluate_ranking(ranked_ids: list[str], relevant_ids: set[str]) -> dict:
    hits_5 = len(set(ranked_ids[:5]) & relevant_ids)
    hits_10 = len(set(ranked_ids[:10]) & relevant_ids)
    recall_5 = hits_5 / len(relevant_ids) if relevant_ids else 0.0
    recall_10 = hits_10 / len(relevant_ids) if relevant_ids else 0.0

    mrr_10 = 0.0
    for rank, doc_id in enumerate(ranked_ids[:10], start=1):
        if doc_id in relevant_ids:
            mrr_10 = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:10], start=1)
        if doc_id in relevant_ids
    )
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), 10) + 1))
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0

    return {"Recall@5": recall_5, "Recall@10": recall_10, "MRR@10": mrr_10, "NDCG@10": ndcg_10}


def main():
    console = Console()
    console.print("[bold cyan]1. Loading BEIR SciFact queries and test qrels...[/bold cyan]")
    queries_ds = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    query_map = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels_map = defaultdict(set)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels_map[str(r["query-id"])].add(str(r["corpus-id"]))

    test_qids = [qid for qid in qrels_map if qid in query_map]
    console.print(f"Loaded [bold green]{len(test_qids)}[/bold green] judged test queries.\n")

    pipeline = RetrievalPipeline(use_reranker=True)
    # Truncate long abstracts at 256 tokens during Cross-Encoder scoring -> 4x faster on CPU!
    pipeline.reranker.model.max_length = 256

    console.print("[bold yellow]2. Batch-encoding all 300 test queries on CPU...[/bold yellow]")
    test_texts = [query_map[qid] for qid in test_qids]
    encoded_vecs = pipeline.dense.model.encode(
        test_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    vec_map = {qid: encoded_vecs[i].tolist() for i, qid in enumerate(test_qids)}

    modes = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    summary_rows = []

    for mode in modes:
        sums = defaultdict(float)
        t0 = time.perf_counter()

        for qid in tqdm(test_qids, desc=f"Running {mode:13s}", ncols=90):
            q_text = query_map[qid]
            q_vec = vec_map[qid]

            if mode == "bm25":
                res = pipeline.bm25.search(q_text, top_k=10)
            elif mode == "dense":
                res = pipeline.dense.search(q_text, top_k=10, query_vec=q_vec)
            elif mode == "hybrid":
                d_res = pipeline.dense.search(q_text, top_k=20, query_vec=q_vec)
                b_res = pipeline.bm25.search(q_text, top_k=20)
                res = reciprocal_rank_fusion(d_res, b_res, top_k=10)
            elif mode == "hybrid_rerank":
                d_res = pipeline.dense.search(q_text, top_k=12, query_vec=q_vec)
                b_res = pipeline.bm25.search(q_text, top_k=12)
                fused = reciprocal_rank_fusion(d_res, b_res, top_k=12)
                pairs = [[q_text, f"{c['title']} {c['text'][:600]}"] for c in fused]
                scores = pipeline.reranker.model.predict(pairs, batch_size=32, show_progress_bar=False)
                for cand, sc in zip(fused, scores):
                    cand["rerank_score"] = float(sc)
                res = sorted(fused, key=lambda x: x["rerank_score"], reverse=True)[:10]

            ranked_ids = [r["doc_id"] for r in res]
            metrics = evaluate_ranking(ranked_ids, qrels_map[qid])
            for k, v in metrics.items():
                sums[k] += v

        avg_ms = ((time.perf_counter() - t0) * 1000) / len(test_qids)
        n = len(test_qids)
        row = (
            mode,
            f"{sums['Recall@5']/n:.4f}",
            f"{sums['Recall@10']/n:.4f}",
            f"{sums['MRR@10']/n:.4f}",
            f"{sums['NDCG@10']/n:.4f}",
            f"{avg_ms:.1f}",
        )
        summary_rows.append(row)
        console.print(
            f"[green]✔ {mode}[/green] -> Recall@5: {row[1]} | Recall@10: {row[2]} | MRR@10: {row[3]} | NDCG@10: {row[4]} | {row[5]} ms/query"
        )

    table = Table(title="\nSieve-RAG Retrieval Benchmark (BEIR SciFact Test Split)")
    table.add_column("Mode", style="bold magenta")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("MRR@10", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")

    for r in summary_rows:
        table.add_row(*r)

    console.print(table)


if __name__ == "__main__":
    main()