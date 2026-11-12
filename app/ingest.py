import pickle
import re
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app import config


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def main():
    print("1. Loading BeIR/scifact corpus...")
    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]

    print(f"2. Loading embedding model ({config.EMBED_MODEL})...")
    model = SentenceTransformer(config.EMBED_MODEL)
    
    # Increase timeout to 60s so Windows Docker disk flushes never time out
    client = QdrantClient(config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=60)

    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)

    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )

    bm25_tokens = []
    bm25_docs = []

    batch_size = 128
    for start in range(0, len(corpus), batch_size):
        batch = corpus[start : start + batch_size]
        texts = [f"{t} {x}" for t, x in zip(batch["title"], batch["text"])]
        vectors = model.encode(texts, normalize_embeddings=True)

        points = []
        for i in range(len(texts)):
            doc_id = str(batch["_id"][i])
            title = batch["title"][i]
            text = batch["text"][i]

            points.append(
                PointStruct(
                    id=start + i,
                    vector=vectors[i].tolist(),
                    payload={
                        "doc_id": doc_id,
                        "title": title,
                        "text": text,
                    },
                )
            )
            bm25_tokens.append(tokenize(texts[i]))
            bm25_docs.append({"doc_id": doc_id, "title": title, "text": text})

        # wait=True only on the final batch so intermediate uploads are instant
        is_last_batch = (start + batch_size) >= len(corpus)
        client.upsert(
            collection_name=config.COLLECTION,
            points=points,
            wait=is_last_batch,
        )
        print(f"Uploaded {start + len(texts)} / {len(corpus)}")

    print("3. Saving BM25 index to data/bm25_index.pkl...")
    bm25 = BM25Okapi(bm25_tokens)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": bm25_docs}, f)

    print("Done. Points in collection:", client.count(config.COLLECTION).count)


if __name__ == "__main__":
    main()