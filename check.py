from qdrant_client import QdrantClient
import redis

print("Qdrant:", QdrantClient("localhost", port=6333).get_collections())
print("Redis:", redis.Redis(host="localhost", port=6379).ping())