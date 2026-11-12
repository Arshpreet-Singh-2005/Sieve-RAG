from datasets import load_dataset

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]

print("Documents:", len(corpus))
print("Queries:", len(queries))
print("\nExample document:\n", corpus[0])
print("\nExample query:\n", queries[0])