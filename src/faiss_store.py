import faiss
import pickle
import numpy as np
import pyarrow.parquet as pq
import os

class FAISSVectorStore:
    def __init__(self, parquet_path: str, embedding_dim: int = 384, 
                 index_path: str = "data/faiss/index.faiss",
                 meta_path: str = "data/faiss/meta.pkl"):
        self.parquet_path = parquet_path
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.meta_path = meta_path

        self.index = None
        self.documents = []
        self.metadata = []

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()
        else:
            self._build_index()
            self._save()

    def _build_index(self):
        print("🔹 Building FAISS index (streaming Parquet)...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        pq_file = pq.ParquetFile(self.parquet_path)
        for batch in pq_file.iter_batches(columns=["embedding", "document", "metadata"], batch_size=10_000):
            batch = batch.to_pydict()
            embeddings = np.array([[e for e in emb] for emb in batch["embedding"]], dtype="float32")
            self.index.add(embeddings)
            self.documents.extend(batch["document"])
            self.metadata.extend(batch["metadata"])

        print(f"✅ FAISS index built with {self.index.ntotal} vectors")

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
        print(f"💾 Index saved to {self.index_path} and metadata to {self.meta_path}")

    def _load(self):
        print("🔹 Loading FAISS index and metadata from disk...")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx]
            })
        return results
