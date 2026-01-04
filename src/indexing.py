from chromadb import Client
from chromadb.config import Settings
import pandas as pd
from tqdm import tqdm

class ChromaIndexer:
    def __init__(
        self,
        persist_dir=r"E:\Intelligent-Complaint-Analysis-for-Financial-Services-week_7\vector_store",
        batch_size=5000
    ):
        """
        Initialize Chroma indexer.
        - persist_dir: directory to store vector DB (your custom path)
        - batch_size: number of rows per batch for fast CPU indexing
        """
        self.batch_size = batch_size
        self.chroma_client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))

        self.collection = self.chroma_client.get_or_create_collection(
            name="complaint_vectors",
            metadata={"hnsw:space": "cosine"}  # cosine similarity for MiniLM
        )

        print(f"Vector store will be saved to: {persist_dir}")

    def index(self, df: pd.DataFrame):
        """
        Index the dataframe into Chroma in batches.
        Required columns:
            - chunk_id
            - complaint_id
            - product
            - chunk_text
            - embedding
        """
        total = len(df)
        print(f"Indexing {total} records in batches of {self.batch_size}...\n")

        for i in tqdm(range(0, total, self.batch_size), desc="Chroma indexing"):
            batch = df.iloc[i : i + self.batch_size]

            ids = batch["chunk_id"].astype(str).tolist()
            embeddings = batch["embedding"].tolist()
            documents = batch["chunk_text"].tolist()

            metadatas = batch.apply(lambda row: {
                "complaint_id": str(row["complaint_id"]),
                "product": row["product"]
            }, axis=1).tolist()

            # Add batch into Chroma
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

        print("\n✓ Indexing complete!")
        print("✓ Vector store successfully persisted.")
