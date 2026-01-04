from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=512):
        """
        Initialize the embedder.
        - model_name: the MiniLM model to use
        - batch_size: number of texts to embed at once (adjust for your CPU RAM)
        """
        print(f"Loading embedding model: {model_name}")
        # CPU-only
        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def embed_chunks(self, chunk_df: pd.DataFrame, text_column="chunk_text"):
        """
        Embed all chunks in the dataframe efficiently using batching.
        Adds a new 'embedding' column to the dataframe.
        """
        texts = chunk_df[text_column].tolist()
        embeddings = []

        # Use batching with tqdm for progress
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i+self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,  # faster than lists
                normalize_embeddings=True  # optional, can improve cosine search later
            )
            embeddings.extend(batch_embeddings)

        chunk_df["embedding"] = embeddings
        return chunk_df
