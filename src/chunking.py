import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def chunk_dataframe(self, df, text_column="cleaned_narrative"):
        """
        Splits the text narratives into chunks and expands dataframe.
        Returns a new dataframe with columns:
        - chunk_text
        - chunk_id
        - complaint_id
        - product
        """
        chunked_rows = []

        for _, row in df.iterrows():
            chunks = self.splitter.split_text(row[text_column])
            for i, chunk in enumerate(chunks):
                chunked_rows.append({
                    "complaint_id": row["Complaint ID"],
                    "product": row["Product"],
                    "chunk_id": f"{row['Complaint ID']}_chunk_{i}",
                    "chunk_text": chunk
                })

        return pd.DataFrame(chunked_rows)
