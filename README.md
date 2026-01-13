# Intelligent-Complaint-Analysis-for-Financial-Services-week_7
📘 Intelligent Complaint Analysis for Financial Services
Week 7 – Text Processing, Embeddings & Vector Indexing
### 1. 🔍 Project Overview

This project focuses on transforming raw CFPB consumer complaint data into a searchable vector database using text chunking, embedding, and ChromaDB indexing. The goal is to prepare the dataset for a Retrieval-Augmented Generation (RAG) pipeline that supports intelligent complaint search and analysis.

The workflow includes EDA, data cleaning, text preprocessing, chunking, vector embedding, and vector store indexing.

2. 📊 Task 1: Exploratory Data Analysis (EDA)

In this step, the full CFPB complaint dataset was loaded, inspected, and analyzed.
Key tasks include:

✔ Dataset Structure & Quality Check

-Verified shape, missing values, and datatype consistency.

-Identified mixed-type columns (e.g., Consumer disputed?).

-Computed descriptive statistics for all columns.

✔ Product Distribution

The dataset contains millions of complaints across 21 product categories, with the largest being:

-Credit reporting

-Debt collection

-Mortgage

-Checking or savings account

-Credit card

This distribution helps understand which complaint topics dominate the dataset.

✔ Narrative Length Analysis

-Calculated word count for complaint narratives.

-Found a large number of extremely short narratives (1–2 words).

-Identified very long narratives (up to ~6,400 words).

-This justified the need for text cleaning & chunking.

✔ Narrative Presence

-With narrative: ~2.98 million

-Without narrative: ~6.6 million

Only records with actual text descriptions are useful for LLM/RAG applications.

3. 🧹 Task 1: Data Cleaning & Filtering

After EDA, the dataset was filtered to meet project requirements:

✔ Included Products

Only the following five categories were kept:

-Credit card

-personal loan

-Savings account

-Money transfers

-Checking or savings account

✔ Cleaning Steps

-Removed complaints with empty narratives.

-Text was normalized by:

   -converting to lowercase

   -removing special characters

   -removing boilerplate text (e.g., “I am writing to file a complaint…”)

-Added a narrative length column (narr_len).

-Saved final cleaned dataset to:
/data/processed/filtered_complaints.csv

4. ✂️ Task 2: Text Chunking

Because long narratives cannot be embedded effectively as single vectors, a chunking strategy was applied.

✔ Chunking Method: RecursiveCharacterTextSplitter

-Handles breaking text without cutting sentences unnaturally

-Keeps chunks semantically meaningful

✔ Final Parameters (example)
chunk_size = 500

chunk_overlap = 100

✔ Why This Works

-Ensures enough context in each chunk

-Prevents loss of meaning at chunk boundaries

-Produces suitable inputs for MiniLM embeddings

5. 🧠 Task 2: Text Embeddings
✔ Selected Model: all-MiniLM-L6-v2

-Fast and lightweight

-384-dimension embeddings

-High semantic accuracy for short/medium text

-Perfect for large-scale RAG systems

✔ Output

Each chunk is transformed into a 384-dimensional vector representing its semantic meaning.

6. 🗂️ Task 2: Vector Store Indexing (ChromaDB)
✔ Why ChromaDB

-Lightweight & easy to use

-Supports metadata

-Fast similarity search

-Local storage (no external dependencies)

✔ Stored Metadata

-Each embedded chunk includes:

-complaint ID

-product category

-original narrative

-chunk index

-cleaned chunk text

This enables traceability back to the original complaint.

✔ Output Saved To

vector_store/

Contains:

-chroma.sqlite

-embedding files

-metadata dictionary

## Task 3: RAG Pipeline & Evaluation

### **RAGPipeline**
- Retrieves relevant complaint chunks from FAISS based on a user query.
- Uses a **prompt template** to guide the LLM in generating answers.
- Supports generating answers from **top-k (default k=5)** most relevant chunks.

### **Qualitative Evaluation**
- Created a list of **representative questions** the system should answer well:
  
What issues do customers report about credit card billing?

How do customers describe problems with personal loans?

Which companies have the most complaints about account management?

What are the main complaints regarding mortgage services?

Are there recurring issues related to bank fees?

How do customers report problems with credit reporting agencies?

What sub-issues are most common in debt collection complaints?

Are there specific states where complaints about credit cards are higher?

- Evaluated the system on these questions.
- Produced a **Markdown evaluation table** with:
- Question  
- Generated Answer  
- Retrieved Sources (top 1-2)  
- Quality Score (1-5)  
- Comments/Analysis  

Example table (Markdown):

| Question | Generated Answer | Retrieved Sources (Top 1-2) | Quality Score | Comments/Analysis |
|----------|----------------|------------------------------|---------------|-----------------|
| What issues do customers report about credit card billing? | Customers report billing disputes, unexpected fees... | 1. “Customer reports unauthorized charges on credit card” 2. “Billing errors for monthly statements” | 4 | Good coverage of main billing issues, minor fee details missing |

---

## Task 4: Interactive Gradio Chat Interface

- Built a **user-friendly web app** using Gradio.
- Features:
- Text input for user questions.
- Submit/Ask button to query the RAG pipeline.
- Display of generated answer.
- Display of **top 2 retrieved source chunks** below the answer for transparency.
- Clear button to reset conversation.
- Optional token-by-token streaming (for improved UX).

### **Usage**
1. Launch the app:
```bash
python app.py
2. Type your question in the input box.

3 .Click Ask.

4 .View the answer and the retrieved sources.

5 .Click Clear to reset.