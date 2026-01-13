import gradio as gr
from src.faiss_store import FAISSVectorStore
from src.RAG_Pipeline import RAGPipeline

# Initialize FAISS store and RAG pipeline
store = FAISSVectorStore("complaint_embeddings.parquet")
rag = RAGPipeline(vector_store=store)

# Prompt template
prompt_template = """
You are a financial analyst assistant for CrediTrust.
Use the following retrieved complaint excerpts to answer the user's question.
If the context doesn't contain the answer, say you don't have enough information.

Context: {context}
Question: {question}
Answer:
"""

# Gradio function
def answer_question(user_question):
    answer, retrieved = rag.generate_answer(user_question, prompt_template)
    sources_text = "\n\n".join([f"{i+1}. {r['document']}" for i, r in enumerate(retrieved[:2])])
    return answer, sources_text

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## CrediTrust Complaint RAG Chatbot")
    question = gr.Textbox(label="Enter your question", placeholder="Ask about complaints...")
    submit = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer")
    sources_output = gr.Textbox(label="Retrieved Sources")

    submit.click(answer_question, inputs=question, outputs=[answer_output, sources_output])

    gr.Button("Clear").click(lambda: ("", ""), None, [answer_output, sources_output])

demo.launch()
