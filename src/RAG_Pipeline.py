from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGPipeline:
    def __init__(self, vector_store, embedder_model="all-MiniLM-L6-v2", top_k=5, llm_model_name="google/flan-t5-small"):
        self.vector_store = vector_store
        self.top_k = top_k
        self.embedder = SentenceTransformer(embedder_model)
        self.llm = pipeline("text2text-generation", model=llm_model_name)

    def retrieve(self, question: str):
        query_emb = self.embedder.encode([question], convert_to_numpy=True)
        return self.vector_store.search(query_emb, k=self.top_k)

    def generate_answer(self, question: str, prompt_template=None):
        retrieved = self.retrieve(question)
        context = "\n".join([r["document"] for r in retrieved])
        prompt = prompt_template.format(context=context, question=question) \
                 if prompt_template else f"Context: {context}\nQuestion: {question}\nAnswer:"
        answer = self.llm(prompt)[0]['generated_text']
        return answer, retrieved

    def evaluate_questions(self, questions: list, prompt_template=None):
        evaluation = []
        for q in questions:
            answer, retrieved = self.generate_answer(q, prompt_template)
            sources = retrieved[:2]  # top 2 sources
            evaluation.append({
                "question": q,
                "answer": answer,
                "sources": sources,
                "quality_score": None,  # to fill manually
                "comments": ""
            })
        return evaluation
