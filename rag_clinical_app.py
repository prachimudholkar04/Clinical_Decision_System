import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# --- Prepare the Corpus ---
corpus = [
    "ACE inhibitors are commonly used to treat hypertension in elderly patients.",
    "Calcium channel blockers can be effective in managing resistant hypertension.",
    "Beta blockers should be avoided in asthma patients with hypertension.",
    "Diuretics are often first-line treatment for high blood pressure.",
    "Recent trials show SGLT2 inhibitors may help in heart failure patients with diabetes.",
    "Metformin is the first-line therapy for type 2 diabetes according to most guidelines.",
    "Elderly patients with atrial fibrillation are often managed with anticoagulants and rate control.",
    "Community-acquired pneumonia should be treated based on severity, using macrolides or beta-lactams.",
    "Immunotherapy has shown promise in treating non-small cell lung cancer.",
    "Hypertension in chronic kidney disease requires ACE inhibitors or ARBs as preferred agents."
]

# --- Embed Corpus ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=False).astype("float32")
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# --- Load Generator Model ---
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# --- Define RAG Function ---
def rag_pipeline(query):
    query_embedding = embedder.encode([query], convert_to_tensor=False).astype("float32")
    D, I = index.search(query_embedding, k=3)
    retrieved_docs = [corpus[i] for i in I[0]]
    context = "\n".join(retrieved_docs)

    prompt = f"""You are a clinical assistant. Use the information below to answer the question.

Documents:
{context}

Question:
{query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Warning System ---
    warnings = []

    if len(retrieved_docs) == 0:
        warnings.append("⚠️ No relevant documents were retrieved. Answer may be unreliable.")
    if np.mean(D[0]) > 1.5:  # Rough heuristic: low similarity = high distance
        warnings.append("⚠️ Retrieved documents have low relevance. Please double-check the response.")
    if any(word in answer.lower() for word in ["not sure", "unknown", "no data"]):
        warnings.append("⚠️ The model indicates uncertainty. Verify against guidelines.")

    warning_text = "\n".join(warnings) if warnings else "✅ No warnings. The result seems reasonable."

    return answer, "\n\n".join(retrieved_docs), warning_text


# --- Gradio Interface ---
gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(lines=2, placeholder="Enter your clinical question here..."),
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Top Retrieved Documents"),
        gr.Textbox(label="⚠️ Warnings or Confidence Notes")
    ],
    title="Clinical Decision Support (RAG)",
    description="Ask a medical question and get an answer based on retrieved clinical knowledge."
).launch()
