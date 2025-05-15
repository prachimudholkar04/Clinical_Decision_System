import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import wikipediaapi
from transformers import pipeline
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)

# Optional: Log user queries
import csv

# ICD-10 codes mapped manually (add more if needed)
icd_mapping = {
    "Hypertension": "I10",
    "Diabetes mellitus type 2": "E11",
    "Asthma": "J45",
    "Chronic kidney disease": "N18",
    "Heart failure": "I50",
    "Depression": "F32",
    "Breast cancer": "C50",
    "Pneumonia": "J18",
    "Anemia": "D64",
    "Migraine": "G43"
}

# Init tools
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='RAGHealthBot/1.0 (contact: prachimudholkar0408@gmail.com)'
)
summarizer = pipeline("summarization", model="google/flan-t5-base")
symptom_keywords = ["fatigue", "cough", "pain", "fever", "shortness of breath", "nausea", "headache"]
treatment_keywords = ["metformin", "insulin", "antibiotics", "ACE inhibitors", "beta blockers", "chemotherapy"]

def extract_keywords(text, keywords):
    return ", ".join([k for k in keywords if k.lower() in text.lower()]) or "Not found"

# Collect entries
data = []
for condition in icd_mapping:
    page = wiki.page(condition)
    if not page.exists():
        continue

    text = page.text[:2000]  # Limit to ~2k chars
    try:
        summary = summarizer(text, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
        symptoms = extract_keywords(text, symptom_keywords)
        treatments = extract_keywords(text, treatment_keywords)

        data.append({
            "condition": condition,
            "summary": summary,
            "symptoms": symptoms,
            "treatments": treatments,
            "ICD_code": icd_mapping[condition],
            "source": f"https://en.wikipedia.org/wiki/{condition.replace(' ', '_')}"
        })
    except Exception as e:
        print(f"Error with {condition}: {e}")

# Save
df = pd.DataFrame(data)
df.to_csv("enriched_medical_corpus.csv", index=False)
print("✅ CSV saved: enriched_medical_corpus.csv")
#----------------------------------------------------------------
# --- Load and embed corpus ---
df = pd.read_csv("enriched_medical_corpus.csv")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus_texts = (
    df["summary"].fillna("") + " Symptoms: " + df["symptoms"].fillna("") +
    " Treatments: " + df["treatments"].fillna("")
).tolist()
corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=False).astype("float32")

# --- Build FAISS index ---
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# --- Load generation model (FLAN-T5) ---
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# --- RAG Pipeline Function ---
def rag_search_and_answer(query, top_k=3):
    query_vec = embedder.encode([query], convert_to_tensor=False).astype("float32")
    D, I = index.search(query_vec, k=top_k)
    retrieved_entries = df.iloc[I[0]]

    # Build context
    context = "\n\n".join(
        f"Condition: {row['condition']}\nSummary: {row['summary']}\nSymptoms: {row['symptoms']}\nTreatments: {row['treatments']}"
        for _, row in retrieved_entries.iterrows()
    )

    prompt = f"""You are a medical assistant. Use the clinical information below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context

log_file = "rag_query_logs.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "answer", "warning", "retrieved_conditions"])

# --- RAG Function with Robust Error Handling ---
def rag_search_and_answer(query, top_k=3):
    try:
        query_vec = embedder.encode([query], convert_to_tensor=False).astype("float32")
        D, I = index.search(query_vec, k=top_k)

        if not I[0].size or np.mean(D[0]) > 1.5:
            warning = "⚠️ Retrieved documents may not be relevant. Please review with caution."
        else:
            warning = "✅ Retrieval confidence is acceptable."

        retrieved_entries = df.iloc[I[0]]
        retrieved_conditions = ", ".join(retrieved_entries["condition"].tolist())

        # Build context
        context = "\n\n".join(
            f"Condition: {row['condition']}\nSummary: {row['summary']}\nSymptoms: {row['symptoms']}\nTreatments: {row['treatments']}"
            for _, row in retrieved_entries.iterrows()
        )

        prompt = f"""You are a medical assistant. Use the clinical information below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=256)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log result
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([query, answer, warning, retrieved_conditions])

        return answer, context + f"\n\n{warning}"

    except Exception as e:
        logging.exception("❌ Error occurred during processing.")
        return f"❌ Error: {str(e)}", "No context due to error."

# --- Gradio Interface ---
gr.Interface(
    fn=rag_search_and_answer,
    inputs=gr.Textbox(label="Enter a clinical question"),
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Top Retrieved Medical Knowledge + Warning")
    ],
    title="RAG Clinical Assistant (FAISS + LLM)",
    description="Ask a medical question and receive an answer using retrieval-augmented generation with warnings and logging."
).launch()

