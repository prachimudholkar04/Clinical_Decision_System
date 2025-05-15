# 🏥 RAG Clinical Assistant

**RAG Clinical Assistant** is an AI-powered question-answering system for healthcare. It combines **semantic search (FAISS)** with **large language models (Flan-T5)** to generate context-aware answers to medical questions using a curated Wikipedia-based knowledge base enriched with symptoms, treatments, and ICD-10 codes.

---

## 🧾 Features

- 🔍 Semantic medical knowledge retrieval using FAISS
- 🧠 Contextual answer generation with Hugging Face's Flan-T5
- 📚 Automatic corpus building from Wikipedia + ICD-10 mapping
- ⚠️ Warning system for low-confidence retrieval
- 📝 Feedback logging for active learning
- 💬 Gradio interface for interactive Q&A

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/rag-clinical-assistant.git
cd rag-clinical-assistant
pip install -r requirements.txt
