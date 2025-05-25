# Aswanthlal_KCCQueryAssistant

# 🔍 KCC Query Assistant

A conversational search assistant built with **Streamlit**, **ChromaDB**, and **SentenceTransformers** to allow natural language querying over the KCC dataset.

---

## 📦 Project Structure

```
KccAssignment/
├── app.py                    # Main Streamlit application
├── rag_pipeline.py          # RAG logic: loading, searching, embedding
├── preprocessed_kcc_data.jsonl  # JSONL file with cleaned Q&A data
├── chroma_db/               # Persistent ChromaDB vector store
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🚀 Features

- 🔎 **Semantic Search** over Q&A pairs using `sentence-transformers`
- ⚡ **GPU-accelerated embeddings** with PyTorch (if available)
- 🧠 RAG-style prompt formatting for embedding ingestion
- 🧾 Search results ranked by cosine similarity
- 💾 Persistent local vector database using `ChromaDB`

---

## 🔧 Installation

> Python 3.9+ recommended

### 1. Clone the repository

```bash
git clone https://github.com/your-username/kcc-assistant.git
cd kcc-assistant
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Ensure you have PyTorch with GPU support:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧠 First-Time Setup: Embed Your Data

To embed the KCC data and populate ChromaDB, run:

```bash
python rag_pipeline.py
```

This will:
- Load `preprocessed_kcc_data.jsonl`
- Generate embeddings using `sentence-transformers/all-mpnet-base-v2`
- Store vectors in local `chroma_db/` directory

---

## 💬 Launch the Assistant

```bash
streamlit run app.py
```

Then open the URL shown in terminal (e.g. `http://localhost:8501`) in your browser.

---

## ⚙️ Configuration

- Modify `rag_pipeline.py` to control:
  - Embedding model
  - Data size limit
  - Batch size for encoding

- Modify `app.py` to change:
  - Page layout
  - Number of top results

---

## 📚 Dependencies

Listed in `requirements.txt`, including:

- `streamlit`
- `chromadb`
- `sentence-transformers`
- `torch`
- `tqdm`

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📎 Notes

- Requires `preprocessed_kcc_data.jsonl` to be formatted with:
  ```json
  {"question": "What is ...?", "answer": "...", "metadata": {...}}
  ```
- Use GPU to accelerate embedding generation.
- Vector DB is persistent — delete `chroma_db/` to reset.

---

## 🛠️ Future Improvements

- [ ] Add live chatbot integration (e.g. OpenAI or Llama)
- [ ] UI filtering by tags/metadata
- [ ] Support uploading new data through the interface

---

## 📄 License

MIT License. See `LICENSE` file for details.
