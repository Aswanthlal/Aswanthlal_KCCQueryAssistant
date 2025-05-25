from sentence_transformers import SentenceTransformer
import json
from chromadb import PersistentClient

# Load embedding model
model_name = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = SentenceTransformer(model_name)

# Setup ChromaDB client
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="kcc_data_1_8th")

input_file = "preprocessed_kcc_data_8.jsonl"

docs = []
metadatas = []
ids = []

max_valid_records = 10000  # Limit to 10k valid entries
valid_count = 0
record_idx = 0  # Unique ID generator

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line {record_idx}")
            record_idx += 1
            continue

        text = f"Q: {record['question']} A: {record['answer']}"
        metadata = record.get('metadata', {})

        docs.append(text)
        metadatas.append(metadata)
        ids.append(str(record_idx))

        valid_count += 1
        record_idx += 1

        if valid_count >= max_valid_records:
            break

print(f"✔️ Loaded {valid_count} valid records.")
print("🔄 Generating embeddings...")
embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=32)

print("⬆️ Ingesting into ChromaDB...")
collection.add(
    documents=docs,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print("✅ Embedding generation and ingestion complete.")
print(f"📊 ChromaDB now contains {collection.count()} documents.")
