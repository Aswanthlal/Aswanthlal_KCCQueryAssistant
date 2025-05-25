from sentence_transformers import SentenceTransformer
import json
from chromadb import PersistentClient

# Load your embedding model
model_name = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = SentenceTransformer(model_name)

# Setup persistent ChromaDB client (use your folder for persistence)
client = PersistentClient(path="./chroma_db")

# Create or load collection (no embedding_function needed because embeddings provided)
collection = client.get_or_create_collection(name="kcc_data_1_8th")

# Your input JSONL file with preprocessed KCC data
input_file = "preprocessed_kcc_data_8.jsonl"

docs = []
metadatas = []
ids = []

# Load data from JSONL file
import json

with open(input_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue  # skip empty lines

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line {idx}: {e}")
            continue

        text = f"Q: {record['question']} A: {record['answer']}"
        metadata = record.get('metadata', {})

        docs.append(text)
        metadatas.append(metadata)
        ids.append(str(idx))



print("Generating embeddings...")
embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=32)

print("Ingesting into ChromaDB...")
collection.add(
    documents=docs,
    embeddings=embeddings.tolist(),  # convert to list if needed
    metadatas=metadatas,
    ids=ids
)

print("✅ Embedding generation and ingestion complete.")
print(f"📊 ChromaDB now contains {collection.count()} documents.")
