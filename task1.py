import os

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Configuration

DOCUMENTS_FOLDER = "./documents"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 2. Load the Sentence Transformer model
print("Loading embedding model...")
model = SentenceTransformer(
    EMBEDDING_MODEL
)
print("Embedding model loaded successfully.")

# 3. Create persistent ChromaDB client
print("Creating ChromaDB client...")
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

# 4. Create or get ChromaDB collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)
print(
    f"ChromaDB collection '{COLLECTION_NAME}' is ready."
)

# 5. Read the documents
documents = []
ids = []
metadatas = []

# Get all files from documents folder
file_names = sorted(
    os.listdir(DOCUMENTS_FOLDER)
)

for file_name in file_names:
    # Process only text files
    if not file_name.lower().endswith(".txt"):
        continue

    file_path = os.path.join(
        DOCUMENTS_FOLDER,
        file_name
    )
    
    # Read document
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()


    # Skip empty documents

    if not text:
        print(
            f"Skipping empty document: {file_name}"
        )
        continue

    # One document = one chunk

    chunk = text

    # Create unique ID

    chunk_id = (
        f"{file_name}_chunk_0"
    )

    # Store information

    documents.append(chunk)
    ids.append(chunk_id)
    metadatas.append(
        {
            "source": file_name,
            "chunk_number": 0
        }
    )


    print(
        f"Processed: {file_name}"
    )


# 6. Check whether documents were found

if not documents:

    raise ValueError(
        "No .txt documents were found "
        "inside the documents folder."
    )


print()
print(
    f"Total documents/chunks: {len(documents)}"
)


# 7. Generate embeddings

print()
print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)


print("Embeddings generated successfully.")


# 8. Convert embeddings to Python lists

embeddings = embeddings.tolist()

# 9. Store documents, embeddings and metadata in ChromaDB

print()
print("Storing data in ChromaDB...")

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# 10. Verify collection

print()
print("============================================")
print("TASK 1 COMPLETED SUCCESSFULLY")
print("============================================")

print(
    "Collection name:",
    collection.name
)

print(
    "Total chunks stored:",
    collection.count()
)

print(
    "ChromaDB location:",
    CHROMA_DB_PATH
)