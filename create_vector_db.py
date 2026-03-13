from sentence_transformers import SentenceTransformer
import chromadb

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# read knowledge file
with open("medical_knowledge.txt","r",encoding="utf-8") as f:
    data = f.read()

documents = data.split("-----------------")

# create persistent database
client = chromadb.PersistentClient(path="vector_db")

collection = client.get_or_create_collection(
    name="medical_knowledge"
)

for i, doc in enumerate(documents):

    doc = doc.strip()

    if doc == "":
        continue

    embedding = model.encode(doc).tolist()

    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[str(i)]
    )

print("Vector database created successfully")