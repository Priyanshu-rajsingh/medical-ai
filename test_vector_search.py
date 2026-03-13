from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="vector_db")

collection = client.get_collection("medical_knowledge")

query = "mosquito disease with fever"

embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=2
)

print(results["documents"])