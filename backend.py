from fastapi import FastAPI
import chromadb
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

app = FastAPI()

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# VECTOR DATABASE
# -----------------------------
chroma_client = chromadb.PersistentClient(path="vector_db")
collection = chroma_client.get_collection("medical_knowledge")

# -----------------------------
# DISEASE PREDICTION MODEL
# -----------------------------
disease_model = pickle.load(open("models/disease_model.pkl","rb"))

train_df = pd.read_csv("datasets/Training.csv")
symptoms = list(train_df.columns[:-1])

# -----------------------------
# GROQ CLIENT
# -----------------------------
import os
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# RETRIEVE MEDICAL CONTEXT
# -----------------------------
def retrieve_context(query):

    embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )

    docs = results["documents"][0]

    return "\n".join(docs)

# -----------------------------
# DISEASE PREDICTION
# -----------------------------
def predict_disease(text):

    text = text.lower()

    detected = []

    for symptom in symptoms:
        clean = symptom.replace("_"," ")
        if clean in text:
            detected.append(symptom)

    input_vector = [0]*len(symptoms)

    for s in detected:
        idx = symptoms.index(s)
        input_vector[idx] = 1

    if sum(input_vector) == 0:
        return None

    prediction = disease_model.predict([input_vector])[0]

    return prediction

# -----------------------------
# LLM CALL
# -----------------------------
def ask_llm(context, question):

    prompt = f"""
You are a helpful medical assistant.

Medical Knowledge:
{context}

User Question:
{question}

Explain clearly and suggest precautions.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message.content

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.get("/chat")

def chat(query:str):

    prediction = predict_disease(query)

    context = retrieve_context(query)

    answer = ask_llm(context, query)

    return {
        "prediction": prediction,
        "answer": answer
    }