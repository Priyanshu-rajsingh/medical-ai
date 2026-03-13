import requests
import chromadb
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# CONNECT VECTOR DATABASE
# -----------------------------

client = chromadb.PersistentClient(path="vector_db")

collection = client.get_collection("medical_knowledge")

# -----------------------------
# LOAD DISEASE PREDICTION MODEL
# -----------------------------

disease_model = pickle.load(open("models/disease_model.pkl", "rb"))

train_df = pd.read_csv("datasets/Training.csv")

symptoms = list(train_df.columns[:-1])

# -----------------------------
# VECTOR SEARCH FUNCTION
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
# DISEASE PREDICTION FROM TEXT
# -----------------------------

def predict_disease_from_text(text):

    text = text.lower()

    detected = []

    for symptom in symptoms:
        clean = symptom.replace("_", " ")

        if clean in text:
            detected.append(symptom)

    input_vector = [0] * len(symptoms)

    for s in detected:
        idx = symptoms.index(s)
        input_vector[idx] = 1

    if sum(input_vector) == 0:
        return None

    prediction = disease_model.predict([input_vector])[0]

    return prediction


# -----------------------------
# LLM FUNCTION
# -----------------------------

def ask_llm(context, question):

    prompt = f"""
You are a helpful medical assistant.

Use the medical knowledge below to answer the user.

Medical Knowledge:
{context}

User Question:
{question}

Explain the condition and suggest precautions if relevant.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "response" not in data:
        print("Ollama returned:", data)
        return "Model failed to generate response."

    return data["response"]


# -----------------------------
# MAIN CHAT LOOP
# -----------------------------

def medical_chat():

    print("\nMedical AI Assistant Ready")
    print("Type 'exit' to stop\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # disease prediction
        prediction = predict_disease_from_text(user_input)

        if prediction:
            print("\nPredicted Disease:", prediction)

        # retrieve knowledge
        context = retrieve_context(user_input)

        # ask LLM
        answer = ask_llm(context, user_input)

        print("\nAI:", answer)
        print()


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    medical_chat()