import pandas as pd

# load datasets
desc = pd.read_csv("datasets/disease_description.csv")
prec = pd.read_csv("datasets/disease_precaution.csv")

documents = []

for i in range(len(desc)):
    
    disease = desc.iloc[i,0]
    description = desc.iloc[i,1]

    precautions = prec.iloc[i,1:].dropna().tolist()

    text = f"""
Disease: {disease}

Description:
{description}

Precautions:
"""

    for p in precautions:
        text += f"- {p}\n"

    documents.append(text)


# save to file
with open("medical_knowledge.txt","w",encoding="utf-8") as f:
    for doc in documents:
        f.write(doc)
        f.write("\n-----------------\n")

print("Medical documents created successfully")