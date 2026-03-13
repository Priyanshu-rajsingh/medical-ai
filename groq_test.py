from groq import Groq

try:
    client = Groq(api_key="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")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Explain dengue fever"}
        ]
    )

    print("RAW RESPONSE:")
    print(response)

    print("\nMODEL ANSWER:")
    print(response.choices[0].message.content)

except Exception as e:
    print("ERROR:", e)