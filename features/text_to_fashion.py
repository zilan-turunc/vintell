
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# Create OpenAI client from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def suggest_fashion_items(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo" if you want cheaper
        messages=[
            {"role": "system", "content": "You are a personal stylist and fashion assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

