import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class MoodboardAgent:
    def run(self, description):
        print(f"[Agent] Generating hashtags for: {description}")

        prompt = (
            f"Bir moda gönderisi için '{description}' açıklamasına uygun ve benzer, "
            f"bu içeriği aramayı kolaylaştıracak 5 kısa içerik etiketi öner. "
            f"Sadece etiketleri sırayla ver, açıklama ekleme."
        )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )

        hashtags = response.choices[0].message.content
        hashtags = response.choices[0].message.content
        print("\n[Agent] Suggested Hashtags:\n")
        print(hashtags)
        return hashtags

if __name__ == "__main__":
    desc = input("Aradığın ürünü gir: ")
    desc = input("Aradığın ürünü gir: ")
    agent = MoodboardAgent()
    agent.run(desc)