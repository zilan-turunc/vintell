import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

class MoodboardAgent:
    def run(self, description):
        print(f"[Agent] Generating metadata for: {description}")

        prompt = (
            f"Given the following fashion item description:\n\n'{description}'\n\n"
            f"Generate structured metadata with the following fields:\n"
            f"- Category: (1–2 words)\n"
            f"- Style Tags: (exactly 5 short hashtags)\n"
            f"- Occasions: (brief list of where/when to wear it)\n"
            f"- Pairing Suggestions: (short list of items it pairs well with)\n\n"
            f"Format:\n"
            f"Category: <text>\n"
            f"Style Tags: <#tag1 #tag2 #tag3 #tag4 #tag5>\n"
            f"Occasions: <comma-separated>\n"
            f"Pairing Suggestions: <comma-separated>\n\n"
            f"Do not include extra commentary or explanation."
        )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=150
        )

        metadata = response.choices[0].message.content.strip()
        print("\n[Agent] Structured Metadata:\n")
        print(metadata)
        return metadata

if __name__ == "__main__":
    desc = input("Enter your product description:\n> ")
    agent = MoodboardAgent()
    agent.run(desc)
