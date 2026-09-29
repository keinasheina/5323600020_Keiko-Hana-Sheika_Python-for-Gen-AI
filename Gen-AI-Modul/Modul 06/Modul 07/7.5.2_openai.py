from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi client menggunakan OpenRouter
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    response_format={"type": "json_object"},  # enforces valid JSON[cite: 4]
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}""",
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla.",
        },
    ],
)

result = json.loads(response.choices[0].message.content)
print(result)
# Output: {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}