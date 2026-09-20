import os
from dotenv import load_dotenv
from groq import Groq

from src.config import GROQ_MODEL

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "user", "content": "Reply with exactly this: pipeline ready."}
    ],
    temperature=0
)

print(response.choices[0].message.content)