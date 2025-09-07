import openai
import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def generate_caption(topic, tone):
    prompt = (
        f"Write an engaging Instagram caption about '{topic}' in a {tone.lower()} tone. "
        "Add 5-10 relevant hashtags and emojis. Keep it under 2200 characters. "
        "Include a short call-to-action at the end."
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.8
    )

    caption = response.choices[0].message.content
    return caption.strip()
