import openai
import os
from dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# def generate_image(topic, style):
#     prompt = f"Instagram post image about '{topic}' in a {style.lower()} style, vibrant and high-quality."

#     # response = client.chat.completions.create(
#     #     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     #     prompt=prompt,
#     #     n=1,
#     #     size="1024x1024"
#     # )
#     # return response['data'][0]['url']

#     response = client.chat.completions.create(
#         model="whisper-large-v3-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content.strip()





def generate_image(topic, style, resolution="1024x1024"):
    # Step 1: Generate a detailed image description using Groq
    prompt = f"Instagram post image about '{topic}' in a {style.lower()} style, vibrant and high-quality."
    groq_response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    image_prompt = groq_response.choices[0].message.content.strip()

    # Step 2: Generate an actual image from the prompt using Eden AI
    eden_response = requests.post(
        "https://api.edenai.run/v2/image/generation",
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNGI5MDE1NWYtMzllZi00MTc4LThlMTktOWQwMjZiMzlkNTcyIiwidHlwZSI6ImFwaV90b2tlbiJ9.TNntuW1snUs-l7MdiF_cVkOoljLnw-JMU5iJfPhRK-U"},
        json={
            "providers": "openai",  # or stabilityai / replicate / deepai
            "text": image_prompt,
            "resolution": resolution
        }
    )

    eden_response.raise_for_status()
    data = eden_response.json()

    try:
        image_url = data["openai"]["items"][0]["image"]
        return image_url
    except (KeyError, IndexError):
        raise ValueError("Failed to generate image from Eden AI")
    