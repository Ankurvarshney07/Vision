import streamlit as st
import base64
from groq import Groq
import os
from PIL import Image
# import io
import requests
from io import BytesIO

# Set up Groq API key (replace with your actual key)
# GROQ_API_KEY = "your_groq_api_key"

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to encode the image as base64 (useful if needed to send back with the response)
def encode_image(image):
    """Converts an image to base64 encoding."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save in PNG format for consistency
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Streamlit UI Setup
st.set_page_config(page_title="Text to Image Generator", layout="centered")
st.title("🎨 Text to Image Generator")

# Text input for the user prompt
prompt = st.text_input("Enter your prompt", "A fantasy landscape with mountains and a river.")

# API Call to generate the image when button is pressed
if st.button("Generate Image"):
    if client and prompt:
        try:
            st.spinner("Generating image...")

            # Sending the prompt to the Groq model via chat completion to generate an image description
            response = client.chat.completions.create(
                model="llava-v1.5-7b-4096-preview",  # Replace with the correct model name from Groq
                messages=[
                    {"role": "system", "content": "You are an image generation assistant."},
                    {"role": "user", "content": f"Generate an image based on this description: {prompt}"}
                ]
            )

            # Debug: print the response to see its structure
            st.write("Response from Groq:", response)

            # Ensure the response is correctly structured
            if hasattr(response, 'choices') and len(response.choices) > 0:
                # Correctly access the content inside the 'choices' array
                generated_content = response.choices[0].message.content
                st.subheader("Generated Description / Content:")
                st.write(generated_content)

                # Assuming Groq's response would give us a URL to the generated image
                # If the response provides a URL, you could fetch the image.
                if "http" in generated_content:
                    image_url = generated_content.strip()  # Extract URL from response content (simplified for demonstration)

                    # Fetch the image
                    image = Image.open(BytesIO(requests.get(image_url).content))

                    # Convert the image to base64 (if needed to send back as a part of the response)
                    base64_image = encode_image(image)

                    # Display the generated image
                    st.image(image, caption="Generated Image", use_column_width=True)

                else:
                    st.error("No image URL found in the response content. Please check the API response format.")
            else:
                st.error("No valid choices found in the response.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a valid Groq API key and a prompt.")



