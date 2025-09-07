import streamlit as st
import base64
from groq import Groq
import os
from PIL import Image
import io

# Set up Groq API key (replace with your actual key)
# GROQ_API_KEY = "your_groq_api_key"

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # Streamlit UI
# st.set_page_config(page_title="Image to Text - Cheetah🐆 Vision", layout="centered")
# st.title("🖼️ Image to Text using Cheetah🐆 Vision API")

# # File uploader
# uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# # Function to encode image as base64
# def encode_image(image_file):
#     return base64.b64encode(image_file.read()).decode('utf-8')

# if uploaded_file:
#     # Convert image to base64
#     base64_image = encode_image(uploaded_file)
    
#     # Display uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=False)

#     # API call button
#     if st.button("Generate Text"):
#         with st.spinner("Processing..."):
#             try:
#                 # Send request to Groq API
#                 response = client.chat.completions.create(
#                     model="llama-3.2-11b-vision-preview",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": [
#                                 # {"type": "text", "text": "convert it into a LaTeX equation."},
#                                 {"type": "text", "text": "Describe this Image."},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                                 },
#                             ],
#                         }
#                     ],
#                 )

#                 # Extract and display response
#                 generated_text = response.choices[0].message.content
#                 st.subheader("📝 Generated Text:")
#                 st.write(generated_text)

#             except Exception as e:
#                 st.error(f"❌ Error: {str(e)}")




# # Streamlit UI
# st.set_page_config(page_title="Math/Science Equation to LaTeX", layout="centered")
# st.title("🧮 Image to LaTeX Converter (Math, Chemistry, Physics)")

# # File uploader
# uploaded_file = st.file_uploader("Upload an image of an equation...", type=["jpg", "png", "jpeg"])

# # Function to encode image as base64
# def encode_image(image_file):
#     return base64.b64encode(image_file.read()).decode('utf-8')

# if uploaded_file:
#     # Get file size
#     file_size = uploaded_file.size  # Size in bytes

#     # Display uploaded image
#     st.image(uploaded_file, caption="Uploaded Equation", use_column_width=True)

#     # Allow small images (as math equations may be small)
#     if file_size < 700:  # Less than 700 bytes
#         st.warning("⚠️ Image is very small. The output may be inaccurate.")

#     # API call button
#     if st.button("Convert to LaTeX"):
#         with st.spinner("Processing..."):
#             try:
#                 # Convert image to base64
#                 base64_image = encode_image(uploaded_file)

#                 # Send request to Groq API
#                 response = client.chat.completions.create(
#                     model="llama-3.2-11b-vision-preview",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": "Extract the raw LaTeX code from this equation."},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                                 },
#                             ],
#                         }
#                     ],
#                 )

#                 # Extract LaTeX response
#                 latex_equation = response.choices[0].message.content.strip()

#                 # Display the raw LaTeX code
#                 st.subheader("📜 Extracted LaTeX Equation:")
#                 st.code(latex_equation, language="latex")

#                 # Render LaTeX output
#                 st.subheader("📌 Rendered Equation:")
#                 st.latex(latex_equation)

#             except Exception as e:
#                 st.error(f"❌ Error: {str(e)}")




st.set_page_config(page_title="Math/Science Equation to LaTeX", layout="centered")
st.title("🧮 Image to LaTeX Converter (Math, Chemistry, Physics)")

# File uploader
uploaded_file = st.file_uploader("Upload an image of an equation...", type=["jpg", "png", "jpeg", "gif"])

# Function to encode image as base64
def encode_image(image):
    """Converts an image to base64 encoding."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # Save in PNG format for consistency
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if uploaded_file:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Convert GIF to PNG (if necessary)
    if uploaded_file.type == "image/gif":
        st.warning("⚠️ GIF detected! Converting to PNG for processing...")
        image = image.convert("RGB")  # Convert to a non-animated format

    # Get file size
    file_size = uploaded_file.size  # Size in bytes

    # Display uploaded image
    st.image(image, caption="Uploaded Equation", use_column_width=True)

    # Allow small images (as math equations may be small)
    if file_size < 700:  # Less than 700 bytes
        st.warning("⚠️ Image is very small. The output may be inaccurate.")

    # API call button
    if st.button("Convert to LaTeX"):
        with st.spinner("Processing..."):
            try:
                # Convert image to base64
                base64_image = encode_image(image)

                # Identify MIME type dynamically
                image_format = uploaded_file.type.split("/")[-1]  # Extract format from MIME type
                mime_type = f"image/{'png' if image_format == 'gif' else image_format}"

                # Send request to Groq API
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract only the LaTeX expression from this image."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                )

                # Extract LaTeX response
                latex_equation = response.choices[0].message.content.strip()

                # Display the raw LaTeX code
                st.subheader("📜 Extracted LaTeX Equation:")
                st.code(latex_equation, language="latex")

                # Render LaTeX output
                st.subheader("📌 Rendered Equation:")
                st.latex(latex_equation)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")