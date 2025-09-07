# import streamlit as st
# from utils.content_gen import generate_caption
# from utils.image_gen import generate_image

# st.set_page_config(page_title="Instagram Post Generator", layout="centered")
# st.title("📸 Auto Instagram Post Generator")

# topic = st.text_input("Enter your post topic (e.g., 'Monday Motivation')")
# tone = st.selectbox("Select a tone", ["Friendly", "Motivational", "Professional", "Playful"])
# style = st.selectbox("Select image style", ["Realistic", "Bold Text Overlay", "Illustration", "Minimalist"])

# if st.button("Generate Post"):
#     with st.spinner("Creating your caption and image..."):
#         caption = generate_caption(topic, tone)
#         image_url = generate_image(topic, style)

#         st.image(image_url, caption="Generated Image", use_column_width=True)
#         st.text_area("📄 Instagram Caption", caption, height=200)
#         st.download_button("Download Caption", caption, file_name="caption.txt")




import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from utils.content_gen import generate_caption
from utils.image_gen import generate_image

st.set_page_config(page_title="Instagram Post Generator", layout="centered")
st.title("📸 Auto Instagram Post Generator")

topic = st.text_input("Enter your post topic (e.g., 'Monday Motivation')")
tone = st.selectbox("Select a tone", ["Friendly", "Motivational", "Professional", "Playful"])
style = st.selectbox("Select image style", ["Realistic", "Bold Text Overlay", "Illustration", "Minimalist"])

def display_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        st.image(image, caption="Generated Image", use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Failed to load image: {e}")

if st.button("Generate Post"):
    with st.spinner("Creating your caption and image..."):
        caption = generate_caption(topic, tone)
        image_url = generate_image(topic, style)

        display_image_from_url(image_url)
        st.text_area("📄 Instagram Caption", caption, height=200)
        st.download_button("Download Caption", caption, file_name="caption.txt")
