# import streamlit as st
# import requests

# st.set_page_config(page_title="College Q&A Chatbot", layout="centered")
# st.title("🎓 College Chatbot (Groq + FAISS)")
# st.markdown("Ask about courses, colleges, exams etc.")

# question = st.text_input("Ask your question:")

# if st.button("Submit") and question:
#     with st.spinner("Thinking..."):
#         res = requests.post("http://localhost:8000/ask", json={"question": question})
#         st.success("Answer received!")
#         st.write("### 💡 Answer")
#         st.write(res.json()["answer"])



import streamlit as st
import requests

st.set_page_config(page_title="🎓 College AI Chatbot", layout="centered")

# ---- Title ----
st.title("🎓 AskCareer360 - College AI Chatbot")
st.markdown("Ask anything about Indian colleges, courses, entrance exams, fees, and more.")

# ---- Chat history ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Sidebar with preset topics ----
with st.sidebar:
    st.header("🎯 Quick Questions")
    if st.button("Top Engineering Colleges"):
        st.session_state.chat_history.append(("user", "What are the top engineering colleges in India?"))
    if st.button("Top AI Courses"):
        st.session_state.chat_history.append(("user", "Which colleges offer the best AI and ML courses?"))
    if st.button("Affordable B.Tech Programs"):
        st.session_state.chat_history.append(("user", "Suggest affordable B.Tech programs with good placements."))
    if st.button("Exams for Engineering"):
        st.session_state.chat_history.append(("user", "What are the main exams for engineering admissions in India?"))

# ---- Chat Box ----
user_input = st.text_input("💬 Type your question here:")

if st.button("Ask"):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))

# ---- Display & process chat ----
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"👤 **You:** {msg}")
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://localhost:8000/ask", json={"question": msg})
                answer = response.json()["answer"]
            except Exception as e:
                answer = f"❌ Failed to connect to backend: {e}"
        st.session_state.chat_history.append(("bot", answer))
        st.markdown(f"🤖 **Bot:** {answer}")
    elif role == "bot":
        st.markdown(f"🤖 **Bot:** {msg}")

# ---- Reset ----
st.markdown("---")
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
