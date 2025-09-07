# # # app.py
# # import os
# # import streamlit as st
# # from dotenv import load_dotenv
# # import PyPDF2
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_openai import OpenAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_openai import OpenAI
# # from langchain.chains.question_answering import load_qa_chain
# # import time

# # # Load environment variables from .env file
# # load_dotenv()

# # def get_pdf_text(pdf_docs):
# #     """
# #     Extracts text from a list of PDF documents.

# #     Args:
# #         pdf_docs (list): A list of uploaded PDF files.

# #     Returns:
# #         str: The concatenated text from all PDF documents.
# #     """
# #     text = ""
# #     for pdf in pdf_docs:
# #         try:
# #             pdf_reader = PyPDF2.PdfReader(pdf)
# #             for page in pdf_reader.pages:
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     text += page_text
# #         except Exception as e:
# #             st.error(f"Error reading {pdf.name}: {e}")
# #     return text

# # def get_text_chunks(text):
# #     """
# #     Splits a long text into smaller chunks.

# #     Args:
# #         text (str): The input text.

# #     Returns:
# #         list: A list of text chunks.
# #     """
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000,
# #         chunk_overlap=200,
# #         length_function=len
# #     )
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # def get_vector_store(text_chunks):
# #     """
# #     Creates a FAISS vector store from text chunks.

# #     Args:
# #         text_chunks (list): A list of text chunks.
# #     """
# #     try:
# #         embeddings = OpenAIEmbeddings()
# #         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #         st.session_state.vector_store = vector_store
# #         st.success("Vector store created successfully!")
# #     except Exception as e:
# #         st.error(f"Error creating vector store: {e}")
# #         st.error("Please ensure your OpenAI API key is correct and has credit.")

# # def get_conversational_chain():
# #     """
# #     Loads a question-answering chain.

# #     Returns:
# #         A loaded question-answering chain.
# #     """
# #     llm = OpenAI(temperature=0.7)
# #     chain = load_qa_chain(llm, chain_type="stuff")
# #     return chain

# # def user_input(user_question):
# #     """
# #     Handles user input, performs a similarity search, and gets the answer.

# #     Args:
# #         user_question (str): The question asked by the user.
# #     """
# #     if "vector_store" not in st.session_state or st.session_state.vector_store is None:
# #         st.warning("Please upload and process your PDFs first.")
# #         return

# #     try:
# #         # Perform similarity search
# #         with st.spinner("Searching for relevant information..."):
# #             docs = st.session_state.vector_store.similarity_search(user_question)

# #         # Get the answer from the chain
# #         with st.spinner("Generating answer..."):
# #             chain = get_conversational_chain()
# #             response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# #         # Display the answer with a typewriter effect
# #         st.subheader("Answer:")
# #         answer_placeholder = st.empty()
# #         full_answer = response["output_text"]
# #         displayed_answer = ""
# #         for char in full_answer:
# #             displayed_answer += char
# #             answer_placeholder.markdown(displayed_answer + "▌")
# #             time.sleep(0.01)
# #         answer_placeholder.markdown(displayed_answer)

# #     except Exception as e:
# #         st.error(f"An error occurred: {e}")
# #         st.error("This could be due to an issue with the OpenAI API. Please check your key and usage limits.")


# # def main():
# #     """
# #     Main function to run the Streamlit app.
# #     """
# #     st.set_page_config(page_title="GenAI Ebook Q&A Assistant", page_icon="📚", layout="wide")

# #     # Custom CSS for styling that works with both light and dark themes
# #     st.markdown("""
# #         <style>
# #             /* General styling */
# #             .st-emotion-cache-1y4p8pa {
# #                 max-width: 100%;
# #             }

# #             /* Button styling */
# #             .stButton>button {
# #                 border-radius: 12px;
# #                 padding: 10px 24px;
# #                 border: none;
# #                 transition: background-color 0.3s, transform 0.1s;
# #                 font-weight: 600;
# #             }
# #             .stButton>button:hover {
# #                 transform: scale(1.02);
# #             }
# #             .stButton>button:active {
# #                 transform: scale(0.98);
# #             }

# #             /* Specific button colors */
# #             div[data-testid="stSidebarUserContent"] .stButton>button {
# #                 background-color: #008CBA; /* Blue for process button */
# #                 color: white;
# #             }
# #             div[data-testid="stSidebarUserContent"] .stButton>button:hover {
# #                 background-color: #007399;
# #             }

# #             /* Input field styling */
# #             .stTextInput>div>div>input {
# #                 border-radius: 8px;
# #             }

# #             /* File uploader styling */
# #             .stFileUploader>div>div>button {
# #                 border-radius: 12px;
# #                 border: 2px dashed #4CAF50;
# #                 background-color: transparent;
# #                 color: #4CAF50;
# #             }
# #             .stFileUploader>div>div>button:hover {
# #                 border-color: #45a049;
# #                 color: #45a049;
# #             }

# #             /* Custom container for a card-like effect */
# #             .card {
# #                 background-color: #FFFFFF20; /* Semi-transparent white for adaptability */
# #                 padding: 20px;
# #                 border-radius: 15px;
# #                 margin-bottom: 20px;
# #                 box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
# #             }
# #         </style>
# #     """, unsafe_allow_html=True)

# #     # Header
# #     st.header("GenAI Ebook Q&A Assistant 📚")
# #     st.write("Upload your ebooks in PDF format, and ask any question about their content.")

# #     # Sidebar for PDF upload
# #     with st.sidebar:
# #         st.subheader("Your Ebooks")
# #         pdf_docs = st.file_uploader(
# #             "Upload your PDF files here",
# #             accept_multiple_files=True,
# #             type="pdf"
# #         )
# #         if st.button("Process Documents"):
# #             if pdf_docs:
# #                 with st.spinner("Processing PDFs... This may take a moment."):
# #                     # 1. Get PDF text
# #                     raw_text = get_pdf_text(pdf_docs)
# #                     if not raw_text:
# #                         st.error("Could not extract text from the uploaded PDF(s). Please try other files.")
# #                         return

# #                     # 2. Get text chunks
# #                     text_chunks = get_text_chunks(raw_text)
# #                     if not text_chunks:
# #                         st.error("Could not split the text into chunks.")
# #                         return

# #                     # 3. Create vector store
# #                     get_vector_store(text_chunks)
# #             else:
# #                 st.warning("Please upload at least one PDF file.")

# #     # Main content area for Q&A
# #     st.subheader("Ask a Question")
# #     user_question = st.text_input("What would you like to know from your ebooks?", placeholder="e.g., What are the main themes of the book?", key="user_question")

# #     if user_question:
# #         user_input(user_question)

# #     # Instructions and footer
# #     st.markdown("---")
# #     with st.expander("How to use this app"):
# #         st.info("""
# #             1.  **Upload**: Drag and drop one or more PDF files into the uploader in the sidebar.
# #             2.  **Process**: Click the 'Process Documents' button and wait for the confirmation message.
# #             3.  **Ask**: Type your question in the text box and press Enter. The AI will search the documents and generate an answer for you.
# #         """)

# # if __name__ == "__main__":
# #     main()






# # app.py
# import os
# import streamlit as st
# from dotenv import load_dotenv
# import PyPDF2
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# from langchain.chains.question_answering import load_qa_chain
# import time

# # Load environment variables from .env file
# load_dotenv()

# def get_pdf_text(pdf_docs):
#     """
#     Extracts text from a list of PDF documents.

#     Args:
#         pdf_docs (list): A list of uploaded PDF files.

#     Returns:
#         str: The concatenated text from all PDF documents.
#     """
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PyPDF2.PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text
#         except Exception as e:
#             st.error(f"Error reading {pdf.name}: {e}")
#     return text

# def get_text_chunks(text):
#     """
#     Splits a long text into smaller chunks.

#     Args:
#         text (str): The input text.

#     Returns:
#         list: A list of text chunks.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     """
#     Creates a FAISS vector store from text chunks using OpenAI Embeddings.

#     Args:
#         text_chunks (list): A list of text chunks.
#     """
#     try:
#         # Embeddings are still handled by OpenAI as Groq doesn't provide an embedding model
#         embeddings = OpenAIEmbeddings()
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         st.session_state.vector_store = vector_store
#         st.success("Vector store created successfully!")
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         st.error("Please ensure your OpenAI API key is correct and has credit.")

# def get_conversational_chain():
#     """
#     Loads a question-answering chain using the Groq API.

#     Returns:
#         A loaded question-answering chain.
#     """
#     # Using a Groq model for the conversational part
#     llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
#     chain = load_qa_chain(llm, chain_type="stuff")
#     return chain

# def user_input(user_question):
#     """
#     Handles user input, performs a similarity search, and gets the answer.

#     Args:
#         user_question (str): The question asked by the user.
#     """
#     if "vector_store" not in st.session_state or st.session_state.vector_store is None:
#         st.warning("Please upload and process your PDFs first.")
#         return

#     try:
#         # Perform similarity search
#         with st.spinner("Searching for relevant information..."):
#             docs = st.session_state.vector_store.similarity_search(user_question)

#         # Get the answer from the chain
#         with st.spinner("Generating answer with Groq..."):
#             chain = get_conversational_chain()
#             response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)


#         # Display the answer with a typewriter effect
#         st.subheader("Answer:")
#         answer_placeholder = st.empty()
#         full_answer = response["output_text"]
#         displayed_answer = ""
#         for char in full_answer:
#             displayed_answer += char
#             answer_placeholder.markdown(displayed_answer + "▌")
#             time.sleep(0.01)
#         answer_placeholder.markdown(displayed_answer)

#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         st.error("This could be due to an issue with the Groq API. Please check your key and usage limits.")


# def main():
#     """
#     Main function to run the Streamlit app.
#     """
#     st.set_page_config(page_title="GenAI Ebook Q&A Assistant", page_icon="📚", layout="wide")

#     # Check for API keys
#     if not os.getenv("OPENAI_API_KEY"):
#         st.error("OpenAI API key is not set. Please add it to your .env file.")
#     if not os.getenv("GROQ_API_KEY"):
#         st.error("Groq API key is not set. Please add it to your .env file.")


#     # Custom CSS for styling that works with both light and dark themes
#     st.markdown("""
#         <style>
#             /* General styling */
#             .st-emotion-cache-1y4p8pa {
#                 max-width: 100%;
#             }

#             /* Button styling */
#             .stButton>button {
#                 border-radius: 12px;
#                 padding: 10px 24px;
#                 border: none;
#                 transition: background-color 0.3s, transform 0.1s;
#                 font-weight: 600;
#             }
#             .stButton>button:hover {
#                 transform: scale(1.02);
#             }
#             .stButton>button:active {
#                 transform: scale(0.98);
#             }

#             /* Specific button colors */
#             div[data-testid="stSidebarUserContent"] .stButton>button {
#                 background-color: #008CBA; /* Blue for process button */
#                 color: white;
#             }
#             div[data-testid="stSidebarUserContent"] .stButton>button:hover {
#                 background-color: #007399;
#             }

#             /* Input field styling */
#             .stTextInput>div>div>input {
#                 border-radius: 8px;
#             }

#             /* File uploader styling */
#             .stFileUploader>div>div>button {
#                 border-radius: 12px;
#                 border: 2px dashed #4CAF50;
#                 background-color: transparent;
#                 color: #4CAF50;
#             }
#             .stFileUploader>div>div>button:hover {
#                 border-color: #45a049;
#                 color: #45a049;
#             }

#             /* Custom container for a card-like effect */
#             .card {
#                 background-color: #FFFFFF20; /* Semi-transparent white for adaptability */
#                 padding: 20px;
#                 border-radius: 15px;
#                 margin-bottom: 20px;
#                 box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     # Header
#     st.header("GenAI Ebook Q&A Assistant 📚")
#     st.write("Upload your ebooks in PDF format, and ask any question about their content.")

#     # Sidebar for PDF upload
#     with st.sidebar:
#         st.subheader("Your Ebooks")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF files here",
#             accept_multiple_files=True,
#             type="pdf"
#         )
#         if st.button("Process Documents"):
#             if pdf_docs:
#                 with st.spinner("Processing PDFs... This may take a moment."):
#                     # 1. Get PDF text
#                     raw_text = get_pdf_text(pdf_docs)
#                     if not raw_text:
#                         st.error("Could not extract text from the uploaded PDF(s). Please try other files.")
#                         return

#                     # 2. Get text chunks
#                     text_chunks = get_text_chunks(raw_text)
#                     if not text_chunks:
#                         st.error("Could not split the text into chunks.")
#                         return

#                     # 3. Create vector store
#                     get_vector_store(text_chunks)
#             else:
#                 st.warning("Please upload at least one PDF file.")

#     # Main content area for Q&A
#     st.subheader("Ask a Question")
#     user_question = st.text_input("What would you like to know from your ebooks?", placeholder="e.g., What are the main themes of the book?", key="user_question")

#     if user_question:
#         user_input(user_question)

#     # Instructions and footer
#     st.markdown("---")
#     with st.expander("How to use this app"):
#         st.info("""
#             1.  **Upload**: Drag and drop one or more PDF files into the uploader in the sidebar.
#             2.  **Process**: Click the 'Process Documents' button and wait for the confirmation message.
#             3.  **Ask**: Type your question in the text box and press Enter. The AI will search the documents and generate an answer for you.
#         """)

# if __name__ == "__main__":
#     main()




# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import time

# Load environment variables from .env file
load_dotenv()

# --- Core Functions ---

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Splits a long text into smaller, overlapping chunks."""
    # A simple chunking strategy
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def create_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    try:
        model = load_embedding_model()
        
        with st.spinner("Creating embeddings for the documents... This may take a moment."):
            # 1. Create embeddings
            embeddings = model.encode(text_chunks, convert_to_tensor=False)
            
            # 2. Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))

        # Store index and chunks in session state
        st.session_state.vector_index = index
        st.session_state.text_chunks = text_chunks
        st.success("Documents processed successfully!")

    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_answer_from_groq(context, question):
    """Generates an answer using the Groq API based on context and a question."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Construct the prompt
        prompt = f"""
        Use the following pieces of context to answer the user's question. 
        If you don't know the answer from the provided context, just say that you don't know. Don't try to make up an answer.

        Context:
        {context}

        Question:
        {question}

        Helpful Answer:
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return "Sorry, I couldn't get an answer from the AI model."


def handle_user_question(user_question):
    """Processes a user's question against the vector store."""
    if "vector_index" not in st.session_state:
        st.warning("Please upload and process your documents first.")
        return

    model = load_embedding_model()
    
    with st.spinner("Searching for relevant information..."):
        # 1. Embed the user's question
        question_embedding = model.encode([user_question])

        # 2. Search the FAISS index
        k = 5 # Number of relevant chunks to retrieve
        distances, indices = st.session_state.vector_index.search(np.array(question_embedding).astype('float32'), k)
        
        # 3. Retrieve the relevant text chunks
        relevant_chunks = [st.session_state.text_chunks[i] for i in indices[0]]
        context = "\n\n---\n\n".join(relevant_chunks)

    with st.spinner("Generating answer with Groq..."):
        # 4. Get the answer from Groq
        answer = get_answer_from_groq(context, user_question)

        # 5. Display the answer
        st.subheader("Answer:")
        answer_placeholder = st.empty()
        displayed_answer = ""
        for char in answer:
            displayed_answer += char
            answer_placeholder.markdown(displayed_answer + "▌")
            time.sleep(0.01)
        answer_placeholder.markdown(displayed_answer)


# --- Streamlit UI ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="GenAI Ebook Q&A Assistant", page_icon="📚", layout="wide")

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Please add it to your .env file.")

    st.markdown("""
        <style>
            .st-emotion-cache-1y4p8pa { max-width: 100%; }
            .stButton>button { border-radius: 12px; padding: 10px 24px; border: none; transition: background-color 0.3s, transform 0.1s; font-weight: 600; }
            .stButton>button:hover { transform: scale(1.02); }
            div[data-testid="stSidebarUserContent"] .stButton>button { background-color: #008CBA; color: white; }
            .stTextInput>div>div>input { border-radius: 8px; }
            .stFileUploader>div>div>button { border-radius: 12px; border: 2px dashed #4CAF50; background-color: transparent; color: #4CAF50; }
        </style>
    """, unsafe_allow_html=True)

    st.header("GenAI Ebook Q&A Assistant 📚")
    st.write("Upload your ebooks, and ask any question about their content. This version runs without LangChain.")

    with st.sidebar:
        st.subheader("Your Ebooks")
        pdf_docs = st.file_uploader("Upload your PDF files here", accept_multiple_files=True, type="pdf")
        if st.button("Process Documents"):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    create_vector_store(text_chunks)
            else:
                st.warning("Please upload at least one PDF file.")

    st.subheader("Ask a Question")
    user_question = st.text_input("What would you like to know?", placeholder="e.g., What are the main themes of the book?")

    if user_question:
        handle_user_question(user_question)

    st.markdown("---")
    with st.expander("How this version works"):
        st.info("""
            This app demonstrates a manual RAG pipeline:
            1.  **PDF Parsing**: `PyPDF2` reads the text from your documents.
            2.  **Text Chunking**: A custom function splits the text into smaller pieces.
            3.  **Embeddings**: `sentence-transformers` runs a local model to turn text chunks into vectors.
            4.  **Vector Store**: `faiss` creates a searchable index of these vectors.
            5.  **Generation**: The `groq` client sends the relevant chunks and your question to the Llama3 model to get a final answer.
        """)

if __name__ == "__main__":
    main()
