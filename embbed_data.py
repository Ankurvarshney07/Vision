# import json
# import numpy as np
# import streamlit as st
# import torch
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer

# import faiss


# # Load Llama 3.2:1B Model using Hugging Face
# model_id = "gpt2"  # Use GPT-2 or any other publicly available model
# pipe = pipeline("text-generation", model=model_id)  


# # Load Embedding Model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")



# # Load Database Schema from JSON file
# with open("table_schema.json", "r") as f:
#     db_schema = json.load(f)

# # Store Each Table Schema in FAISS
# table_names = list(db_schema.keys())
# embeddings = []

# for table, fields in db_schema.items():
#     field_descriptions = ", ".join([f"{f['name']} ({f['type']})" for f in fields])
#     schema_text = f"Table '{table}' has fields: {field_descriptions}."
#     embedding = embed_model.encode(schema_text)
#     embeddings.append(embedding)

# # Convert embeddings to numpy array
# embeddings = np.array(embeddings, dtype=np.float32)

# # Initialize FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)



import json
import faiss
import numpy as np
import streamlit as st
import torch
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings

# Load Llama 3.2:1B Model using Hugging Face
# Load Llama 3.2:1B Model using Hugging Face
model_id = "gpt2"  # Use GPT-2 or any other publicly available model
pipe = pipeline("text-generation", model=model_id)  


# Load Embedding Model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS Index
embedding_dim = 384  # Adjust based on the embedding model
index = faiss.IndexFlatL2(embedding_dim)
schema_metadata = []

# Load Database Schema from JSON file
with open("table_schema.json", "r") as f:
    db_schema = json.load(f)

# Store Each Table Schema in FAISS
for table, fields in db_schema.items():
    field_descriptions = ", ".join([f"{f['name']} ({f['type']})" for f in fields])
    schema_text = f"Table '{table}' has fields: {field_descriptions}."
    embedding = embed_model.embed_query(schema_text)
    index.add(np.array([embedding], dtype=np.float32))
    schema_metadata.append({"table": table, "schema": schema_text})

def search_schema(query):
    """Search FAISS for the closest schema match."""
    query_embedding = np.array([embed_model.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, 1)
    return schema_metadata[indices[0][0]]["schema"] if indices[0][0] < len(schema_metadata) else "Schema not found."

def query_llm(question):
    """Query Llama model."""
    result = pipe(question, max_length=256)
    return result[0]["generated_text"].strip()



