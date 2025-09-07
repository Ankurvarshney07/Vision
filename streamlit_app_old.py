# import mysql.connector
import openai
import json
import faiss
import numpy as np
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import requests
from groq import Groq

OPENAI_API_KEY = ''

# Set OpenAI API Key
client = OpenAI(api_key=OPENAI_API_KEY)

import dotenv

dotenv.load_dotenv()

DB_HOST = os.environ.get('DB_HOST')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DATABASE = os.environ.get('DB_NAME')

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY
# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define FAISS index file path
FAISS_INDEX_PATH = "faiss_index.bin"
TABLE_NAMES_PATH = "table_names.json"



# Load schema from JSON file
def load_schema():
    with open("table_schema.json", "r") as f:
        return json.load(f)

# Create or load FAISS index
def load_faiss_index(schema):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TABLE_NAMES_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(TABLE_NAMES_PATH, "r") as f:
            table_names = json.load(f)
    else:
        table_names = list(schema.keys())
        table_descriptions = [" ".join([col["name"] for col in columns]) for columns in schema.values()]
        table_embeddings = np.array([embedding_model.encode(desc) for desc in table_descriptions])
        index = faiss.IndexFlatL2(table_embeddings.shape[1])
        index.add(table_embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(TABLE_NAMES_PATH, "w") as f:
            json.dump(table_names, f)
    return index, table_names

# Find relevant tables based on user input
def find_relevant_tables(requirement, index, table_names, schema, provided_tables=None, provided_columns=None, top_k=5):
    if provided_tables:
        relevant_tables = {table: schema[table] for table in provided_tables if table in schema}
    else:
        requirement_embedding = np.array([embedding_model.encode(requirement)])
        distances, indices = index.search(requirement_embedding, top_k)
        relevant_tables = {table_names[i]: schema[table_names[i]] for i in indices[0] if i < len(table_names)}

    if provided_columns:
        for table in relevant_tables:
            relevant_tables[table] = [col for col in relevant_tables[table] if col["name"] in provided_columns]

    return relevant_tables


# Generate SQL dynamically using OpenAI with enhanced prompt
def generate_sql_with_openai(requirement, relevant_schema):
    # Extract table definitions in a structured format
    table_definitions = "\n".join([
        f"Table: {table}\nColumns: {', '.join([col['name'] + ' (' + col['type'] + ')' for col in columns])}"
        for table, columns in relevant_schema.items()
    ])

    # Construct a better prompt using table definitions
    prompt = f"""
    You are an expert SQL query generator.

    Use the following table definitions to satisfy the database query:
    
    {table_definitions}

    Now, generate an optimized SQL query to retrieve data for the requirement: "{requirement}"
    """

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

def generate_sql_with_groq(requirement, relevant_schema):
    # Extract table definitions in a structured format
    table_definitions = "\n".join([
        f"Table: {table}\nColumns: {', '.join([col['name'] + ' (' + col['type'] + ')' for col in columns])}"
        for table, columns in relevant_schema.items()
    ])

    # Construct a better prompt using table definitions
    prompt = f"""
    You are an expert SQL query generator.

    Use the following table definitions to satisfy the database query:
    
    {table_definitions}

    Now, generate an optimized SQL query to retrieve data for the requirement: "{requirement}"
    """

    # Call OpenAI API
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()


import ollama

def generate_sql_with_tinyllama(requirement, relevant_schema):
    # Format table definitions
    table_definitions = "\n".join([
        f"Table: {table}\nColumns: {', '.join([col['name'] + ' (' + col['type'] + ')' for col in columns])}"
        for table, columns in relevant_schema.items()
    ])

    # Construct a concise prompt
    prompt = f"""
    Generate an optimized SQL query using the following table definitions:

    {table_definitions}

    Requirement: "{requirement}"
    """

    # Call TinyLlama using Ollama
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"].strip()


def generate_sql_with_deepseek(requirement, relevant_schema):
    # Format table definitions
    table_definitions = "\n".join([
        f"Table: {table}\nColumns: {', '.join([col['name'] + ' (' + col['type'] + ')' for col in columns])}"
        for table, columns in relevant_schema.items()
    ])

    # Construct a concise prompt
    prompt = f"""
    You are an expert SQL query generator. Given the following database schema, generate an optimized SQL query:

    {table_definitions}

    Requirement: "{requirement}"
    """

    # Call DeepSeek-R1 1.5B via Ollama
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"].strip()


# Database Connection
def connect_db():
    return mysql.connector.connect(
        host="your_host",
        user="your_user",
        password="your_password",
        database="your_database"
    )

# Execute SQL query
def execute_sql(query):
    if not query:
        return "No relevant table found."
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

# Workflow function using Multi-Agent Collaboration
def workflow(requirement, provided_tables=None, provided_columns=None):
    print("UserProxy requests data analytics task.")

    schema = load_schema()
    index, table_names = load_faiss_index(schema)
    relevant_schema = find_relevant_tables(requirement, index, table_names, schema, provided_tables, provided_columns)

    # sql_query = generate_sql_with_tinyllama(requirement, relevant_schema)
    # sql_query = generate_sql_with_deepseek(requirement, relevant_schema)
    # sql_query = generate_sql_with_grok(requirement, relevant_schema)
    # sql_query = generate_sql_with_groq(requirement, relevant_schema)
    sql_query = generate_sql_with_openai(requirement, relevant_schema)
    print("DataEngineer generated SQL:", sql_query)
    return sql_query
    result = execute_sql(sql_query)
    print("SrDataAnalyst executed SQL and obtained result:", result)

    print("ProductManager validates the result.")
    return result


# Streamlit UI for User Interaction
def streamlit_app():
    st.title("Helloooo ....     📞")

    user_query = st.text_area("Enter your requirement (e.g., 'Get total users count 🤪')")
    provided_tables = st.text_input("Optional: Provide table names (comma-separated) 😏")
    provided_columns = st.text_input("Optional: Provide column names (comma-separated) 😎")

    provided_tables = [t.strip() for t in provided_tables.split(",")] if provided_tables else None
    provided_columns = [c.strip() for c in provided_columns.split(",")] if provided_columns else None

    if st.button("Generate SQL & Execute 💩"):
        if user_query:
            final_result = workflow(user_query, provided_tables, provided_columns)
            st.subheader("Generated SQL Query & Execution Result:")
            st.write(final_result)
        else:
            st.warning("Please enter a valid requirement.")

# Run Streamlit App
if __name__ == "__main__":
    streamlit_app()