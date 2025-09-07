import json
import faiss
import numpy as np
import os
import dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load environment variables
dotenv.load_dotenv()
OPENAI_API_KEY =  os.getenv("GROQ_API_KEY")
SLACK_BOT_TOKEN = os.getenv("GROQ_API_KEY")
SLACK_SIGNING_SECRET = os.getenv("GROQ_API_KEY")
SLACK_APP_TOKEN = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Slack bot
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index and schema paths
FAISS_INDEX_PATH = "faiss_index.bin"
TABLE_NAMES_PATH = "table_names.json"
SCHEMA_PATH = "table_schema.json"

# Load schema from JSON
def load_schema():
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)

# Load or create FAISS index
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

# Find relevant tables
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

# Generate SQL using OpenAI
def generate_sql_with_openai(requirement, relevant_schema):
    table_definitions = "\n".join([
        f"Table: {table}\nColumns: {', '.join([col['name'] + ' (' + col['type'] + ')' for col in columns])}"
        for table, columns in relevant_schema.items()
    ])

    prompt = f"""
    You are an expert SQL query generator.

    Use the following table definitions to satisfy the database query:
    
    {table_definitions}

    Now, generate an optimized SQL query to retrieve data for the requirement: "{requirement}"
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

# Workflow
def workflow(requirement, provided_tables=None, provided_columns=None):
    schema = load_schema()
    index, table_names = load_faiss_index(schema)
    relevant_schema = find_relevant_tables(requirement, index, table_names, schema, provided_tables, provided_columns)
    
    sql_query = generate_sql_with_openai(requirement, relevant_schema)
    return sql_query

# Slack bot command
@app.message()
def handle_message(message, say):
    user_text = message["text"]
    
    # Extract table and column names if provided in message
    provided_tables = None
    provided_columns = None
    print("messages:" , message)

    if "|" in user_text:
        parts = user_text.split("|")
        user_text = parts[0].strip()
        if len(parts) > 1:
            provided_tables = [t.strip() for t in parts[1].split(",")] if parts[1].strip() else None
        if len(parts) > 2:
            provided_columns = [c.strip() for c in parts[2].split(",")] if parts[2].strip() else None

    sql_query = workflow(user_text, provided_tables, provided_columns)
    print(sql_query, "sdksjdksjd--")
    say(f"📝 **Generated SQL Query:**\n```sql\n{sql_query}\n```")

# Start Slack bot
if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()