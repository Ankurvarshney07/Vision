# from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pickle

# # Elasticsearch connection
# es = Elasticsearch("https://elastic.careers360.de")
# index = "colleges_3"

# # Load SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Fetch docs from ES
# def fetch_college_data():
#     # res = es.search(index=index, body={"query": {"match_all": {}}}, size=1000)
#     res = es.search(index=index, query={"match_all": {}}, size=1000)
#     print("connected----->>>>>>>>>>>>>>")
#     docs = []
#     ids = []
#     for hit in res['hits']['hits']:
#         doc = hit['_source']
#         text = doc["college_name"] + " - " + ", ".join([c["course_name"] for c in doc.get("flagship_course", [])])
#         docs.append(text)
#         print(text)
#         ids.append(str(doc["college_id"]))
#     return docs, ids

# # Create FAISS index
# def build_faiss_index(docs, ids):
#     embeddings = model.encode(docs)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings))
#     # Save index and metadata
#     faiss.write_index(index, "vector.index")
#     with open("metadata.pkl", "wb") as f:
#         pickle.dump({"texts": docs, "ids": ids}, f)

# if __name__ == "__main__":
#     docs, ids = fetch_college_data()
#     # build_faiss_index(docs, ids)






# import os
# import json
# import pickle
# import faiss
# import uuid
# import numpy as np
# from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveJsonSplitter
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS



# # === Setup Config ===
# ES_INDEX = "colleges_3"
# ES_HOST = "https://elastic.careers360.de"
# INDEX_PATH = "vector.index"
# METADATA_PATH = "metadata.pkl"
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# CHUNK_SIZE = 1000  # Max tokens or char size
# CHUNK_OVERLAP = 100

# # === Init Elasticsearch, Model, FAISS ===
# # es = Elasticsearch(ES_HOST)
# # embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
# # embedding_dim = embedding_model.get_sentence_embedding_dimension()
# # faiss_index = faiss.IndexFlatL2(embedding_dim)
# # metadata_store = []

# # # === JSON Chunker ===
# # splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# # # === Query Elasticsearch ===
# # res = es.search(index=ES_INDEX, query={"match_all": {}}, size=10000)

# # for hit in res["hits"]["hits"]:
# #     source = hit["_source"]

# #     # Filter: Must have at least one flagship course with domain == 4
# #     courses = source.get("flagship_course", [])
# #     filtered_courses = [c for c in courses if c.get("course_domain") == 4]
# #     if not filtered_courses:
# #         continue

# #     # Prepare JSON input for splitting (college + relevant courses only)
# #     college_entry = {
# #         "college_id": source.get("college_id"),
# #         "college_name": source.get("college_name"),
# #         "location": source.get("loc_string"),
# #         "courses": filtered_courses,
# #         "state": source.get("state", []),
# #         "city": source.get("city"),
# #         "url": source.get("overview_url"),
# #     }

# #     # Split into chunks using Langchain JSON splitter
# #     chunks = splitter.split_text(college_entry)
# #     embeddings = embedding_model.encode(chunks)

# #     for chunk, embedding in zip(chunks, embeddings):
# #         faiss_index.add(embedding.reshape(1, -1))
# #         metadata_store.append({
# #             "college_id": source.get("college_id"),
# #             "college_name": source.get("college_name"),
# #             "chunk": chunk
# #         })

# # print(f"✅ Embedded and indexed {len(metadata_store)} JSON chunks from filtered colleges.")

# # # === Save Index & Metadata ===
# # faiss.write_index(faiss_index, INDEX_PATH)
# # with open(METADATA_PATH, "wb") as f:
# #     pickle.dump(metadata_store, f)


# es = Elasticsearch(ES_HOST)
# model = SentenceTransformer(EMBED_MODEL_NAME)
# embedding_dim = model.get_sentence_embedding_dimension()
# faiss_index = faiss.IndexFlatL2(embedding_dim)
# metadata_store = []
# splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)

# # === Efficient Elasticsearch Query ===
# res = es.search(
#     index=ES_INDEX,
#     query={
#         "nested": {
#             "path": "flagship_course",
#             "query": {
#                 "term": {
#                     "flagship_course.course_domain": 4
#                 }
#             }
#         }
#     },
#     size=10000
# )

# print(f"📦 Fetched {len(res['hits']['hits'])} colleges from Elasticsearch")

# # === Process Each College ===
# for hit in res["hits"]["hits"]:
#     source = hit["_source"]

#     # Optional double-check
#     courses = source.get("flagship_course", [])
#     has_domain_4 = any(course.get("course_domain") == 4 for course in courses)
#     if not has_domain_4:
#         continue

#     full_college_json = source
#     # chunks = splitter.split_text(json.dumps(full_college_json))
#     chunks = splitter.split_text(full_college_json)
#     embeddings = model.encode(chunks)

#     for chunk, embedding in zip(chunks, embeddings):
#         faiss_index.add(np.array([embedding]))
#         metadata_store.append({
#             "college_id": full_college_json.get("college_id"),
#             "college_name": full_college_json.get("college_name"),
#             "chunk": chunk,
#             "full_data": full_college_json
#         })

# print(f"✅ Stored {len(metadata_store)} chunks from filtered colleges (with domain 4)")

# # === Save FAISS index and metadata ===
# faiss.write_index(faiss_index, INDEX_PATH)
# with open(METADATA_PATH, "wb") as f:
#     pickle.dump(metadata_store, f)

# print("💾 FAISS index and metadata saved.")










import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def load_college_blocks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [block.strip() for block in content.split("--------------------------------------------------") if block.strip()]

def embed_texts(text_blocks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_blocks, show_progress_bar=True)
    return embeddings, model

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_and_metadata(index, metadata, out_path_prefix="college_vector_store"):
    faiss.write_index(index, f"{out_path_prefix}.index")
    with open(f"{out_path_prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def main():
    txt_file = "college_data.txt"  # Replace with your cleaned data file
    blocks = load_college_blocks(txt_file)
    
    embeddings, model = embed_texts(blocks)
    index = build_faiss_index(np.array(embeddings))

    metadata = [{"text": block} for block in blocks]

    save_faiss_and_metadata(index, metadata)
    print(f"✅ Stored {len(blocks)} colleges into FAISS vector DB.")

if __name__ == "__main__":
    main()
