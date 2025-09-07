
import streamlit as st

from embbed_data import search_schema, query_llm
# Streamlit UI

st.title("Database AI Agent")
query = st.text_input("Ask a question about the database schema:")
if st.button("Generate SQL"):
    if query:
        matched_schema = search_schema(query)
        llm_response = query_llm(f"You are a Senior SQL Engineer. Based on this database schema: {matched_schema}, , provide a professional response to the following query: {query}")
        st.write("### Response:skdlskldlsdsdsddsdksdl")
        st.write(llm_response)
    else:
        st.warning("Please enter a query!")

