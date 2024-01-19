from langchain_helper import qa_chain
# from langchain_helper import create_vector_db
import streamlit as st


st.title("CSV Q&A")
btn = st.button("create knowledge base")
if btn:
    pass

question = st.text_input("Question: ")
if question:
    chain = qa_chain()
    response = chain(question)

    st.header("Response:")
    st.write(response['result'])
