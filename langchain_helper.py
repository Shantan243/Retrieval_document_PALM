import langchain
from langchain.llms import GooglePalm
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st


# vectordb_file_path = "C:/Users/shant_w5mrdz3/OneDrive/Desktop/Langchain_examples/palm/CSV_Palm_Q_A/FAISS_index/index.faiss"
# def create_vector_db():
#   loader = CSVLoader("C:/Users/shant_w5mrdz3/OneDrive/Desktop/Langchain_examples/palm/News_Finance_query_langchain/dataset_sample/codebasics_faqs.csv",source_column="prompt")
#    data = loader.load()
#    vectordb = FAISS.from_documents(documents=data,embedding=embeddings)
#   vectordb.save_local(vectordb_file_path)

@st.cache_resource
def qa_chain():
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.7)
    embeddings = HuggingFaceInstructEmbeddings()
    loader = CSVLoader("C:/Users/shant_w5mrdz3/OneDrive/Desktop/Langchain_examples/palm/News_Finance_query_langchain/dataset_sample/codebasics_faqs.csv",source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    # vectordb = FAISS.load_local(vectordb_file_path, embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """Given the following context and a question, generate answer from context only. 
                    In the answer try to provide as much text as possible from "response" from the source document. 
                    If the answer is not found in the context, kindly say "I dont know" . Dont try to make up answer.
    CONTEXT:{context}
    QUESTION:{question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, input_key="query",
                                        return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    return chain
if __name__ == "__main__":
    chain = qa_chain()
    print(chain("do you have a policy refund?"))





