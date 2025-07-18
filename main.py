import os
import streamlit as st
import pickle
import time
import langchain
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
st.markdown("""
    <style>
    /* Background & main layout */
    .stApp {
        background-color: #FFF0F5;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #FADADD;
    }

    /* Sidebar title: NEWS ARTICLE URLS */
    section[data-testid="stSidebar"] h1 {
        color: #FC8EAC ;
        font-size: 1.5rem;
    }

    /* Button: Process URLS */
    button[kind="secondary"] {
        color: white !important;
        background-color: #FC8EAC !important; /* Flamingo Pink */
        border: none;
        border-radius: 8px;
    }

    button[kind="secondary"]:hover {
        background-color: #F88379 !important; /* Coral Pink on hover */
    }

    /* Text input label: QUESTIONS */
    label {
        color: #6C5B7B !important; /* Muted Violet */
        font-weight: bold;
    }

    /* Center block content */
    .block-container {
        text-align: center;
        padding-top: 2rem;
    }
    
    input[type="text"] {
    font-size: 1.1rem !important;
    color: #6C5B7B !important;
    background-color: #FFF0F5 !important;
    border: 1px solid #FC8EAC !important;
    border-radius: 10px !important;
    padding: 10px !important;
    width: 100% !important;
}
input::placeholder {
    color: #8A2BE2;
    font-weight: 500;
}

    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div style="
        background-color: #FFE4EC;
        border-radius: 15px;
        padding: 15px 20px;
        margin: 20px auto;
        max-width: 320;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    ">
        <h1 style='color: #FC8EAC; margin-bottom: 10px;'>NEWS RESEARCH TOOL 📰</h1>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.title("NEWS ARTICLE URLS")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
valid_urls = [u.strip() for u in urls if u.strip()]

process_url_clicked = st.sidebar.button("Process URLS")
if process_url_clicked and not valid_urls:
    st.error("Enter at least one URL.")
    st.stop()

file_path ="faiss_store_ollama"

main_placeholder =st.empty()


if process_url_clicked:
   loader = UnstructuredURLLoader(urls=valid_urls)
   main_placeholder.text("Data Loading....Started...")
   data = loader.load()
   if not data:
       st.error("No readable text found at those URLs.")
       st.stop()

   #spliting dataset
   text_splitter=RecursiveCharacterTextSplitter(
       separators=['\n\n','\n','.',','],
       chunk_size=1000
   )
   main_placeholder.text("Text Splitter...Started...")
   docs=text_splitter.split_documents(data)
   if not docs:
        st.error("Text splitter produced zero chunks.")
        st.stop()
   # create embeddings
   embeddings = OllamaEmbeddings(model="llama3")
   vectorindex_ollama = FAISS.from_documents(docs, embeddings)

   vectorindex_ollama = FAISS.from_documents(docs, embeddings)
   vectorindex_ollama.save_local(file_path)

st.markdown("""
<h7 style='color: #6C5B7B; text-align: center; font-size: 1.5rem;'>
    Ask a query related to the articles
</h7>
""", unsafe_allow_html=True)

query = st.text_input(" ", placeholder="Type your query here...")
if query:
    if not os.path.exists(file_path):
        st.error("Vector store not found — click 'Process URLS' first.")
        st.stop()

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.load_local(file_path, embeddings,allow_dangerous_deserialization=True)
    llm = Ollama(model="llama3")
    chain= RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result= chain({"question": query},return_only_outputs=True)
    st.header("Answer")
    st.markdown(f"""
        <div style="
            background-color: #fdf6f9;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #f5c2cc;
            margin-top: 20px;
            font-size: 1.1rem;
            color: #4B4453;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        ">
            <strong>Answer:</strong><br>{result["answer"]}
        </div>
    """, unsafe_allow_html=True)
