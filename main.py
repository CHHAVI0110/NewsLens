import os
import time
import streamlit as st

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
# ---------------- SESSION STATE INIT ---------------- #
if "processed" not in st.session_state:
    st.session_state.processed = False


# ---------------- UI STYLES ---------------- #
st.markdown("""
<style>
.stApp { background-color: #FFF0F5; }

[data-testid="stSidebar"] { background-color: #FADADD; }

section[data-testid="stSidebar"] h1 {
    color: #FC8EAC;
    font-size: 1.5rem;
}

button[kind="secondary"] {
    color: white !important;
    background-color: #FC8EAC !important;
    border-radius: 8px;
}

button[kind="secondary"]:hover {
    background-color: #F88379 !important;
}

label {
    color: #6C5B7B !important;
    font-weight: bold;
}

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
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
with st.container():
    st.markdown("""
    <div style="background-color:#FFE4EC;border-radius:15px;
    padding:15px 20px;margin:20px auto;max-width:320px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);">
        <h1 style='color:#FC8EAC;'>NEWSLENS 📰</h1>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("NEWS ARTICLE URLS")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
valid_urls = [u.strip() for u in urls if u.strip()]

process_url_clicked = st.sidebar.button("Process URLS")
file_path = "faiss_store_ollama"
main_placeholder = st.empty()

# ---------------- PROCESS URLS ---------------- #
if process_url_clicked:
    if not valid_urls:
        st.error("Enter at least one URL.")
        st.stop()

    # HARD LIMIT (IMPORTANT)
    valid_urls = valid_urls[:2]

    main_placeholder.markdown("""
    <div style="background-color:#E6F2FF;color:#0B2C4A;
    padding:14px;border-radius:12px;font-weight:700;">
    📥 Data loading started...
    </div>
    """, unsafe_allow_html=True)

    start = time.time()
    loader = WebBaseLoader(valid_urls)
    data = loader.load()
    st.write("URL loading time:", round(time.time() - start, 2), "seconds")

    if not data:
        st.error("No readable content found.")
        st.stop()

    main_placeholder.markdown("""
    <div style="background-color:#E6F2FF;color:#0B2C4A;
    padding:14px;border-radius:12px;font-weight:700;">
    ✂️ Text splitting started...
    </div>
    """, unsafe_allow_html=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("No text chunks created.")
        st.stop()

    docs = docs[:40]

    main_placeholder.markdown("""
    <div style="background-color:#E6F2FF;color:#0B2C4A;
    padding:14px;border-radius:12px;font-weight:700;">
    🧠 Creating embeddings...
    </div>
    """, unsafe_allow_html=True)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorindex_ollama = FAISS.from_documents(docs, embeddings)
    vectorindex_ollama.save_local(file_path)

    main_placeholder.markdown("""
    <div style="background-color:#DFF6DD;color:#1E4620;
    padding:14px;border-radius:12px;font-weight:700;">
    ✅ Processing complete!
    </div>
    """, unsafe_allow_html=True)
    st.session_state.processed = True

# ---------------- QUERY SECTION ---------------- #
# ---------------- QUERY SECTION ---------------- #
st.markdown("""
<h3 style='color:#6C5B7B;'>Ask a query related to the articles</h3>
""", unsafe_allow_html=True)

query = st.text_input(
    " ",
    placeholder="Process URLs first, then ask your question...",
    disabled=not st.session_state.processed
)

if query and st.session_state.processed:

    with st.spinner("🔎 Searching for answer..."):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vectorstore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = Ollama(model="llama3")

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),

        )

        result = chain(
            {"question": query},
            return_only_outputs=True
        )

    st.markdown(f"""
    <div style="
        background-color:#fdf6f9;
        padding:20px;
        border-radius:12px;
        border:1px solid #f5c2cc;
        font-size:1.1rem;
        color:#4B4453;
        margin-top:20px;
        box-shadow:0 2px 10px rgba(0,0,0,0.05);
    ">
        <strong>Answer:</strong><br>
        {result["answer"]}
    </div>
    """, unsafe_allow_html=True)
