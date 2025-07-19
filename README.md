# NewsLens
# 📰 NewsLens – News Research Tool using LangChain & LLaMA3

NewsLens is an AI-powered research assistant that lets you input **news article URLs**, then **summarizes and answers questions** using **LangChain**, **Ollama (LLaMA3)**, **FAISS**, and **Streamlit**.

## Features
-  Ask questions about any news article using LLMs
-  Extracts article content via LangChain & FAISS
-  Answers user queries using semantic search and LLaMA3
-  Easy-to-use Streamlit interface with custom styling
-  
## Tech Stack
- **Python**
- **LangChain** (`langchain`, `langchain-community`, `langchain-ollama`)
- **Ollama** with **LLaMA3**
- **FAISS** (vector database for semantic search)
- **Unstructured** (for parsing website content)
- **Streamlit** (UI)
- **Tiktoken** (token counting)
  
## How It Works

1. Paste **up to 3 news article URLs** into the sidebar
2. Click **“Process URLs”**
3. Ask a question like:
   - “What is this article about?”
   - “List 3 key facts”
4. The app retrieves, splits, embeds, and runs your query on the vector store
5. You receive a clean, AI-generated answer with source reference

## Example URLs to Try
- https://economictimes.indiatimes.com/nri/work/new-zealand-to-expand-work-hours-for-international-students-along-with-these-key-changes-from-november-2025/articleshow/122777973.cms
- https://www.newscientist.com/article/2486023-deep-sleep-seems-to-lead-to-more-eureka-moments/
- https://economictimes.indiatimes.com/nri/work/new-zealand-to-expand-work-hours-for-international-students-along-with-these-key-changes-from-november-2025/articleshow/122777973.cms
  
## Installation
1. Clone the repository:
```bash
git clone 'https://github.com/CHHAVI0110/NewsLens.git'
cd NewsLens

