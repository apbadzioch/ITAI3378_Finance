from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM

import os
import bs4

# ----------------------------------------------------
# 1. Loading PDFs using PyPDFLoader
docs = []
pdf_files = [
    "data/Visa_10k_2025.pdf"
]
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

# Loading web pages using WebBaseLoader
# pages = []
# web_pages = WebBaseLoader(
#    ["https://www.sec.gov/search-filings", "https://www.edgar-online.com/"]
#)

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Build the vectorstore (FAISS) and save to directory
INDEX_PATH = "faiss_index"

if os.path.exists(INDEX_PATH):
    print("Loading from file...")
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings
    )
else:
    print("Building index...")
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(INDEX_PATH)

# 5. Use Ollama LLM for Langchain
llm = OllamaLLM(
    model="gemma3:4b",
    temperature=0.2
)

# Build RetrievalQA chain (Question Answering)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_kwargs={"k":3}
    )
)

while True:
    queries = input("How may I help you?").strip()
    if queries == {"quit", "exit"}:
        break

    result = qa_chain.invoke({"query": queries})
    print("\nAnswer:\n", result["result"], "\n")



