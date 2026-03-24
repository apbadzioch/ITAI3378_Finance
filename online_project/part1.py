
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import os
# import requests
# import json
# import bs4

# ----------------------------------------------------------------------------------
# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=150
)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# USE OLLAMA LLM FOR LANGCHAIN
llm = OllamaLLM(
    model="gemma3:4b",
    temperature=0.2
)

# ----------------------------------------------------------------------------------
# CUSTOM FINANCIAL ANALYST PROMPT
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a financial analyst assistant specialized in analyzing 10-K SEC filings.
    Use only the provided context to answer the question.
    If the answer cannot be found in the context, clearly state that.
    Always cite specific figures, section, or page references when available.
    Be concise and precise; this is financial analysis, not general conversation.
    
    Context: {context}
    Question: {question}
    Answer:
    
    """
)

'''
When using the webscraper, the SEC redirected to their security page. This code was kept
for future reference.
# ----------------------------------------------------------------------------------
# SEC EDGAR FUNCTIONS
# CIK (Central Index Key): a unique ID assigned to every company that file with the SEC.
SEC_HEADERS = {"User-Agent": "MyApp"}

def search_cik(company_name: str) -> tuple[str, str] | None:
    """
    Search SEC EDGAR for a company by name and returns (official_name, CIK).
    Returns None if not found.
    """
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{company_name}%22&forms=10-K"
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=10)
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            source = hits[0]["_source"]
            return source.get("entity_name", company_name), source.get("file_num", "")
    except Exception as e:
        print(f"CIK search error: {e}")
        return None

def get_10k_url(cik: str) -> str | None:
    """
    Given a CIK number, find the most recent 10-K filing URL from SEC EDGAR.
    """
    # zero-pad CIK to 10 digits as required by sec API
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=10)
        data = response.json()
        filings = data["filings"]["recent"]

        for i, form in enumerate(filings["form"]):
            if form == "10-K":
                accession = filings["accessionNumber"][i].replace("-", "")
                doc_name = filings["primaryDocument"][i]
                cik_raw = str(cik).strip("0")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_raw}/{accession}/{doc_name}"
                )
                return filing_url

    except Exception as e:
        print(f"Error fetching 10-K URL for CIK {cik}: {e}")

    return None

def load_company_from_sec(company_name: str, cik: str) -> list:
    """
    Fetch a company's most recent 10-K from SEC and return tagged document chunks.
    """
    print(f"Fetching 10-K for {company_name} (CIK: {cik})...")
    filing_url = get_10k_url(cik)

    if not filing_url:
        print(f"Could not find 10-K URL for {company_name}")
        return []
    print(f"Loading from: {filing_url}")

    try:
        loader = WebBaseLoader(
            web_paths=[filing_url],
            bs_kwargs={
                "parse_only": bs4.SoupStrainer(["p", "td", "th", "h1", "h2", "h3"])
            },
            requests_kwargs={"headers": SEC_HEADERS}
        )
        pages = loader.load()

        # tag every page with company metadata
        for doc in pages:
            doc.metadata["company"] = company_name
            doc.metadata["cik"] = cik
            doc.metadata["source_url"] = filing_url

        return pages

    except Exception as e:
        print(f"Error loading {company_name} filings: {e}")
        return []

# -------------------------------------------------------------
# INDEXING MANAGEMENT
INDEX_PATH = "online_project/faiss_index"
INDEXED_COMPANIES_PATH = "online_project/indexed_companies.json"

def load_indexed_companies() -> set:
    """Load the set of already-indexed company names from disk."""
    if os.path.exists(INDEXED_COMPANIES_PATH):
        with open(INDEXED_COMPANIES_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_indexed_companies(companies: set):
    """Persist the set of indexed company names to disk."""
    os.makedirs(os.path.dirname(INDEXED_COMPANIES_PATH), exist_ok=True)
    with open(INDEXED_COMPANIES_PATH, "w") as f:
        json.dump(list(companies), f)

def ensure_company_indexed(company_name: str, cik: str):
    """
    Check if company is already in the vector store.
    If not, fetch from SEC, split, embed, and add to the index.
    """
    global vector_store, indexed_companies

    if company_name in indexed_companies:
        print(f"{company_name} already indexed. Skipping.")
        return

    docs = load_company_from_sec(company_name, cik)
    if not docs:
        print(f"No documents loaded for {company_name}.")
        return

    splits = text_splitter.split_documents(docs)
    print(f"Adding {len(splits)} chunks for {company_name} to index...")

    if vector_store is None:
        vector_store = FAISS.from_documents(splits, embeddings)
    else:
        vector_store.add_documents(splits)

    indexed_companies.add(company_name)
    vector_store.save_local(INDEX_PATH)
    save_indexed_companies(indexed_companies)
    print(f"{company_name} indexed successfully.")

# -------------------------------------------------------------
# VECTOR STORE INITIALIZATION
indexed_companies = load_indexed_companies()

if os.path.exists(INDEX_PATH):
    print("Loading from file...")
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("No existing index found. Starting fresh.")
    vector_store = None

# -------------------------------------------------------------------
# PRE-LOAD DEFAULT COMPANIES - add as needed
# CIK numbers can be found at https://www.sec.gov/cgi-bin/browse-edgar
DEFAULT_COMPANIES = [
    ("Visa", "0001403161"),
    ("DigitalOcean", "0001560327"),
]
for name, cik in DEFAULT_COMPANIES:
    ensure_company_indexed(name, cik)
'''

# -------------------------------------------------------------------
# UPDATED : switch back to reading PDFs for SEC security reasons.
# LOAD PDFs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
docs = []
pdf_files = [
    (os.path.join(BASE_DIR, "data", "Visa_10K_2025.pdf"), "Visa"),
    (os.path.join(BASE_DIR, "data", "DigitalOcean_10K_2025.pdf"), "DigitalOcean"),
    (os.path.join(BASE_DIR, "data", "Apple_10K_2025.pdf"), "Apple"),
    (os.path.join(BASE_DIR, "data", "Amazon_10K_2025.pdf"), "Amazon"),
]

# BUILD OR LOAD FAISS INDEX
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

if os.path.exists(INDEX_PATH):
    print("Loading existing index from disk...")
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    for pdf_path, company_name in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue
        print(f"Loading {company_name} from {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for doc in pages:
            doc.metadata["company"] = company_name
        docs.extend(pages)
        print(f"Loaded {len(pages)} pages for {company_name}.")

    print("Building index from PDFs...")
    splits = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(splits)}")
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(INDEX_PATH)
    print("Index saved.")
indexed_companies = set(company for _, company in pdf_files)

# -------------------------------------------------------------------
# COMPANY DETECTION FROM QUERY
def detect_company(query: str) -> str | None:
    """
    Detect which company the user is asking about.
    """
    query_lower = query.lower()
    for company_name in indexed_companies:
        if company_name.lower() in query_lower:
            return company_name
    return None

# -------------------------------------------------------------------
# MAIN ASK FUNCTION
def ask(query: str) -> str:
    """
    Main entry point called by Gradio.
    Detects company from query, filters retrieval accordingly
    and returns a grounded financial analysis answer.
    """
    if vector_store is None:
        return "No documents have been indexed yet. Please add a company first."
    company = detect_company(query)

    # Build retriever with or without company filter
    if company:
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                 "filter": {"company": company}
            }
        )
    else:
        # no specific company is detected - search across all
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 6}
        )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    result = qa_chain.invoke({"query": query})
    answer = result["result"]

    # append source references to the answer
    sources = result.get("source_documents", [])
    if sources:
        refs = set()
        for doc in sources:
            company_tag = doc.metadata.get("company", "Unknown")
            page = doc.metadata.get("page", "?")
            refs.add(f"{company_tag} - page {page}")
        answer += "\n\n Sources: " + " | ".join(sorted(refs))
    return answer


# -------------------------------------------------------------------
# DYNAMIC COMPANY ADDITION (for future use)
#def add_company(company_name: str, cik: str) -> str:
    """
    Dynamically add a new company name to the index at runtime.
    Can be called from Gradio UI or an agent tool.
    """
    #ensure_company_indexed(company_name, cik)
    #return f"{company_name} has been added to the index."

