
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import os, json, re
from datetime import datetime

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
    model="gemma3:27b-cloud",
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

# -------------------------------------------------------------------
# SECTION METADATA FILTERING
SECTION_MAP = [
    (r"item\s*1a", "Item 1A", "Risk Factors"),
    (r"item\s*1b", "Item 1B", "Unresolved Staff Comments"),
    (r"item\s*1[^a-z0-9]", "Item 1", "Business"),
    (r"item\s*2[^a-z0-9]", "Item 2", "Properties"),
    (r"item\s*3[^a-z0-9]", "Item 3", "Legal Proceedings"),
    (r"item\s*4[^a-z0-9]", "Item 4", "Mine Safety Disclosures"),
    (r"item\s*5[^a-z0-9]", "Item 5", "Market for Registrant Equity"),
    (r"item\s*6[^a-z0-9]", "Item 6", "Selected Financial Data"),
    (r"item\s*7a", "Item 7A", "Quantitative Market Risk"),
    (r"item\s*7[^a-z0-9]", "Item 7", "MD&A"),
    (r"item\s*8[^a-z0-9]", "Item 8", "Financial Statements"),
    (r"item\s*9a", "Item 9A", "Controls and Procedures"),
    (r"item\s*9b", "Item 9B", "Other Information"),
    (r"item\s*9[^a-z0-9]", "Item 9", "Disagreements with Accountants"),
    (r"item\s*10", "Item 10", "Directors and Corporate Governance"),
    (r"item\s*11", "Item 11", "Executive Compensation"),
    (r"item\s*12", "Item 12", "Security Ownership"),
    (r"item\s*13", "Item 13", "Certain Relationships"),
    (r"item\s*14", "Item 14", "Principal Accountant Fees"),
    (r"item\s*15", "Item 15", "Exhibits"),
]
def detect_section(text: str):
    """
    Scan the first 300 chars of a page for a 10-K section heading.
    Returns (item_id, section_name) or (None, None) if no new section is found
    """
    sample = text[:300].lower()
    for pattern, item_id, name in SECTION_MAP:
        if re.search(pattern, sample):
            return item_id, name
    return None, None

# -------------------------------------------------------------------
# METADATA ENRICHMENT
def enrich_metadata(doc, company: str, fiscal_year: int,
                    chunk_index: int, total_chunks: int,
                    section_id: str, section_name: str) -> None:
    """
    Mutates doc.metadata with enriched fields.
    Called on every chunk AFTER splitting.
    """
    text = doc.page_content

    doc.metadata.update({
        # identity
        "company": company,
        "fiscal_year": fiscal_year,
        "filing_type": "10-K",

        # location
        "section": section_id,
        "section_name": section_name,

        # chunk position
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "is_first_chunk": chunk_index == 0,
        "is_last_chunk": chunk_index == total_chunks - 1,

        # content signals
        "has_numbers": bool(re.search(r'\$[\d,]+|\d+\.?\d*%|\d{4}', text)),
        "has_table": text.count('\n') > 10 and '\t' in text,
        "is_short_chunk": len(text) < 200,

        # financial keyword flags

        # chunk quality
        "char_count": len(text),
        "word_count": len(text.split()),

        # provenance
        "source_file": doc.metadata.get("source", ""),
        "indexed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

# -------------------------------------------------------------------
# PDF LOADER WITH SECTION DETECTION
def load_pdf_with_sections(pdf_path: str, company_name: str, fiscal_year: int):
    """
    Load a PDF, detect 10-K sections at page level, split into chunks,
    then enrich every chunk's metadata.
    """
    # 1. Load pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 2. detect section per page and tag before splitting
    current_section_id = "Unknown"
    current_section_name = "Unknown"

    for doc in pages:
        detected_id, detected_name = detect_section(doc.page_content)
        if detected_id:
            current_section_id = detected_id
            current_section_name = detected_name
            # write onto page so splitter inherits it on every child chunk
        doc.metadata["company"] = company_name
        doc.metadata["fiscal_year"] = fiscal_year
        doc.metadata["section"] = current_section_id
        doc.metadata["section_name"] = current_section_name

    print(f" {len(pages)} pages loaded for {company_name}.")

    # 3. split - chunks inherit page metadata automatically
    splits = text_splitter.split_documents(pages)
    total = len(splits)
    print(f" {total} chunks created for {company_name}.")

    # 4. enrich every chunk
    for i, chunk in enumerate(splits):
        enrich_metadata(
            chunk,
            company = company_name,
            fiscal_year = fiscal_year,
            chunk_index = i,
            total_chunks = total,
            section_id = chunk.metadata["section"],
            section_name = chunk.metadata["section_name"],
        )
    return splits

# -------------------------------------------------------------------
# JSON METADATA SUMMARY (one file per company, used by UI / charts tab
def save_company_metadata_json(splits: list, company_name: str, output_dir: str) -> None:
    """
    Write a structured JSON summary of sections + stats for a company.
    """
    sections: dict = {}
    for chunk in splits:
        s = chunk.metadata.get("section", "Unknown")
        sname = chunk.metadata.get("section_name", "Unknown")
        page = chunk.metadata.get("page", 0)
        if s not in sections:
            sections[s] = {
                "section_id": s,
                "section_name": sname,
                "page_start": page,
                "page_end": page,
                "chunk_count": 0,
            }
        else:
            sections[s]["page_end"] = max(sections[s]["page_end"], page)
        sections[s]["chunk_count"] += 1

    output = {
        "company": company_name,
        "fiscal_year": splits[0].metadata.get("fiscal_year") if splits else None,
        "filing_type": "10_k",
        "total_pages": max(c.metadata.get("page", 0) for c in splits),
        "total_chunks": len(splits),
        "indexed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sections": list(sections.values()),
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{company_name}_metadata.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f" Metadata JSON saved: {path}")

# When using the webscraper, the SEC redirected to their security page.
# -------------------------------------------------------------------
# UPDATED : switch back to reading PDFs for sec.gov security reasons.
# LOAD PDFs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_files = [
    (os.path.join(BASE_DIR, "data", "Visa_10K_2025.pdf"), "Visa", 2025),
    (os.path.join(BASE_DIR, "data", "DigitalOcean_10K_2025.pdf"), "DigitalOcean", 2025),
    (os.path.join(BASE_DIR, "data", "Apple_10K_2025.pdf"), "Apple", 2025),
    (os.path.join(BASE_DIR, "data", "Amazon_10K_2025.pdf"), "Amazon", 2025),
    (os.path.join(BASE_DIR, "data", "AMD_10K_2025.pdf"), "AMD", 2025),
    (os.path.join(BASE_DIR, "data", "CROWDSTRIKE_10K_2025.pdf"), "Crowdstrike", 2025),
    (os.path.join(BASE_DIR, "data", "IBM_10K_2025.pdf"), "IBM", 2025),
    (os.path.join(BASE_DIR, "data", "GOOG_10K_2025.pdf"), "Google", 2025),
    (os.path.join(BASE_DIR, "data", "INTEL_10K_2025.pdf"), "Intel", 2025),
    (os.path.join(BASE_DIR, "data", "ORACLE_10K_2025.pdf"), "Oracle", 2025),
    (os.path.join(BASE_DIR, "data", "NIKE_10K_2025.pdf"), "Nike", 2025),
    (os.path.join(BASE_DIR, "data", "META_10K_2025.pdf"), "META", 2025),
]

# BUILD OR LOAD FAISS INDEX
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
INDEXED_COMPANIES_PATH = os.path.join(BASE_DIR, "indexed_companies.json")

def load_indexed_companies() -> set:
    if os.path.exists(INDEXED_COMPANIES_PATH):
        with open(INDEXED_COMPANIES_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_indexed_companies(companies: set) -> None:
    with open(INDEXED_COMPANIES_PATH, "w") as f:
        json.dump(list(companies), f, indent=2)

# -------------------------------------------------------------------
indexed_companies= load_indexed_companies()

if os.path.exists(INDEX_PATH):
    print("Loading existing index from disk...")
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    new_splits = []
    for pdf_path, company_name, fiscal_year in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue
        if company_name not in indexed_companies:
            print(f"New company found, indexing {company_name}...")
            splits = load_pdf_with_sections(pdf_path, company_name, fiscal_year)
            save_company_metadata_json(splits, company_name, METADATA_DIR)
            new_splits.extend(splits)
            indexed_companies.add(company_name)

    if new_splits:
        print(f"Adding {len(new_splits)} new chunks...")
        vector_store.add_documents(new_splits)
        vector_store.save_local(INDEX_PATH)
        save_indexed_companies(indexed_companies)
        print("Index updated.")
    else:
        print("No new companies to index.")

else:
    all_splits = []
    for pdf_path, company_name, fiscal_year in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue
        print(f"Loading {company_name}...")
        splits = load_pdf_with_sections(pdf_path, company_name, fiscal_year)
        save_company_metadata_json(splits, company_name, METADATA_DIR)
        all_splits.extend(splits)
        indexed_companies.add(company_name)

    print(f"Building FAISS index from {len(all_splits)} total chunks...")
    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local(INDEX_PATH)
    print("Index saved.")
indexed_companies = set(company for _, company, _ in pdf_files)

# -------------------------------------------------------------------
# SMART FILTER BUILDER
def build_filter(query: str, company: str | None) -> dict:
    """
    Build a FAISS metadata filter based on company + query intent.
    Falls back if no company is found.
    """
    query_lower = query.lower()
    base = {"company": company} if company else {}

    # section aware routing
    if any(w in query_lower for w in ["risk", "risks", "uncertainty", "adverse"]):
        return {**base, "section": "Item 1A", "is_short_chunk": False}
    if any(w in query_lower for w in ["competition", "competitor", "competitive", "market position"]):
        return {**base, "section": "Item 1", "is_short_chunk": False}
    if any(w in query_lower for w in ["revenue", "income", "profit", "margin", "earnings", "ebitda"]):
        return {**base, "has_numbers": True, "is_short_chunk": False}
    if any(w in query_lower for w in ["outlook", "guidance", "forecast", "expect", "future"]):
        return {**base, "section": "Item 7", "is_short_chunk": False}
    if any(w in query_lower for w in ["financial statement", "balance sheet", "cash flow"]):
        return {**base, "section": "Item 8", "is_short_chunk": False}
    if any(w in query_lower for w in ["management", "discussion", "analysis", "md&a"]):
        return {**base, "section": "Item 7", "is_short_chunk": False}

    # default: just filter by company and skip noise
    return {**base, "is_short_chunk": False} if base else {}

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
    meta_filter = build_filter(query, company)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": meta_filter if meta_filter else None,
        }
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
            section_name = doc.metadata.get("section_name", "")
            ref = f"{company_tag} - {section_name} (p.{page})" if section_name else f"{company_tag} (p.{page})"
            refs.add(ref)
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

