
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import os
import requests
import json
import re
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
                    section_id: str, section_name: int) -> None:
    """
    Mutates doc.metadata with enriched fields.
    Called on every chunk AFTER splitting.
    """
    text = doc.page_content

    doc.metadata_update({
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
        "has_numbers": bool(re.search(r'\$[\d,]+|\d+\.?\d*%|\d{4}')),
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
        total = len(pages)
        print(f" {total} chunks created for {company_name}.")

        # 4. enrich every chunk
        for i, chunk in enumerate(splits):
            enrich_metadata(
                chunk,
                company = company_name,
                fiscal_year = fiscal_year,
                chunk_index = i,
                total_chunks = chunk.metadata["section"],
                section_name = chunk.metadata["section_name"],
            )
        return splits

# -------------------------------------------------------------------
# JSON METADATA SUMMARY (one file per company, used by UI / charts tab
def save_company_metadata_json(splits: list, company_name: str, output_dir: str) -> None:
    """
    Write a structured HSON summary of sections + stats for a company.
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
        "total_pages": max(c.metadata.get("pages", 0) for c in splits),
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
docs = []
pdf_files = [
    (os.path.join(BASE_DIR, "data", "Visa_10K_2025.pdf"), "Visa", 2025),
    (os.path.join(BASE_DIR, "data", "DigitalOcean_10K_2025.pdf"), "DigitalOcean", 2025),
    (os.path.join(BASE_DIR, "data", "Apple_10K_2025.pdf"), "Apple", 2025),
    (os.path.join(BASE_DIR, "data", "Amazon_10K_2025.pdf"), "Amazon", 2025),
]

# BUILD OR LOAD FAISS INDEX
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

if os.path.exists(INDEX_PATH):
    print("Loading existing index from disk...")
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    all_splits = {}
    for pdf_path, company_name, fiscal_year in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue

        print(f"Loading {company_name}...")
        splits = load_pdf_with_sections(pdf_path, company_name, fiscal_year)
        save_company_metadata_json(splits, company_name, fiscal_year)
        all_splits.extend(splits)

    print(f"Building FAISS index from {len(all_splits)} total chunks...")
    vector_store = FAISS.from_documents(all_splits, embeddings)
    vector_store.save_local(INDEX_PATH)
    print("Index saved.")
indexed_companies = set(company for _, company, _ in pdf_files)

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

