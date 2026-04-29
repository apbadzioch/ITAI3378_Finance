'''
imports: langchain/graph, faiss, embeddings, ...
constants/paths: INDEX_PATH, PDF_DIR, ...
models: HuggingFace, ChatOllama(must be Chat model for .bind tools)
load PDFs:
vector store:
helper functions:
@tools:
agent graph:
public interface:
index bootstrap:
'''

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

import os, json, re, subprocess, tempfile
from datetime import datetime

from charts import build_sankey
import plotly.graph_objects as go
import yfinance as yf

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage


# --- CONSTANTS/PATHS ---
# PDFs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FAISS INDEX
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
INDEXED_COMPANIES_PATH = os.path.join(BASE_DIR, "indexed_companies.json")
vector_store = None

# Analysis Prompt
prompt_template = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
    You are a financial analyst assistant specialized in analyzing 10-K SEC fillings.
    Use only the provided context to answer the question.
    Always cite specific figures, section, or page references wheh available.
    Be concise and precise, this is a financial analysis, not general conversation.
    Context: {context}
    Question: {question}
    Answer:
    """
)

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

# TICKER MAP — maps indexed company names to yfinance ticker symbols
TICKER_MAP = {
    "Visa": "V",
    "DigitalOcean": "DOCN",
    "Apple": "AAPL",
    "Amazon": "AMZN",
    "AMD": "AMD",
    "Crowdstrike": "CRWD",
    "IBM": "IBM",
    "Google": "GOOGL",
    "Intel": "INTC",
    "Oracle": "ORCL",
    "Nike": "NKE",
    "META": "META",
}

# CUSTOM SANKEY TEMPLATE PROMPT
sankey_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    {question}
    Filing context: {context}
    Respond ONLY with the JSON object. No explanation, no markdown fences. 
    """
)

# SANKEY SECTIONS
SANKEY_QUERY_TEMPLATE = """
You are analyzing the 10-K filing for {company}.

Your task: extract a Sankey diagram structure showing how revenue flows through this company.

Rules:
- The first node MUST be "Revenue" (the root)
- Use the actual line item names this company uses in their filing
- Every link value must be in millions USD
- Numbers must be internally consistent: Revenue = Cost of Revenue + Gross Profit, etc.
- Include 6-10 nodes total
- Only use figures explicitly stated in the filing
Respond ONLY with a JSON object in this exact format:
{{
    "nodes" : ["Revenue", "Cost of Revenue", "Gross Profit", ...],
    "links" : [
        {{"source": 0, "target": 1, "value": 1234}},
        {{"source": 0, "target": 2, "value": 5678}},
        ...
    ],
    "node_types": ["income", "cost", "income", "cost", ...]
}}
node_types must be "income" or "cost" for each node.    
"""


# --- MODELS ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 750,
    chunk_overlap = 200,
)
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)
llm = ChatOllama(
    model = "gemma4:31b-cloud",
    temperature = 0.2,
)


# --- LOAD PDFs ---
def detect_section(text: str):
    """
    Scan the first 300 characters of a page for a 10-K section heading.
    """
    sample = text[:300].lower()
    for pattern, item_id, name in SECTION_MAP:
        if re.search(pattern, sample):
            return item_id, name
    return None, None

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


# --- METADATA ---
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

# --- VECTOR STORE ---
def load_indexed_companies() -> set:
    if os.path.exists(INDEXED_COMPANIES_PATH):
        with open(INDEXED_COMPANIES_PATH, "r") as f:
            return set(json.load(f))
    return set()
indexed_companies= load_indexed_companies()

def save_indexed_companies(companies: set) -> None:
    with open(INDEXED_COMPANIES_PATH, "w") as f:
        json.dump(list(companies), f, indent=2)

def load_or_build_index() -> FAISS:
    if os.path.exists(INDEX_PATH):
        print("Loading index from disk...")
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings = embeddings,
            allow_dangerous_deserialization=True,
        )
        new_splits = []
        for pdf_path, company_name, fiscal_year in pdf_files:
            if not os.path.exists(pdf_path):
                continue
            if company_name not in indexed_companies:
                print(f"New company found, indexing {company_name}.")
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
    return vector_store

# --- HELPER FUNCTIONS ---
def detect_company(query: str) -> str | None:
    """
    Detect which company the user is asking about.
    """
    query_lower = query.lower()
    for company_name in indexed_companies:
        if company_name.lower() in query_lower:
            return company_name
    return None

# --- TOOLS ---
# Main ask() function
@tool
def ask(query: str) -> str:
    """
    Use this tool to answer specific questions about company 10-K filings.
    It performs a deep search across financial documents to find figures,
    risk factors, and management discussions.
    Input should be a clear question (e.g., 'What are the primary risk factors for Apple in 2025?')
    Main entry point called by gradio.
    Detects company from query, filters retrieval accordingly
    and returns a grounded financial analysis answer.
    """
    global vector_store
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

# sankey graph
@tool
def extract_sankey_structure(company:str) -> dict | None:
    """
    Extracts a structured JSON representation of a company's revenue flow
    to be used for creating a Sankey diagram.
    Use this when the user asks for a visual breakdown of revenue,
    income, or cost structures.
    """
    global vector_store
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": {
                "company": company,
                "has_numbers": True,
                "is_short_chunk": False,
            }
        }
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": sankey_template}
    )
    try:
        result = qa_chain.invoke({"query": SANKEY_QUERY_TEMPLATE.format(company=company)})
        raw = result["result"]

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            print(f"[extract_sankey_structure] No JSON found for {company}")
            return None

        data = json.loads(match.group())

        if not all(k in data for k in ("nodes", "links", "node_types")):
            print(f"[extract_sankey_structure] Missing keys for {company}")
            return None

        data["links"] = [
            {**lnk, "value": float(lnk.get("value") or 0)}
            for lnk in data["links"]
            if float(lnk.get("value") or 0) > 0
        ]

        return data

    except Exception as e:
        print(f"[extract_sankey_structure] Error for {company}: {e}")
        return None

@tool
def list_companies() -> list:
    """
    Returns a list of all companies currently indexed in the financial database.
    Use this tool first to verify if a company's data is available before
    calling 'ask' or 'extract_sankey_structure'.
    """
    return list(indexed_companies)

@tool
def build_chart(company_name: str):
    """Build the sankey diagram for the company."""
    data = extract_sankey_structure({"company": company_name})
    if not data or "nodes" not in data:
        return f"Not enough data to build a sankey diagram."
    payload = {
        "type": "sankey",
        "company": company_name,
        "data": data
    }
    return f"DATA_PAYLOAD:{json.dumps(payload)}"

@tool
def get_stock_info(company: str) -> str:
    """
    Retrieves live stock market data for a company using yfinance.
    Use this when the user asks about stock price, market cap, P/E ratio,
    52-week high/low, dividend yield, analyst targets, or any current
    market information for an indexed company.
    Input should be the company name as it appears in the index
    (e.g. 'Apple', 'Google', 'AMD').
    """
    ticker_symbol = TICKER_MAP.get(company)
    if not ticker_symbol:
        for name, sym in TICKER_MAP.items():
            if name.lower() == company.lower():
                ticker_symbol = sym
                break

    if not ticker_symbol:
        available = ", ".join(TICKER_MAP.keys())
        return (
            f"No ticker found for '{company}'. "
            f"Available companies: {available}"
        )

    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info

        def fmt_large(val):
            if val is None:
                return "N/A"
            if val >= 1_000_000_000:
                return f"${val / 1_000_000_000:.2f}B"
            if val >= 1_000_000:
                return f"${val / 1_000_000:.2f}M"
            return f"${val:,.2f}"

        def fmt_price(val):
            return f"${val:,.2f}" if val is not None else "N/A"

        def fmt_pct(val):
            return f"{val * 100:.2f}%" if val is not None else "N/A"

        name        = info.get("longName", company)
        exchange    = info.get("exchange", "N/A")
        currency    = info.get("currency", "USD")
        price       = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close  = info.get("previousClose")
        day_low     = info.get("dayLow")
        day_high    = info.get("dayHigh")
        week52_low  = info.get("fiftyTwoWeekLow")
        week52_high = info.get("fiftyTwoWeekHigh")
        mkt_cap     = info.get("marketCap")
        pe_trailing = info.get("trailingPE")
        pe_forward  = info.get("forwardPE")
        eps         = info.get("trailingEps")
        div_yield   = info.get("dividendYield")
        target_mean = info.get("targetMeanPrice")
        analyst_rec = info.get("recommendationKey", "N/A").upper()
        volume      = info.get("volume")
        avg_volume  = info.get("averageVolume")

        lines = [
            f"## {name} ({ticker_symbol}) — {exchange}",
            f"**Current Price:** {fmt_price(price)} {currency}",
            f"**Previous Close:** {fmt_price(prev_close)}",
            f"**Day Range:** {fmt_price(day_low)} – {fmt_price(day_high)}",
            f"**52-Week Range:** {fmt_price(week52_low)} – {fmt_price(week52_high)}",
            "",
            f"**Market Cap:** {fmt_large(mkt_cap)}",
            f"**Trailing P/E:** {f'{pe_trailing:.2f}' if pe_trailing else 'N/A'}",
            f"**Forward P/E:** {f'{pe_forward:.2f}' if pe_forward else 'N/A'}",
            f"**EPS (TTM):** {fmt_price(eps)}",
            f"**Dividend Yield:** {fmt_pct(div_yield)}",
            "",
            f"**Volume:** {f'{volume:,}' if volume else 'N/A'}",
            f"**Avg Volume:** {f'{avg_volume:,}' if avg_volume else 'N/A'}",
            "",
            f"**Analyst Consensus:** {analyst_rec}",
            f"**Mean Price Target:** {fmt_price(target_mean)}",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching stock data for {company} ({ticker_symbol}): {e}"

@tool
def get_stock_chart(company: str, period: str = "1y") ->  str:
    """
    Retrieves stock history. Returns a DATA_PAYLOAD string for a candlestick chart.
    """
    ticker_symbol = TICKER_MAP.get(company)
    if not ticker_symbol:
        return f"Ticker not found for {company}"

    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period=period)

        if hist.empty:
            return f"No historical data returned for {ticker_symbol}."

        payload = {
            "type": "stock_chart",
            "company": company,
            "ticker": ticker_symbol,
            "dates": hist.index.strftime("%Y-%m-%d").tolist(),
            "open": hist["Open"].tolist(),
            "high":hist["High"].tolist(),
            "low": hist["Low"].tolist(),
            "close": hist["Close"].tolist()
        }
        return f"DATA_PAYLOAD:{json.dumps(payload)}"

    except Exception as e:
        return f"Error fetching chart data for {company} ({ticker_symbol}): {e}"

# --- REPORT GENERATION ---
AUDIENCE_INSTRUCTIONS = {
    "analyst": (
        "You are writing for a professional equity analyst. "
        "Use precise financial terminology, cite specific figures with units, "
        "reference GAAP line items by name, and maintain a formal, objective tone. "
        "Assume the reader understands concepts like EBITDA, operating leverage, and segment reporting."
    ),
    "investor": (
        "You are writing for a retail investor with some financial literacy. "
        "Explain key figures clearly but don't over-simplify. "
        "Relate numbers to business outcomes (e.g. 'margins expanded because...'). "
        "Avoid heavy jargon but don't omit important detail."
    ),
    "general": (
        "You are writing for a general audience with no finance background. "
        "Use plain language. Explain any financial terms you use. "
        "Focus on the 'so what' — why does this number matter to the company's future? "
        "Use analogies where helpful. Keep sentences short and accessible."
    ),
}

REPORT_SECTION_QUERIES = {
    "business_overview": "Describe {company}'s core business model, primary products or services, revenue segments, and key markets they operate in.",
    "financial_highlights": "What were {company}'s key financial results? Include revenue, gross profit, operating income, net income, and year-over-year changes with specific figures.",
    "mda_summary": "Summarize {company}'s Management Discussion and Analysis. What did management highlight as key drivers of performance, operational changes, and strategic priorities?",
    "risk_factors": "What are the top 5 material risk factors disclosed by {company}? Focus on the most significant business, market, and operational risks.",
    "outlook": "What is {company}'s forward-looking guidance and strategic outlook? Include any targets, initiatives, or market opportunities management discussed.",
}

REPORT_BUILDER_SCRIPT = os.path.join(BASE_DIR, "build_report.js")

@tool
def generate_report(company: str, audience: str = "investor") -> str:
    """
    Generates a professional 5-section 10-K analysis report as a .docx file.
    Sections covered: Business Overview, Financial Highlights, MD&A Summary,
    Risk Factors, and Strategic Outlook.
    'audience' controls the tone and terminology level:
        - 'analyst'  : professional equity analyst, full financial jargon
        - 'investor' : retail investor, clear but not dumbed down (default)
        - 'general'  : plain English, no finance background assumed
    Returns the file path to the generated .docx for download.
    Use this when the user asks to generate, write, create, or export a report.
    """
    global vector_store
    if vector_store is None:
        return "No docs indexed."

    audience = audience.lower()
    if audience not in AUDIENCE_INSTRUCTIONS:
        audience = "investor"

    audience_prompt = AUDIENCE_INSTRUCTIONS[audience]

    # --- Query each section from the RAG system ---
    sections = {}
    for section_key, query_template in REPORT_SECTION_QUERIES.items():
        query = query_template.format(company=company)
        meta_filter = build_filter(query, company)

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": meta_filter if meta_filter else {"company": company},
            }
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["context", "question"],
                    template=(
                        f"{audience_prompt}\n\n"
                        "Use only the provided 10-K filing context to answer.\n"
                        "Be thorough but focused. Write 2-4 paragraphs.\n"
                        "Context: {{context}}\n"
                        "Question: {{question}}\n"
                        "Answer:"
                    )
                )
            },
        )
        result = qa_chain.invoke({"query": query})
        sections[section_key] = result["result"].strip()

        # --- Build the payload for the JS builder ---
        audience_label = \
        {"analyst": "Analyst Edition", "investor": "Investor Summary", "general": "Plain English Guide"}[audience]
        payload = {
            "company": company,
            "ticker": TICKER_MAP.get(company, ""),
            "audience": audience,
            "audience_label": audience_label,
            "generated_at": datetime.utcnow().strftime("%B %d, %Y"),
            "sections": sections,
        }

    # Write payload to a temp JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f, indent=2)
        payload_path = f.name

        # Output path
        out_dir = os.path.join(BASE_DIR, "reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{company}_10K_Report_{audience}.docx")

    # Call the Node.js builder
    try:
        result = subprocess.run(
            ["node", REPORT_BUILDER_SCRIPT, payload_path, out_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return f"Report builder error: {result.stderr}"
    except Exception as e:
        return f"Failed to run report builder: {e}"
    finally:
        os.unlink(payload_path)

    return f"REPORT_PATH:{out_path}"

# --------------------------------------------------------------------------
# --- AGENT GRAPH ---
# Define the state for the agent
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
# Initialize tools
toolbox = [ask, extract_sankey_structure, list_companies, build_chart, get_stock_info, get_stock_chart, generate_report]
tool_node = ToolNode(toolbox)
# Bind tools to the LLM
llm_with_tools = llm.bind_tools(toolbox)

def call_model(state: AgentState):
    """Decision maker: determines if a tool is needed or if the answer is ready."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

def should_continue(state: AgentState):
    """Router: checks if the last message from LLM contains tool calls."""
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END
# Constructing the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
# Compile the graph
agent_app = workflow.compile()

# --- PUBLIC INTERFACE ---
def chat_interface(message, history):
    """gradio wrapper for the langgraph agent."""
    # initialize state with user message
    inputs = {"messages": [HumanMessage(content=message)]}
    # run the graph
    result = agent_app.invoke(inputs)
    # return the final message content
    final_message = result["messages"][-1].content
    return final_message

# --- INDEX BOOTSTRAP ---
def bootstrap():
    """Ensures the FAISS index is initialized before the app starts."""
    global vector_store
    print("Bootstrapping Financial Index ...")
    vector_store = load_or_build_index()
    print("Bootstrap complete. System ready.")


















