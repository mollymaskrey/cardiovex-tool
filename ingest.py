"""
ingest.py

Chunks the Cardiovex source documents and loads them into ChromaDB.

Usage:
    python ingest.py                        # uses data/chroma_db locally
    python ingest.py --db /var/data/chroma_db  # custom db path
    python ingest.py --reset                # rebuild from scratch
"""

import argparse
import os
import re

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS = {
    "shield1": {
        "path": "docs/SHIELD-1_NEJM.txt",
        "source": "SHIELD-1 Trial (NEJM)",
        "doc_type": "clinical_trial",
    },
    "shield2": {
        "path": "docs/SHIELD-2_NEJM.txt",
        "source": "SHIELD-2 Trial (NEJM)",
        "doc_type": "clinical_trial",
    },
    "fda_label": {
        "path": "docs/Cardiovex_FDA_Label.txt",
        "source": "Cardiovex FDA Label",
        "doc_type": "fda_label",
    },
}

COLLECTION_NAME = "cardiovex_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Text cleaning and section splitting
# ---------------------------------------------------------------------------

def clean_text(text):
    """Basic text cleanup."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def split_into_sections(text, doc_key):
    """
    Split document into logical sections.
    For clinical trials and FDA labels, look for common section headers.
    """
    # Common section patterns (case-insensitive)
    section_patterns = [
        r'^ABSTRACT\s*$',
        r'^INTRODUCTION\s*$',
        r'^METHODS\s*$',
        r'^RESULTS\s*$',
        r'^DISCUSSION\s*$',
        r'^CONCLUSIONS?\s*$',
        r'^REFERENCES\s*$',
        r'^INDICATIONS AND USAGE\s*$',
        r'^DOSAGE AND ADMINISTRATION\s*$',
        r'^CONTRAINDICATIONS\s*$',
        r'^WARNINGS AND PRECAUTIONS\s*$',
        r'^ADVERSE REACTIONS\s*$',
        r'^CLINICAL STUDIES\s*$',
        r'^CLINICAL PHARMACOLOGY\s*$',
        r'^\d+\.?\s+[A-Z][A-Za-z\s]{3,}$',  # Numbered sections
    ]
    
    lines = text.split('\n')
    sections = []
    current_section = "HEADER"
    current_text = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if this line is a section header
        is_header = False
        for pattern in section_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                # Save previous section
                if current_text:
                    sections.append((current_section, '\n'.join(current_text)))
                
                current_section = line_stripped
                current_text = []
                is_header = True
                break
        
        if not is_header:
            current_text.append(line)
    
    # Add final section
    if current_text:
        sections.append((current_section, '\n'.join(current_text)))
    
    # If no sections found, return whole document
    if len(sections) <= 1:
        return [("FULL DOCUMENT", text)]
    
    return sections


def chunk_section(section_title, section_text, max_chars=1200, overlap_chars=200):
    """
    Chunk a section into smaller pieces with overlap.
    Similar to dupixent chunking strategy.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
    
    chunks = []
    current = ""
    
    for para in paragraphs:
        # If paragraph is very long, split on sentences
        if len(para) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
        else:
            # Try to add paragraph to current chunk
            if len(current) + len(para) + 2 <= max_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para
    
    if current:
        chunks.append(current)
    
    # Add overlap between chunks
    if len(chunks) > 1 and overlap_chars > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(tail + "\n\n" + chunks[i])
        return overlapped
    
    return chunks


def build_chunks(doc_key, doc_config):
    """
    Load a doc, split into sections, chunk each section.
    """
    path = doc_config["path"]
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = clean_text(text)
    sections = split_into_sections(text, doc_key)
    
    if not sections:
        sections = [("FULL DOCUMENT", text)]
    
    all_chunks = []
    
    for section_title, section_text in sections:
        chunks = chunk_section(section_title, section_text)
        for i, chunk in enumerate(chunks):
            full_text = f"[{section_title}]\n\n{chunk}"
            
            chunk_id = f"{doc_key}__{section_title[:40]}__chunk{i}"
            chunk_id = re.sub(r"[^a-zA-Z0-9_\-]", "", chunk_id)
            
            all_chunks.append({
                "id": chunk_id,
                "text": full_text,
                "metadata": {
                    "source": doc_config["source"],
                    "doc_type": doc_config["doc_type"],
                    "doc_key": doc_key,
                    "section": section_title[:100],
                    "chunk_index": i,
                },
            })
    
    return all_chunks


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(db_path, reset=False):
    print(f"\nCardiovex — ChromaDB Ingestion")
    print(f"DB path: {db_path}")
    print("=" * 60)
    
    client = chromadb.PersistentClient(path=db_path)
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"Collection already has {existing_count} chunks.")
        print("Use --reset to rebuild from scratch.\n")
        print_summary(collection)
        return
    
    # Process documents
    total_chunks = 0
    for doc_key, doc_config in DOCS.items():
        print(f"\nProcessing: {doc_config['source']}")
        chunks = build_chunks(doc_key, doc_config)
        if not chunks:
            continue
        
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
        
        sections = set(c["metadata"]["section"] for c in chunks)
        print(f"  Sections found: {len(sections)}")
        print(f"  Chunks loaded:  {len(chunks)}")
        total_chunks += len(chunks)
    
    print(f"\n{'='*60}")
    print(f"Total chunks in collection: {collection.count()}")
    print_summary(collection)


def print_summary(collection):
    """Print breakdown by source and run sample retrieval tests."""
    print("\nCollection contents:")
    results = collection.get(include=["metadatas"])
    sources = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {count:3d} chunks  —  {src}")
    
    # Sample retrieval tests
    test_queries = [
        "cardiovascular risk reduction",
        "primary endpoint results",
        "adverse events safety",
        "dosing administration",
    ]
    
    print("\nSample retrieval tests:")
    for query in test_queries:
        res = collection.query(query_texts=[query], n_results=1)
        if res["metadatas"] and res["metadatas"][0]:
            meta = res["metadatas"][0][0]
            snippet = res["documents"][0][0][:100].replace("\n", " ")
            print(f"\n  Q: {query}")
            print(f"  → [{meta['source']}] {meta['section'][:50]}")
            print(f"     {snippet}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="data/chroma_db",
        help="Path to ChromaDB directory (default: data/chroma_db)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the collection from scratch",
    )
    args = parser.parse_args()
    
    os.makedirs(args.db, exist_ok=True)
    ingest(db_path=args.db, reset=args.reset)
