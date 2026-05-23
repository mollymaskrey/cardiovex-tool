"""
ingest.py

Build FAISS index from Cardiovex source documents.

Usage:
    python ingest.py
"""

import os
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer


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

OUTPUT_DIR = "data/faiss_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def clean_text(text):
    """Basic text cleanup."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def split_into_sections(text, doc_key):
    """Split document into logical sections."""
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
        r'^\d+\.?\s+[A-Z][A-Za-z\s]{3,}$',
    ]
    
    lines = text.split('\n')
    sections = []
    current_section = "HEADER"
    current_text = []
    
    for line in lines:
        line_stripped = line.strip()
        is_header = False
        
        for pattern in section_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                if current_text:
                    sections.append((current_section, '\n'.join(current_text)))
                current_section = line_stripped
                current_text = []
                is_header = True
                break
        
        if not is_header:
            current_text.append(line)
    
    if current_text:
        sections.append((current_section, '\n'.join(current_text)))
    
    if len(sections) <= 1:
        return [("FULL DOCUMENT", text)]
    
    return sections


def chunk_section(section_title, section_text, max_chars=1200, overlap_chars=200):
    """Chunk a section into smaller pieces with overlap."""
    paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
    
    chunks = []
    current = ""
    
    for para in paragraphs:
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
            if len(current) + len(para) + 2 <= max_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para
    
    if current:
        chunks.append(current)
    
    # Add overlap
    if len(chunks) > 1 and overlap_chars > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(tail + "\n\n" + chunks[i])
        return overlapped
    
    return chunks


def build_chunks(doc_key, doc_config):
    """Load a doc, split into sections, chunk each section."""
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
            
            metadata = {
                "source": doc_config["source"],
                "doc_type": doc_config["doc_type"],
                "doc_key": doc_key,
                "section": section_title[:100],
                "chunk_index": i,
            }
            
            all_chunks.append({
                "text": full_text,
                "metadata": metadata,
            })
    
    return all_chunks


# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------

def ingest():
    print("\nCardiovex — FAISS Index Build")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Collect all chunks
    all_chunks = []
    all_metadatas = []
    
    for doc_key, doc_config in DOCS.items():
        print(f"\nProcessing: {doc_config['source']}")
        chunks = build_chunks(doc_key, doc_config)
        
        if not chunks:
            continue
        
        texts = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        all_chunks.extend(texts)
        all_metadatas.extend(metadatas)
        
        sections = set(m['section'] for m in metadatas)
        print(f"  Sections found: {len(sections)}")
        print(f"  Chunks created: {len(chunks)}")
    
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    
    # Encode all chunks
    print(f"\nEncoding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index_file = os.path.join(OUTPUT_DIR, "index.pkl")
    
    with open(index_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'chunks': all_chunks,
            'metadatas': all_metadatas,
        }, f)
    
    print(f"\nFAISS index saved to: {index_file}")
    print(f"Size: {embeddings.shape}")
    
    # Test retrieval
    print("\nSample retrieval test:")
    test_query = "cardiovascular risk reduction"
    query_embedding = model.encode([test_query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    similarities = np.dot(embeddings, query_embedding)
    top_idx = np.argmax(similarities)
    
    print(f"  Query: {test_query}")
    print(f"  Top result: {all_metadatas[top_idx]['source']} - {all_metadatas[top_idx]['section'][:50]}")
    print(f"  Score: {similarities[top_idx]:.3f}")
    snippet = all_chunks[top_idx][:150].replace('\n', ' ')
    print(f"  Text: {snippet}...")


if __name__ == "__main__":
    ingest()