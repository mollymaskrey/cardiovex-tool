"""
rag.py

FAISS-based RAG system for Cardiovex documents.
Lightweight, no ChromaDB, no Rust dependencies.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class CardiovexRAG:
    def __init__(self, db_path="data/faiss_db"):
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "index.pkl")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load prebuilt index
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_file}. "
                f"Run 'python ingest.py' first."
            )
        
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.chunks = data['chunks']
            self.metadatas = data['metadatas']
    
    def retrieve(self, query, n_results=5):
        """Retrieve top N chunks for a query."""
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top N indices
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'metadata': self.metadatas[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def format_context(self, chunks):
        """Format retrieved chunks into context string."""
        if not chunks:
            return "(No context retrieved.)"
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            source = meta.get('source', 'Unknown')
            section = meta.get('section', '')
            text = chunk['text']
            
            context_parts.append(f"[{source} - {section}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)