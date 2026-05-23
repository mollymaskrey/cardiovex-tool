"""
rag.py

RAG (Retrieval-Augmented Generation) for Cardiovex field rep support.
Retrieves relevant chunks from ChromaDB and generates responses via Anthropic API.
"""

import os
from typing import List, Dict, Optional

import anthropic
import chromadb
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "cardiovex_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# RAG Class
# ---------------------------------------------------------------------------

class CardiovexRAG:
    """RAG system for Cardiovex product knowledge."""
    
    def __init__(self, db_path: str = "data/chroma_db", api_key: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            db_path: Path to ChromaDB directory
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.db_path = db_path
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"]["source"]
            section = chunk["metadata"]["section"]
            text = chunk["text"]
            
            context_parts.append(
                f"[Source {i}: {source} — {section}]\n{text}\n"
            )
        
        return "\n---\n\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        persona: str = "cardiologist_visit",
        n_results: int = 5,
        temperature: float = 0.0
    ) -> Dict:
        """
        Generate a response using RAG.
        
        Args:
            query: User's question
            persona: Persona key from personas.py
            n_results: Number of chunks to retrieve
            temperature: LLM temperature (0.0 for deterministic)
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, n_results=n_results)
        
        # Format context
        context = self.format_context(chunks)
        
        # Get persona prompt
        persona_prompt = get_persona_prompt(persona, product_name="Cardiovex")
        
        # Build the full prompt
        system_prompt = f"""{persona_prompt}

You will be provided with relevant excerpts from Cardiovex documentation.
Use ONLY the information provided in these excerpts to answer questions.

If the information needed to answer a question is not in the provided excerpts, say so clearly.
Do not make up information or extrapolate beyond what is explicitly stated.

For safety questions, adverse events, or contraindications, be especially precise and cite the specific source.
"""
        
        user_prompt = f"""Here is the relevant documentation:

{context}

---

Question: {query}

Please provide a clear, accurate answer based solely on the documentation above."""
        
        # Call Anthropic API
        message = self.anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        response_text = message.content[0].text
        
        return {
            "response": response_text,
            "query": query,
            "persona": persona,
            "chunks_retrieved": len(chunks),
            "sources": [
                {
                    "source": chunk["metadata"]["source"],
                    "section": chunk["metadata"]["section"],
                    "distance": chunk["distance"]
                }
                for chunk in chunks
            ],
            "model": ANTHROPIC_MODEL,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens
            }
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def ask(
    query: str,
    persona: str = "cardiologist_visit",
    db_path: str = "data/chroma_db",
    n_results: int = 5
) -> str:
    """
    Quick interface to ask a question.
    
    Args:
        query: Question to ask
        persona: Persona to use
        db_path: Path to ChromaDB
        n_results: Number of chunks to retrieve
        
    Returns:
        Response text
    """
    rag = CardiovexRAG(db_path=db_path)
    result = rag.generate_response(query, persona=persona, n_results=n_results)
    return result["response"]


# ---------------------------------------------------------------------------
# Test/Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    # Quick test
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What were the primary endpoints in the SHIELD trials?"
    
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    rag = CardiovexRAG()
    result = rag.generate_response(query, persona="clinical_trial")
    
    print(f"\nResponse:\n{result['response']}")
    print(f"\n{'-' * 60}")
    print(f"Chunks retrieved: {result['chunks_retrieved']}")
    print(f"Input tokens: {result['usage']['input_tokens']}")
    print(f"Output tokens: {result['usage']['output_tokens']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['source']} — {source['section']}")
