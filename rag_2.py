#!/usr/bin/env python3
"""
RAG System v2 using ChromaDB and Ollama

This module provides a RAG pipeline with:
- ChromaDB for vector storage
- Ollama for LLM generation
- Support for Text and CSV ingestion
- Document merging capabilities

Requirements:
    pip install chromadb ollama pandas
"""

import os
import uuid
import time
import json
import pandas as pd
from typing import List, Dict, Optional, Any
import ollama
import requests

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("Please install chromadb: pip install chromadb")

class RAGSystemV2:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./shenlab_db"):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Using default Chroma embedding function (sentence-transformers)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_fn
        )
        self.model_name = "qwen3:8b"

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a single text document to the collection."""
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        return doc_id

    def add_csv(self, file_path: str, text_column: str, metadata_columns: Optional[List[str]] = None):
        """Add documents from a CSV file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        documents = df[text_column].astype(str).tolist()
        
        metadatas = []
        if metadata_columns:
            metadatas = df[metadata_columns].to_dict('records')
        else:
            metadatas = [{} for _ in range(len(documents))]
            
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} rows from {file_path}")

    def merge_documents(self, doc_ids: List[str], new_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Merge multiple documents into a single new document.
        Retrieves existing documents by ID, concatenates them, and adds a new entry.
        """
        results = self.collection.get(ids=doc_ids)
        if not results['documents']:
            raise ValueError("No documents found for the provided IDs")
            
        merged_text = "\n---\n".join(results['documents'])
        
        # Combine existing metadatas if none provided
        if not new_metadata:
            new_metadata = {"merged_from": ",".join(doc_ids), "type": "merged"}
            
        return self.add_text(merged_text, new_metadata)

    def query(self, question: str, n_results: int = 3) -> str:
        """Query the RAG system using Ollama."""
        # Retrieve context from ChromaDB
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        context = "\n\n".join(results['documents'][0])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query_openrouter(self, question: str, model_name: str = "google/gemma-3-27b-it:free", api_key: Optional[str] = None, n_results: int = 3) -> str:
        """Query the RAG system using OpenRouter (OpenAI-compatible API)."""
        # Retrieve context from ChromaDB
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        context = "\n\n".join(results['documents'][0])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know.

Context:
{context}

Question: {question}

Answer:"""

        # Use provided key or check environment variable
        actual_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not actual_api_key:
            return "Error: OpenRouter API key not found. Please provide it or set OPENROUTER_API_KEY environment variable."

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {actual_api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"OpenRouter Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Exception during OpenRouter query: {str(e)}"

    def query_remote_ollama(self, question: str, host: str, model_name: Optional[str] = None, n_results: int = 3) -> str:
        """Query the RAG system using a remote Ollama instance."""
        # Retrieve context from ChromaDB
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        context = "\n\n".join(results['documents'][0])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know.

Context:
{context}

Question: {question}

Answer:"""

        try:
            # Create a client for the remote host
            remote_client = ollama.Client(host=host)
            response = remote_client.generate(model=model_name or self.model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error connecting to remote Ollama at {host}: {str(e)}"

def main():
    # Example Usage
    rag = RAGSystemV2(collection_name="test_collection")
    
    '''    print("\n--- 1. Adding Text Documents ---")
    id1 = rag.add_text("The capital of France is Paris.", {"subject": "geography"})
    id2 = rag.add_text("The Eiffel Tower is located in Paris.", {"subject": "landmarks"})
    print(f"Added text docs: {id1}, {id2}")
    
    print("\n--- 2. Adding CSV Data ---")
    # Assuming test_data.csv exists with 'content' column
    try:
        rag.add_csv("test_data.csv", text_column="content", metadata_columns=["author"])
    except Exception as e:
        print(f"CSV error: {e}")
    
    print("\n--- 3. Merging Documents ---")
    merged_id = rag.merge_documents([id1, id2], {"subject": "Paris info"})
    print(f"Merged document created with ID: {merged_id}")
    
    print("\n--- 4. Interactive Chat (type 'exit' or Ctrl+C to quit) ---")
    print("Prefixes: 'OR:' for OpenRouter, 'REM:' for Remote Ollama. Default: Local Ollama")
    '''
    session_start = time.perf_counter()
    try:
        while True:
            prompt_str = "\nAsk a question: "
            question = input(prompt_str).strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            if not question:
                continue

            try:
                print("Thinking...")
                req_start = time.perf_counter()
                
                if question.upper().startswith("OR:"):
                    # Use OpenRouter
                    query_text = question[3:].strip()
                    answer = rag.query_openrouter(query_text)
                elif question.upper().startswith("REM:"):
                    # Use Remote Ollama
                    # Example host: http://your-cloud-ip:11434
                    remote_host = os.getenv("OLLAMA_REMOTE_HOST", "http://localhost:11434")
                    query_text = question[4:].strip()
                    answer = rag.query_remote_ollama(query_text, host=remote_host)
                else:
                    # Use local Ollama
                    answer = rag.query(question)
                    
                req_end = time.perf_counter()
                
                print(f"Answer: {answer}")
                print(f"Time taken for this request: {req_end - req_start:.4f} seconds")
            except Exception as e:
                print(f"Query error: {e}")
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected.")
    
    session_end = time.perf_counter()
    print(f"\nExiting session...")
    print(f"Total session time: {session_end - session_start:.2f} seconds")

if __name__ == "__main__":
    main()