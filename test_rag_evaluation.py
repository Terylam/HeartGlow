#!/usr/bin/env python3
"""
RAG Evaluation Script: Compare RAG vs Normal Response with same chatbot
Usage: python test_rag_evaluation.py
"""
import re
import ollama
import os
from pathlib import Path
from typing import List, Tuple, Optional
import time
from datetime import datetime

from sentence_transformers import SentenceTransformer, CrossEncoder

# LangChain imports
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

# Global bi-encoder and cross-encoder instances
bi_encoder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Predefined 20 depression-related questions
DEPRESSION_QUESTIONS = [
    "What are the common symptoms of clinical depression?",
    "How long does depression typically last without treatment?",
    "What are some evidence-based treatments for depression?",
    "How can I help a friend who is experiencing depression?",
    "What is the difference between sadness and clinical depression?",
    "Can depression affect physical health? If so, how?",
    "What are the warning signs that someone may be at risk of suicide?",
    "How does therapy help in treating depression?",
    "What are the side effects of common antidepressant medications?",
    "Can lifestyle changes like exercise help with depression?",
    "What is seasonal affective disorder and how is it treated?",
    "How does depression affect relationships and social life?",
    "What role does genetics play in depression?",
    "Can children and adolescents experience depression?",
    "How can I manage negative thoughts during a depressive episode?",
    "What is the relationship between anxiety and depression?",
    "How does sleep affect depression and vice versa?",
    "What are some coping strategies for dealing with depression at work?",
    "How can I support a loved one with depression without burning out?",
    "What resources are available for people with depression?"
]


class QwenRAG:
    """Qwen-3 RAG implementation with Ollama - Debugged for FYP Evaluation"""
    
    def __init__(
        self,
        model_name: str = "qwen3-coder-next:cloud",
        ollama_url: str = "http://localhost:11434",
        persist_dir: str = "./chroma_db",
        embedding_model: str = "nomic-embed-text:latest"
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.persist_dir = Path(persist_dir)
        
        # Consistent embedding model is critical for retrieval
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
        
        # Increased chunk size to reduce Omission Risk
        self.text_splitter = None  # Not needed for this evaluation
        self.vectorstore = None

    
    def retrieve(self, query: str, k: int = 5, topic_filter: str = None) -> List[Document]:
        """Retrieve relevant documents for a query using bi-encoder + cross-encoder."""
        # Get script directory and construct absolute paths
        script_dir = Path(__file__).parent
        db_paths = [
            script_dir / "chroma_db77280"
'''            script_dir / "shenlab_db",
            script_dir / "chroma_db",
            script_dir / "therapy_db_48111",
            script_dir / "therapy_db_91890"'''
        ]
        all_candidates = []
        
        for db_path in db_paths:
            # Validation: Ensure the DB file exists before attempting to load
            if db_path.exists() and (db_path / "chroma.sqlite3").exists():
                try:
                    db = Chroma(persist_directory=str(db_path), embedding_function=self.embeddings)
                    # Get top 20 candidates from Chroma for reranking
                    res_with_score = db.similarity_search_with_score(query, k=50)
                    for doc, score in res_with_score:
                        doc.metadata["score"] = score
                        all_candidates.append(doc)
                except Exception as e:
                    print(f"Error searching {db_path}: {e}")
        
        # Step 1: Get top 20 candidates if we don't have enough
        if not all_candidates:
            if self.vectorstore is not None:
                all_candidates = self.vectorstore.similarity_search(query, k=50)
            else:
                return []
        
        # Step 2: Rerank the candidates using cross-encoder for better accuracy
        if all_candidates:
            docs = [doc.page_content for doc in all_candidates]
            pairs = [(query, doc) for doc in docs]
            scores = reranker.predict(pairs)
            
            ranked = sorted(zip(scores, all_candidates), reverse=True, key=lambda x: x[0])
            
            final_results = []
            for score, doc in ranked[:k*2]:  # Get more than filter
                if score < 0.55:
                    continue
                else:
                    # Optional: filter by topic
                    if topic_filter and doc.metadata.get("topic") != topic_filter:
                        continue
                    final_results.append(doc)
                if len(final_results) >= k:
                    break
                    
            return final_results
        return []
    
    def chat_with_rag(self, query: str, history: List[dict] = None, k: int = 3) -> Tuple[str, List[Document]]:
        """
        Multi-turn chat with RAG context.
        
        Args:
            query: User query
            history: List of previous messages
            k: Number of documents to retrieve
            
        Returns:
            Response and retrieved documents
        """
        if history is None:
            history = []
        
        docs = self.retrieve(query, k)
        context = "\n\n".join([doc.page_content for doc in docs])
        print("="*60)
        print(f"\nContext: {context}\n")
        print("="*60)
        
        # Use the first valid document for context, or provide empty context
        valid_docs = [doc for doc in docs if doc.page_content and len(doc.page_content.strip()) > 10]
        if valid_docs:
            context = "\n\n".join([doc.page_content for doc in valid_docs])
        '''
        else:
            context = ""  # Empty context - model should use its own knowledge
        Context from knowledge base RAG (all in English):
  {context}'''
        
        messages = [
            {
                "role": "system",
                "content": f"""[Role: You are a multilingual expert therapist. Your task is to answer the user's question. If relevant context is provided, use it; otherwise, use your own knowledge and expertise. Max around 3 paragraphs.]
Context from knowledge base RAG (all in English):
  {context}
  Instructions:
  1. Context Usage: Use the *Retrieved Context* if relevant, otherwise use your own knowledge.
  2. Formatting: Use **text** for important phrases/bold and *text* for quotes and names/italic (more bold).
  3. Synthesis: Formulate a comprehensive answer. ASK follow-up questions for more info. NO listing steps (like 1. 2. 3.).
  4. Translation: You MUST provide the final response in the SAME language as the "User Query". If the context is in English and the query is in French, translate the information into French.
  5. Professionalism: Act like a therapist."""
         }
        ]
        
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            messages.append(msg)
        
        messages.append({"role": "user", "content": query})

        #response = ollama.chat(model=self.model_name, messages=messages)
        response_text = self.query_remote_ollama(question=messages, host=self.ollama_url, model_name=self.model_name, messages=messages)
        return response_text, docs


    def query_remote_ollama(self, question: str, host: str, model_name: Optional[str] = None, n_results: int = 3, messages: Optional[List[dict]] = None) -> str:
        """Query the RAG system using a remote Ollama instance."""
        try:
            remote_client = ollama.Client(host=host)
            if messages is not None:
                response = remote_client.chat(model=model_name or self.model_name, messages=messages)
                return response['message']['content']
            else:
                # Retrieve context using the internal retrieve method
                docs = self.retrieve(question, k=n_results)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"""

Context:
{context}

Question: {question}

Answer:"""
                
                response = remote_client.generate(model=model_name or self.model_name, prompt=prompt)
                return response['response']
        except Exception as e:
            return f"Error connecting to remote Ollama at {host}: {str(e)}"
    
    def query_normal(self, query: str, history: List[dict] = None) -> str:
        """
        Query the chatbot WITHOUT RAG (normal response without retrieval).
        
        Args:
            query: User query
            history: List of previous messages
            
        Returns:
            Response from chatbot
        """
        if history is None:
            history = []
        
        # Use the same chat method but with empty context (simulating no RAG)
        # We need a separate method that uses the chatbot without retrieval
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and knowledgeable assistant. Provide accurate, empathetic, and well-reasoned responses to the user's question in 2-3 paragraphs (not bullet points)."
            }
        ]
        
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            messages.append(msg)
        
        messages.append({"role": "user", "content": query})
        
        response_text = self.query_remote_ollama(question=query, host=self.ollama_url, model_name=self.model_name, messages=messages)
        return response_text


def format_output(question: str, context: str, rag_response: str, normal_response: str) -> str:
    """
    Format output as specified: Question, Context, Rag_response, Normal response
    All separated with \n\n
    """
    question = "Q: "+ question
    context = "Context: "+ context
    rag_response = "RAG: "+ rag_response
    normal_response = "Normal: "+ normal_response
    re.sub(r'\r?\n', ' ', question)
    re.sub(r'\r?\n', ' ', context)
    re.sub(r'\r?\n', ' ', rag_response)
    re.sub(r'\r?\n', ' ', normal_response)
    output = f"{question}\n\n{context}\n\n{rag_response}\n\n{normal_response}"
    return output


def evaluate_rag_system(
    questions: List[str],
    model_name: str = "qwen3.5:397b-cloud",
    ollama_url: str = "http://localhost:11434",
    k: int = 3,
    output_file: str = None
):
    """
    Evaluate RAG system comparing RAG vs Normal response with same chatbot.
    
    Args:
        questions: List of questions to evaluate
        model_name: Name of the Qwen model
        ollama_url: URL to Ollama server
        k: Number of documents to retrieve
        output_file: Path to save output file
    """
    print(f"Starting RAG Evaluation with {len(questions)} questions...")
    print(f"Model: {model_name}")
    print("-" * 80)
    
    # Initialize RAG system
    rag = QwenRAG(model_name=model_name, ollama_url=ollama_url)
    
    results = []
    total_time = 0
    
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing question: {question[:50]}...")
        
        try:
            start_time = time.perf_counter()
            
            # Get RAG response
            rag_response, docs = rag.chat_with_rag(question, k=k)
            
            # Get normal response (without RAG)
            normal_response = rag.query_normal(question)
            
            # Extract context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            elapsed_time = time.perf_counter() - start_time
            total_time += elapsed_time
            
            # Store results
            results.append({
                "question": question,
                "context": context,
                "rag_response": rag_response,
                "normal_response": normal_response
            })
            
            print(f"  ✓ Completed in {elapsed_time:.2f}s")
            print(f"  RAG response: {rag_response[:100]}...")
            print(f"  Normal response: {normal_response[:100]}...")
            
        except Exception as e:
            print(f"  ✗ Error processing question {i}: {e}")
            results.append({
                "question": question,
                "context": f"ERROR: {str(e)}",
                "rag_response": f"ERROR: {str(e)}",
                "normal_response": f"ERROR: {str(e)}"
            })
    
    # Build output
    output_parts = []
    for result in results:
        output_parts.append(format_output(
            result["question"],
            result["context"],
            result["rag_response"],
            result["normal_response"]
        ))
    
    full_output = "\n\n".join(output_parts) #format!!!
    
    # Save to file if output_file is specified
    if output_file is None:
        output_file = f"./rag_evaluation_rag_vs_normal_qwen_3_0507_.txt" #{datetime.now().strftime('%Y%m%d_%H%M%S')}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAG EVALUATION: RAG vs Normal Response\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of Questions: {len(questions)}\n")
        f.write(f"Total Time: {total_time:.2f}s\n")
        f.write("=" * 80 + "\n\n")
        f.write(full_output)
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per question: {total_time/len(questions):.2f}s")
    print("=" * 80)
    
    return full_output


def main():
    # Evaluate with predefined 20 depression-related questions
    evaluate_rag_system(
        questions=DEPRESSION_QUESTIONS,
        model_name="qwen3-coder-next:cloud", #qwen3:8b gemini-3-flash-preview:cloud nemotron-3-super:cloud
        ollama_url="http://localhost:11434",
        k=3
    )


if __name__ == "__main__":
    main()