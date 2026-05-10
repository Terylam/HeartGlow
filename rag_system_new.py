import ollama
import os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import time
import re
#change the prompt!!!

# Bi-Encoder and Cross-Encoder imports
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

class QwenRAG:
    """Qwen-3 RAG implementation with Ollama - Debugged for FYP Evaluation"""
    
    def __init__(
        self,
        model_name: str = "qwen3:8b",
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
        
        '''        # Increased chunk size to reduce Omission Risk
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "!", "?", "\n\n\n"]
        )'''
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create vector store with explicit string path casting for Chroma."""
        splits = self.text_splitter.split_documents(documents)
        # Cast Path to str to prevent serialization errors
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        return self.vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store."""
        if self.persist_dir.exists():
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            return self.vectorstore
        return None
    
    def retrieve(self, query: str, k: int = 5, topic_filter: str = None) -> List[Document]:
        """Retrieve relevant documents for a query using bi-encoder + cross-encoder."""
        # Get script directory and construct absolute paths
        script_dir = Path(__file__).parent
        db_paths = [
            script_dir / "shenlab_db",
            script_dir / "chroma_db",
            script_dir / "therapy_db_48111",
            script_dir / "therapy_db_91890"
        ]
        all_candidates = []
        
        for db_path in db_paths:
            # Validation: Ensure the DB file exists before attempting to load
            if db_path.exists() and (db_path / "chroma.sqlite3").exists():
                try:
                    db = Chroma(persist_directory=str(db_path), embedding_function=self.embeddings)
                    
                    res_with_score = db.similarity_search_with_score(query, k=50)
                    for doc, score in res_with_score:
                        doc.metadata["score"] = score
                        all_candidates.append(doc)
                except Exception as e:
                    print(f"Error searching {db_path}: {e}")
        
        # Step 1: 
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
                if score > 0.4:
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
        re.sub(r'\r?\n', ' ', context)
        
        # Use the first valid document for context, or provide empty context
        valid_docs = [doc for doc in docs if doc.page_content and len(doc.page_content.strip()) > 10]
        if valid_docs:
            context = "\n\n".join([doc.page_content for doc in valid_docs])
          # Empty context - model should use its own knowledge
        role = "[Role: You are a multilingual expert therapist. Your task is to answer the user's question. If relevant context is provided, use it; otherwise, use your own knowledge and expertise. Max 3 paragraphs] Context from knowledge base RAG (all in English):"
        instructions = " Instructions: 1. Context Usage: Use the *Retrieved Context* if relevant, otherwise use your own knowledge. 2. Formatting: Use **text** for important phrases/bold and *text* for quotes and names/italic (use more bold). 3. Synthesis: Formulate a comprehensive answer. ASK follow-up questions for more info. NO listing steps (like 1. 2. 3.). 4. Translation: You MUST provide the final response in the SAME language as the *User Query*. If language unknown, use English. 5. Professionalism: Act like a therapist."
        prompt = role+context+instructions
        re.sub(r'\r?\n', ' ', prompt)
        messages = [
            {
                "role": "system",
                "content": f"""{prompt}"""
         }
        ]

        '''        messages = [
            {
                "role": "system",
                "content": f"""[Role: You are a multilingual expert therapist. Your task is to answer the user's question using ONLY the provided context. Be concise in at most 2 to 3 paragraphs.]
  Context from knowledge base RAG (all in English):
  {context}
  Instructions:
  1. Analysis: Read the "Retrieved Context" (which may be in a different language).
  2. Formatting: Use **text** for important phrases/bold and *text* for quotes and names/italic (use more bold).
  3. Synthesis: Formulate a comprehensive answer based on the context. ASK follow-up questions for more info. NO listing steps (like 1. 2. 3.).
  4. Translation: You MUST provide the final response in the SAME language as the "User Query". If the context is in English and the query is in French, translate resopnse to French. If language unknwon, should use English.
  5. Professionalism: Act like a therapist. Never mention if the context is available or good enough.
  6. Important: If the context contains no relevant information, respond with "I couldn't find any relevant information to answer your question." and follow up question"""
         }
        ]'''
        
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            messages.append(msg)
        
        messages.append({"role": "user", "content": query})

        #response = ollama.chat(model=self.model_name, messages=messages)
        response_text = self.query_remote_ollama(question=messages, host=self.ollama_url, model_name=self.model_name, messages=messages)
        return response_text, docs

    def load_qa_csv(self, file_path: str) -> List[Document]:
        """Load Q&A pairs from CSV file using semicolon delimiter."""
        try:
            df = pd.read_csv(file_path, delimiter=';', quotechar='"', engine='python')
            
            if 'input' not in df.columns or 'output' not in df.columns:
                raise ValueError("CSV must contain 'input' and 'output' columns")
            
            documents = []
            for idx, row in df.iterrows():
                content = f"Q: {row['input']}\nA: {row['output']}"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "question": str(row['input']),
                        "answer": str(row['output']),
                        "row_index": idx
                    }
                )
                documents.append(doc)
            
            return documents
            
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
            return []

    def load_text(self, file_path: str) -> List[Document]:
        """Load documents from text file."""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return documents

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


def rag_generate(
    query: str,
    csv_path: str = None,
    text_path: str = None,
    persist_dir: str = "./chroma_db",
    model_name: str = "qwen3-coder-next:cloud",#"qwen3:8b",
    ollama_url: str = "http://localhost:11434",
    k: int = 3,
    history: List[dict] = None
) -> str:
    """
    Generate a response using RAG.
    
    Args:
        query: User query
        csv_path: Path to CSV file with Q&A data
        text_path: Path to text file with data
        persist_dir: Directory for vector store
        model_name: Name of the Qwen model
        ollama_url: URL to Ollama server
        k: Number of documents to retrieve
        history: List of previous messages for chat history
        
    Returns:
        Generated response string
    """
    rag = QwenRAG(model_name=model_name, ollama_url=ollama_url, persist_dir=persist_dir)
    
    documents = []
    
    # Load data from CSV
    if csv_path and os.path.exists(csv_path):
        docs = rag.load_qa_csv(csv_path)
        documents.extend(docs)
    
    # Load data from text file
    if text_path and os.path.exists(text_path):
        docs = rag.load_text(text_path)
        documents.extend(docs)
    
    # Load existing vector store or create new one
    vectorstore = rag.load_vectorstore()
    if vectorstore is None:
        if documents:
            vectorstore = rag.create_vectorstore(documents)
        else:
            raise ValueError("No data source provided and no existing vector store found.")
    else:
        # If vectorstore was loaded, re-add documents if any were provided
        if documents:
            vectorstore = rag.create_vectorstore(documents)
    
    # Use chat_with_rag for response
    response, _ = rag.chat_with_rag(query, history, k)
    return response

def test_csv():
    import csv
    from datetime import datetime
    
    # Configuration
    csv_path = "/Users/terencelam/Downloads/Synthetic_Data_10K_@24.csv"
    persist_dir = "./shenlab_db"
    output_file = f"./rag_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    num_questions = 30
    md_name = "qwen3-coder-next:cloud"

    print(f"Evaluating RAG system with {num_questions} questions...")
    print(f"Output file: {output_file}")
    print("-" * 80)
    
    # Read questions from CSV
    questions = []
    ground_truth_answers = []
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        sampled = df.sample(n=num_questions, random_state=42)
        
        for idx, row in sampled.iterrows():
            questions.append(row.get('input', ''))
            ground_truth_answers.append(row.get('output', ''))
        print(f"Loaded {len(questions)} questions from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        questions = ["I can hardly make friends, what should I do?"]
        ground_truth_answers = ["Building friendships takes time and effort..."]
    
    # Initialize RAG system for normal queries
    rag = QwenRAG(model_name=md_name, ollama_url="http://localhost:11434", persist_dir=persist_dir)
    
    # Evaluate RAG system
    results = []
    time_lapsed = 0
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers), 1):
        print(f"[{i}/{len(questions)}] Processing question...")
        try:
            start_time = time.perf_counter()
            
            # Retrieve documents to get context
            retrieved_docs = rag.retrieve(question, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Get RAG response
            rag_response = rag_generate(
                query=question,
                csv_path=csv_path,
                persist_dir=persist_dir,
                model_name=md_name,
                k=3
            )
            # Get Normal (non-RAG) response using query_normal method
            normal_response = rag.query_normal(query=question)
            
            print(f"Q: {question}")
            print()  
            print(f"A:{rag_response}")
            print()
            
            results.append({
                "question": question,
                "rag_response": rag_response,
                "normal_response": normal_response,
                "dataset_answer": ground_truth,
                "context": context
            })
            print(f"total time: {round((time.perf_counter() - start_time),2)}")
            time_lapsed += round((time.perf_counter() - start_time),2)

        except KeyboardInterrupt:
            print()
            print(f"Time used = {time_lapsed}")
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"  Error processing question {i}: {e}")
            print(f"Time used = {time_lapsed}")
            results.append({
                "question": question,
                "rag_response": f"ERROR: {str(e)}",
                "normal_response": f"ERROR: {str(e)}",
                "dataset_answer": ground_truth,
                "context": ""
            })
    
    # Write results to file in formal format
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG SYSTEM EVALUATION RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV Source: {csv_path}\n")
            f.write(f"Model Name: {md_name}\n")
            f.write(f"Number of Questions: {len(questions)}\n")
            f.write(f"Total time: {time_lapsed:.2f} s\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{'=' * 40}\n")
                f.write(f"Question #{i}\n")
                f.write(f"{'=' * 40}\n\n")
                
                f.write("Q: ")
                f.write(re.sub(r'\r?\n', ' ', result["question"]))
                f.write("\n")
                
                f.write("Context: ")
                f.write(re.sub(r'\r?\n', ' ', result["context"]))
                f.write("\n")
                
                f.write("RAG:\n")
                f.write(re.sub(r'\r?\n', ' ', result["rag_response"]))
                f.write("\n")
                
                f.write("Normal:\n")
                f.write(re.sub(r'\r?\n', ' ', result["normal_response"]))
                f.write("\n\n")
        
        output_file_1line = output_file.replace('.txt', '_1line.txt')
        with open(output_file_1line, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG SYSTEM EVALUATION RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV Source: {csv_path}\n")
            f.write(f"Model Name: {md_name}\n")
            f.write(f"Number of Questions: {len(questions)}\n")
            f.write(f"Total time: {time_lapsed:.2f} s\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write("QUESTION:\n")
                f.write(re.sub(r'\r?\n', ' ', result["question"]))
                f.write("\n\n")
                
                f.write("CONTEXT:\n")
                f.write(re.sub(r'\r?\n', ' ', result["context"]))
                f.write("\n\n")
                
                f.write("RAG RESPONSE:\n")
                f.write(re.sub(r'\r?\n', ' ', result["rag_response"]))
                f.write("\n\n")
                
                f.write("NORMAL RESPONSE:\n")
                f.write(re.sub(r'\r?\n', ' ', result["normal_response"]))
                f.write("\n\n")
                
                f.write("=" * 40 + "\n\n")

        print("-" * 80) 
        print(f"Results saved to: {output_file}")
        print(f"Successfully evaluated {len([r for r in results if 'ERROR' not in r['rag_response']])}/{len(results)} questions")
        print()
        print(f"Total time: {time_lapsed}")
    except KeyboardInterrupt:
        print()
        print(f"Time used = {time_lapsed}")
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error writing results file: {e}")

# Standalone function alias for compatibility with test_rag_evaluation.py
chat_with_rag = QwenRAG.chat_with_rag

def chat():
    persist_dir = "./shenlab_db"
    print("\n" + "=" * 60)
    print("Starting Chat Interface")
    print("=" * 60)
    print("Type 'exit' or 'quit' to quit.\n")
    time_lapsed = 0
    init_chat = 0
    modelname = 'qwen3-coder-next:cloud' #'qwen3-coder-next:cloud'
    #'qwen3:8b'
    # 'qwen3.5:9b'
    # 'qwen3.5:397b-cloud'
    # 'medllama2:latest'
    #modelname="qwen3:8b"
    while True:
        
        try:
            start_time = time.perf_counter()
            # Get RAG response
            if init_chat == 0:
                question = "Please greet the user."
                init_chat += 1
            else:
                question = input("\nYou: ").strip()
                init_t = time.perf_counter()
                if question.lower() in ['exit', 'quit', 'q']:
                    print(f"Time used = {time_lapsed}")
                    print("Goodbye!")
                    break
            
            if not question:
                continue
            
            print(f"\n{str(modelname)}: ", end="", flush=True)
                            #model_name='deepseek-r1:8b',
            rag_response = rag_generate(
                query=question,
                persist_dir=persist_dir,
                model_name=modelname,
                k=3
            )
            time_lapsed += round((time.perf_counter() - start_time),2)
            print(rag_response)
            print(f"Time used = {round((time.perf_counter() - start_time),2)}")

        except KeyboardInterrupt:
            print(f"\nTime used = {time_lapsed:2f}")
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":

    while True:
        try:
            mode = input("Do you want to test or chat normally (type 'test' or 'chat')? :")
            if mode == "test":
                test_csv()
                break
            elif mode == "chat":
                chat()
                break
            else: 
                raise ValueError
        except ValueError:
            print("Invalid mode!")