import os
import numpy as np
import requests
from typing import List, Dict, Union
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import textwrap
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tempfile

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    """Handles document loading and chunking."""
    
    @staticmethod
    def load_document(source: str) -> str:
        """Load document from various sources (file path or URL)."""
        # Check if source is a URL
        if DocumentProcessor._is_url(source):
            return DocumentProcessor._load_from_url(source)
        # Otherwise, treat as file path
        elif os.path.exists(source):
            if source.endswith('.pdf'):
                return DocumentProcessor._load_pdf(source)
            elif source.endswith('.docx'):
                return DocumentProcessor._load_docx(source)
            elif source.endswith('.txt'):
                return DocumentProcessor._load_text(source)
            elif source.endswith('.js') or source.endswith('.ts'):
                return DocumentProcessor._load_js_ts(source)
            elif source.endswith('.html'):
                return DocumentProcessor._load_html(source)
            elif source.endswith('.css'):
                return DocumentProcessor._load_css(source)
            else:
                raise ValueError(f"Unsupported file format: {source}")
        else:
            raise ValueError(f"Source not found or invalid: {source}")
    
    @staticmethod
    def load_project_folder(folder_path: str, supported_extensions: List[str] = ['.txt', '.pdf', '.docx', '.js', '.html', '.ts', '.css']) -> Dict[str, str]:
        """Load all supported documents from a project folder."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Project folder not found: {folder_path}")
        
        documents = {}
        
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() in supported_extensions:
                    try:
                        content = DocumentProcessor.load_document(file_path)
                        # Use relative path as document key
                        rel_path = os.path.relpath(file_path, folder_path)
                        documents[rel_path] = content
                        print(f"Loaded: {rel_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
        
        if not documents:
            print(f"No supported documents found in {folder_path}")
        
        return documents
    
    @staticmethod
    def _is_url(text: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def _load_from_url(url: str) -> str:
        """Load content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Handle different content types
            if 'text/html' in content_type:
                # Extract text from HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text = soup.get_text(separator=' ', strip=True)
                
                # Clean text (remove excessive whitespace)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text
                
            elif 'application/pdf' in content_type:
                # Download and process PDF
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                try:
                    text = DocumentProcessor._load_pdf(temp_path)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                
                return text
                
            else:
                # Default to plain text
                return response.text
                
        except Exception as e:
            raise ValueError(f"Failed to load content from URL: {str(e)}")
    
    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """Load content from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def _load_docx(file_path: str) -> str:
        """Load content from DOCX file."""
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    @staticmethod
    def _load_text(file_path: str) -> str:
        """Load content from text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def _load_js_ts(file_path: str) -> str:
        """Load content from JavaScript or TypeScript file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def _load_html(file_path: str) -> str:
        """Load content from HTML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            # Clean text (remove excessive whitespace)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
    
    @staticmethod
    def _load_css(file_path: str) -> str:
        """Load content from CSS file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def chunk_document(text: str, chunk_size: int = 5) -> List[str]:
        """Split document into chunks of sentences."""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i+chunk_size])
            chunks.append(chunk)
            
        return chunks


class EmbeddingGenerator:
    """Generates and manages embeddings for document chunks."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for document chunks."""
        return self.model.encode(chunks)
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        return self.model.encode([query])


class DocumentRetriever:
    """Retrieves relevant document chunks for a query."""
    
    @staticmethod
    def retrieve_chunks(query_embedding: np.ndarray, 
                        chunk_embeddings: np.ndarray,
                        chunks: List[str],
                        sources: List[str],
                        top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve top-k relevant chunks for the query with their sources."""
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            results.append({
                "chunk": chunks[i],
                "source": sources[i],
                "similarity": similarities[i]
            })
        
        return results


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        """Initialize Ollama client with host and model."""
        self.host = host
        self.model = model
        self.api_generate = f"{self.host}/api/generate"
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Ollama API returned status code {response.status_code}")
                available_models = []
            else:
                available_models = [model["name"] for model in response.json().get("models", [])]
                
            if available_models and self.model not in available_models:
                print(f"Warning: Model '{self.model}' not found in available models: {available_models}")
                if available_models:
                    print(f"Using '{available_models[0]}' instead")
                    self.model = available_models[0]
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to Ollama at {self.host}")
            print("Make sure Ollama is running, or specify a different host")
    
    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        """Generate text using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_generate, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {str(e)}")
            return f"Error generating response: {str(e)}"


class AnswerGenerator:
    """Generates answers based on retrieved document chunks and query."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        """Initialize the LLM client."""
        self.client = OllamaClient(host=host, model=model)
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, str]], include_sources: bool = True) -> str:
        """Generate an answer using Ollama."""
        context_parts = []
        
        for idx, item in enumerate(retrieved_chunks, 1):
            chunk = item["chunk"]
            source = item["source"]
            context_parts.append(f"[{idx}] Source: {source}\nContent: {chunk}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant that answers questions based only on the provided context. 
If the answer cannot be found in the context, say so clearly. Do not make up information. Please act as if you are the person who answers the questions. Don't mention that your answers are based on a provided document."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer the question based only on the provided context."""
        
        answer = self.client.generate(user_prompt, system=system_prompt)
        
        if include_sources:
            sources = [f"- {item['source']}" for item in retrieved_chunks]
            sources_str = "\n".join(sources)
            answer += f"\n\nSources:\n{sources_str}"
        
        return answer


class RAGApplication:
    """Main RAG application that orchestrates the entire pipeline."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2', 
                 llm_model: str = "llama3",
                 ollama_host: str = "http://localhost:11434"):
        """Initialize the RAG application."""
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.retriever = DocumentRetriever()
        self.answer_generator = AnswerGenerator(model=llm_model, host=ollama_host)
        
        # These will be populated when loading documents
        self.chunks = []
        self.chunk_sources = []
        self.chunk_embeddings = None
        
        print("RAG application initialized successfully!")
    
    def load_single_document(self, document_source: str):
        """Load a single document into the RAG application."""
        document_text = DocumentProcessor.load_document(document_source)
        new_chunks = DocumentProcessor.chunk_document(document_text)
        
        # Add source information for each chunk
        source_name = os.path.basename(document_source) if not DocumentProcessor._is_url(document_source) else document_source
        new_chunk_sources = [source_name] * len(new_chunks)
        
        # Generate embeddings for new chunks
        new_embeddings = self.embedding_generator.generate_embeddings(new_chunks)
        
        # Append to existing data
        self.chunks.extend(new_chunks)
        self.chunk_sources.extend(new_chunk_sources)
        
        # Update chunk embeddings
        if self.chunk_embeddings is None:
            self.chunk_embeddings = new_embeddings
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
        
        print(f"Loaded document: {source_name} ({len(new_chunks)} chunks)")
    
    def load_project_folder(self, folder_path: str):
        """Load all documents from a project folder."""
        documents = DocumentProcessor.load_project_folder(folder_path)
        
        total_chunks = 0
        for rel_path, content in documents.items():
            new_chunks = DocumentProcessor.chunk_document(content)
            new_chunk_sources = [rel_path] * len(new_chunks)
            
            # Generate embeddings for new chunks
            new_embeddings = self.embedding_generator.generate_embeddings(new_chunks)
            
            # Append to existing data
            self.chunks.extend(new_chunks)
            self.chunk_sources.extend(new_chunk_sources)
            
            # Update chunk embeddings
            if self.chunk_embeddings is None:
                self.chunk_embeddings = new_embeddings
            else:
                self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
            
            total_chunks += len(new_chunks)
        
        print(f"Loaded {len(documents)} documents with {total_chunks} total chunks from {folder_path}")
    
    def answer_question(self, query: str, top_k: int = 3, include_sources: bool = True) -> str:
        """Process a user question and generate an answer."""
        if not self.chunks:
            return "No documents have been loaded. Please load documents first."
        
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        retrieved_chunks = self.retriever.retrieve_chunks(
            query_embedding, self.chunk_embeddings, self.chunks, self.chunk_sources, top_k
        )
        
        answer = self.answer_generator.generate_answer(query, retrieved_chunks, include_sources)
        
        return answer


def main():
    """Run the RAG application as a command-line interface."""
    print("=" * 50)
    print("RAG Application - Answer questions based on documents")
    print("=" * 50)
    print("Using Ollama for text generation")
    
    # Get Ollama settings
    ollama_host = input("Enter Ollama API host (default: http://localhost:11434): ")
    if not ollama_host:
        ollama_host = "http://localhost:11434"
    
    # Try to get available models
    try:
        response = requests.get(f"{ollama_host}/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json().get("models", [])]
            if available_models:
                print("Available models:", ", ".join(available_models))
                default_model = available_models[0]
            else:
                default_model = "llama3"
        else:
            default_model = "llama3"
    except:
        default_model = "llama3"
    
    llm_model = input(f"Enter Ollama model name (default: {default_model}): ")
    if not llm_model:
        llm_model = default_model
    
    # Initialize RAG application
    rag_app = RAGApplication(
        llm_model=llm_model,
        ollama_host=ollama_host
    )
    
    # Ask user for document input method
    input_type = input("Choose how to load documents:\n1. Single file\n2. Project folder\nEnter choice (1 or 2): ")
    
    if input_type == "1":
        # Single document mode
        doc_source = input("Enter the path to a document (PDF, DOCX, TXT) or a URL: ")
        
        if not DocumentProcessor._is_url(doc_source) and not os.path.exists(doc_source):
            print(f"Warning: File not found: {doc_source}")
            proceed = input("Do you want to proceed anyway? (y/n): ")
            if proceed.lower() != 'y':
                return
        
        try:
            rag_app.load_single_document(doc_source)
        except Exception as e:
            print(f"An error occurred while loading the document: {str(e)}")
            return
    
    elif input_type == "2":
        # Project folder mode
        folder_path = input("Enter the path to a project folder: ")
        
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            proceed = input("Do you want to proceed anyway? (y/n): ")
            if proceed.lower() != 'y':
                return
        
        try:
            rag_app.load_project_folder(folder_path)
        except Exception as e:
            print(f"An error occurred while loading the project folder: {str(e)}")
            return
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Check if any documents were loaded
    if not rag_app.chunks:
        print("No documents were successfully loaded. Exiting.")
        return
    
    # Get top-k setting
    try:
        top_k = int(input("Enter the number of chunks to retrieve per query (default: 3): ") or "3")
    except ValueError:
        top_k = 3
        print("Invalid input, using default value of 3.")
    
    # Get source inclusion preference
    include_sources = input("Include source references in answers? (y/n, default: y): ").lower() != 'n'
    
    print("\nRAG application is ready! Type 'exit' to quit.")
    
    # Main question loop
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        answer = rag_app.answer_question(query, top_k=top_k, include_sources=include_sources)
        print("\n" + textwrap.fill(answer, width=80) + "\n")


if __name__ == "__main__":
    main()