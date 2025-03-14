# RAG Application

This project is a Retrieval-Augmented Generation (RAG) application that answers questions based on documents. It uses a combination of document processing, embedding generation, and language model generation to provide accurate answers with references to the source documents.

## Features

- Load documents from various sources (file paths or URLs)
- Support for multiple document formats: PDF, DOCX, TXT, JS, HTML, TS, CSS
- Chunk documents into smaller parts for better processing
- Generate embeddings for document chunks using Sentence Transformers
- Retrieve relevant document chunks based on a query
- Generate answers using the Ollama API
- Include source references in the generated answers

## Requirements

- Python 3.7+
- Required Python packages (install using `pip`):
  - `numpy`
  - `requests`
  - `nltk`
  - `sentence-transformers`
  - `scikit-learn`
  - `PyPDF2`
  - `python-docx`
  - `beautifulsoup4`

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/hayder86al/rag-chat-bot.git
    cd rag-chat-bot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv myenv
    source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download necessary NLTK data:
    ```sh
    python -m nltk.downloader punkt
    ```

## Usage

1. Run the main script:
    ```sh
    python app.py
    ```

2. Follow the prompts to configure the Ollama API host and model.

3. Choose how to load documents:
    - Single file: Enter the path to a document (PDF, DOCX, TXT) or a URL.
    - Project folder: Enter the path to a project folder containing supported documents.

4. Enter your question when prompted. The application will retrieve relevant document chunks and generate an answer.

5. Type `exit` or `quit` to exit the application.

## Example

```sh
==================================================
RAG Application - Answer questions based on documents
==================================================
Using Ollama for text generation
Enter Ollama API host (default: http://localhost:11434): 
Available models: llama3
Enter Ollama model name (default: llama3): 
Choose how to load documents:
1. Single file
2. Project folder
Enter choice (1 or 2): 1
Enter the path to a document (PDF, DOCX, TXT) or a URL: example.pdf
Loaded document: example.pdf (5 chunks)
Enter the number of chunks to retrieve per query (default: 3): 3
Include source references in answers? (y/n, default: y): y

RAG application is ready! Type 'exit' to quit.

Enter your question: What is the main topic of the document?
```

## License

This project is licensed under the MIT License.