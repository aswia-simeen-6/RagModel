
# Multimodal RAG Agent

A full-stack **Retrieval-Augmented Generation (RAG)** application that allows you to chat with your documents, websites, and images using Google Gemini AI.

## 🏗️ Architecture

| Layer | Technology |
|---|---|
| **Frontend** | Angular 21 |
| **Backend** | FastAPI (Python) |
| **Vector Store** | ChromaDB |
| **LLM & Embeddings** | Google Gemini 2.5 Flash + gemini-embedding-001 |
| **Evaluation** | Ragas |

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload PDF files and query their contents
- 🌐 **Web Scraping** — Index any public webpage by providing a URL
- 🖼️ **Image Analysis** — Ask questions about uploaded images (multimodal vision)
- 📊 **Auto Chart Generation** — Automatically extracts numerical data and renders pie charts
- 🧠 **Persistent Context** — Multiple documents/URLs are combined into a single vector store session
- 🧹 **Memory Clear** — Reset the vector store between sessions
- ✅ **RAG Evaluation** — Built-in evaluation pipeline using Ragas metrics

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Gemini API Key

### Backend Setup

```bash
# Navigate to the backend folder
cd backend

# Install dependencies
pip install fastapi uvicorn langchain langchain-community langchain-google-genai \
            langchain-chroma chromadb pypdf ragas datasets python-dotenv

# Create a .env file and add your API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Start the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend Setup

```bash
# Navigate to the UI folder
cd rag-ui

# Install dependencies
npm install

# Start the dev server
ng serve
```

The app will be available at `http://localhost:4200`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and index a PDF file |
| `POST` | `/url` | Scrape and index a webpage |
| `POST` | `/ask` | Ask a question against the indexed knowledge base |
| `POST` | `/analyze-image` | Ask a question about an uploaded image |
| `POST` | `/clear` | Clear the in-memory vector store |

---

## 📊 RAG Evaluation

Run the evaluation pipeline against your test dataset:

```bash
# Ensure eval_data.json exists in the backend folder
python eval.py
```

### Metrics Used

| Metric | Description |
|---|---|
| **Context Precision** | Did ChromaDB rank the best chunks at the top? |
| **Context Recall** | Did ChromaDB retrieve all necessary information? |
| **Faithfulness** | Did the LLM hallucinate? |
| **Answer Relevancy** | Did the LLM actually answer the question asked? |

### `eval_data.json` Format

```json
{
  "question": ["What is X?", "Who is Y?"],
  "answer": ["X is ...", "Y is ..."],
  "contexts": [["chunk1", "chunk2"], ["chunk3"]],
  "ground_truth": ["Expected answer 1", "Expected answer 2"]
}
```

---

## 🔒 Hallucination Prevention

The agent is prompted with strict instructions to:
1. **Never** use outside knowledge — only the provided context
2. Return `"I do not have enough information in the provided documents to answer that."` when the answer cannot be found
3. Answer directly without conversational filler

---

## 📁 Project Structure

```
backend/
├── main.py          # FastAPI backend
├── eval.py          # Ragas evaluation pipeline
├── eval_data.json   # Test dataset for evaluation
├── .env             # API keys (not committed)
└── rag-ui/          # Angular frontend
    └── src/
        └── app/
            ├── rag.service.ts        # API service
            └── rag-component/        # Main chat component
=======
# Multimodal RAG Agent

A full-stack **Retrieval-Augmented Generation (RAG)** application that allows you to chat with your documents, websites, and images using Google Gemini AI.

## 🏗️ Architecture

| Layer | Technology |
|---|---|
| **Frontend** | Angular 21 |
| **Backend** | FastAPI (Python) |
| **Vector Store** | ChromaDB |
| **LLM & Embeddings** | Google Gemini 2.5 Flash + gemini-embedding-001 |
| **Evaluation** | Ragas |

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload PDF files and query their contents
- 🌐 **Web Scraping** — Index any public webpage by providing a URL
- 🖼️ **Image Analysis** — Ask questions about uploaded images (multimodal vision)
- 📊 **Auto Chart Generation** — Automatically extracts numerical data and renders pie charts
- 🧠 **Persistent Context** — Multiple documents/URLs are combined into a single vector store session
- 🧹 **Memory Clear** — Reset the vector store between sessions
- ✅ **RAG Evaluation** — Built-in evaluation pipeline using Ragas metrics

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Gemini API Key

### Backend Setup

```bash
# Navigate to the backend folder
cd backend

# Install dependencies
pip install fastapi uvicorn langchain langchain-community langchain-google-genai \
            langchain-chroma chromadb pypdf ragas datasets python-dotenv

# Create a .env file and add your API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Start the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend Setup

```bash
# Navigate to the UI folder
cd rag-ui

# Install dependencies
npm install

# Start the dev server
ng serve
```

The app will be available at `http://localhost:4200`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and index a PDF file |
| `POST` | `/url` | Scrape and index a webpage |
| `POST` | `/ask` | Ask a question against the indexed knowledge base |
| `POST` | `/analyze-image` | Ask a question about an uploaded image |
| `POST` | `/clear` | Clear the in-memory vector store |

---

## 📊 RAG Evaluation

Run the evaluation pipeline against your test dataset:

```bash
# Ensure eval_data.json exists in the backend folder
python eval.py
```

### Metrics Used

| Metric | Description |
|---|---|
| **Context Precision** | Did ChromaDB rank the best chunks at the top? |
| **Context Recall** | Did ChromaDB retrieve all necessary information? |
| **Faithfulness** | Did the LLM hallucinate? |
| **Answer Relevancy** | Did the LLM actually answer the question asked? |

### `eval_data.json` Format

```json
{
  "question": ["What is X?", "Who is Y?"],
  "answer": ["X is ...", "Y is ..."],
  "contexts": [["chunk1", "chunk2"], ["chunk3"]],
  "ground_truth": ["Expected answer 1", "Expected answer 2"]
}
```

---

## 🔒 Hallucination Prevention

The agent is prompted with strict instructions to:
1. **Never** use outside knowledge — only the provided context
2. Return `"I do not have enough information in the provided documents to answer that."` when the answer cannot be found
3. Answer directly without conversational filler

---

## 📁 Project Structure

```
backend/
├── main.py          # FastAPI backend
├── eval.py          # Ragas evaluation pipeline
├── eval_data.json   # Test dataset for evaluation
├── .env             # API keys (not committed)
└── rag-ui/          # Angular frontend
    └── src/
        └── app/
            ├── rag.service.ts        # API service
            └── rag-component/        # Main chat component
`