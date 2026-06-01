# Hospital Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about hospital policies by extracting knowledge from PDF documents. Built with Gemini, ChromaDB, and Sentence Transformers.

## How It Works

1. **PDF Ingestion** — Extracts text from `hospital_data.pdf` and splits it into overlapping chunks
2. **Embedding & Storage** — Chunks are embedded using `all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection
3. **Retrieval** — User queries are matched against stored chunks using cosine similarity
4. **Generation** — Top matching chunks are sent as context to Gemini, which generates a grounded answer

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/your-username/hospital-chatbot.git
cd hospital-chatbot
uv sync
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Place your hospital PDF as `hospital_data.pdf` in the project root.

### Run

```bash
uv run python main.py
```

On first run, the PDF is processed and indexed into ChromaDB (stored in `./hospital_db/`). Subsequent runs skip ingestion and go straight to the chat loop.

## Project Structure

```
├── main.py              # RAG pipeline and chat loop
├── hospital_data.pdf    # Source PDF document
├── hospital_db/         # ChromaDB persistent storage (auto-generated)
├── pyproject.toml       # Dependencies
└── .env                 # API key (not committed)
```

## Tech Stack

- **LLM** — Google Gemini 2.5 Flash
- **Vector DB** — ChromaDB (persistent, cosine similarity)
- **Embeddings** — Sentence Transformers (`all-MiniLM-L6-v2`)
- **PDF Parsing** — pypdf
