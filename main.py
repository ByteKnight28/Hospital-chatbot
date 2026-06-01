import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

PDF_FILE_NAME = "hospital_data.pdf"
CHROMA_DB_PATH = "./hospital_db"
COLLECTION_NAME = "hospital_pdf_knowledge_base"
MODEL_NAME = "gemini-2.5-flash"
CHROMA_BATCH_LIMIT = 5000

SYSTEM_INSTRUCTION = """
You are a highly accurate hospital assistant.
Answer the user's question using ONLY the provided context extracted from the hospital's PDF documents.
If the information is not present in the context, state "I do not have that information in my database."
Do not guess or use outside knowledge.
"""


def process_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    if not full_text.strip():
        raise ValueError(f"No text could be extracted from '{file_path}'.")

    print(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages.")

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return [c for c in chunks if c]


def init_collection():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in your .env file.")

    gemini_client = genai.Client(api_key=api_key)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    return gemini_client, collection


def ingest_pdf(collection):
    if collection.count() > 0:
        print(f"Database already has {collection.count()} chunks. Skipping ingestion.\n")
        return

    if not os.path.exists(PDF_FILE_NAME):
        raise FileNotFoundError(f"Place '{PDF_FILE_NAME}' in the project directory.")

    print("Processing PDF and indexing...\n")
    chunks = process_pdf(PDF_FILE_NAME)

    ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": PDF_FILE_NAME, "chunk_index": i} for i in range(len(chunks))]

    for batch_start in range(0, len(chunks), CHROMA_BATCH_LIMIT):
        batch_end = batch_start + CHROMA_BATCH_LIMIT
        collection.add(
            documents=chunks[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
            ids=ids[batch_start:batch_end],
        )

    print(f"Indexed {len(chunks)} chunks.\n")


def retrieve_context(collection, query: str, n_results: int = 3) -> str | None:
    results = collection.query(query_texts=[query], n_results=n_results)

    if not results["documents"] or not results["documents"][0]:
        return None

    chunks = results["documents"][0]
    return "\n\n".join(f"--- CHUNK {i+1} ---\n{chunk}" for i, chunk in enumerate(chunks))


def ask(gemini_client, collection, query: str) -> str:
    context = retrieve_context(collection, query)

    if not context:
        return "No relevant context found in the database for your question."

    prompt = f"""
Context from PDF Documents:
{context}

User Question: {query}
"""

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        ),
    )
    return response.text


def main():
    gemini_client, collection = init_collection()
    ingest_pdf(collection)

    print("Hospital Chatbot Ready! Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nSearching & generating response...\n")
        try:
            answer = ask(gemini_client, collection, query)
            print(f"Assistant: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()