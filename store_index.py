
from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings
)

from langchain_community.vectorstores import FAISS

print("Loading PDF files...")

extracted_data = load_pdf_file(data="Data/")

print(f"Loaded {len(extracted_data)} pages")

text_chunks = text_split(extracted_data)

print(f"Created {len(text_chunks)} chunks")

embeddings = download_hugging_face_embeddings()

print("Creating FAISS index...")

vector_store = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)

vector_store.save_local("faiss_index")

print("FAISS index created successfully")