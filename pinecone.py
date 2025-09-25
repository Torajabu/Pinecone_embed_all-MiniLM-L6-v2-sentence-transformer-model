import os
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
import re
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better embedding"""
    # Clean up text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence ending
            last_period = text.rfind('.', start, end)
            last_question = text.rfind('?', start, end)
            last_exclamation = text.rfind('!', start, end)
            
            sentence_end = max(last_period, last_question, last_exclamation)
            
            if sentence_end > start:
                end = sentence_end + 1
            else:
                # Fall back to word boundary
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

# -------------------------------
# 1. Load local embedding model (CPU)
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # outputs 384-dim vectors

# -------------------------------
# 2. Init Pinecone client
# -------------------------------
PINECONE_API_KEY = ""

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "pdf-demo-index"
DIM = 384

# Create index if it doesn't exist
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME, 
        dimension=DIM, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print("Created new index, waiting for it to be ready...")
    time.sleep(10)  # Wait for index to be ready

index = pc.Index(INDEX_NAME)

# -------------------------------
# 3. Process PDF file
# -------------------------------
PDF_PATH = "/home/rajab/coding_setups/TO READ/one_page.pdf"  # Change this to your PDF path

print("Extracting text from PDF...")
pdf_text = extract_text_from_pdf(PDF_PATH)

if not pdf_text:
    print("No text extracted from PDF. Exiting.")
    exit()

print(f"Extracted {len(pdf_text)} characters from PDF")

# Split into chunks
print("Splitting text into chunks...")
text_chunks = chunk_text(pdf_text, chunk_size=500, overlap=50)
print(f"Created {len(text_chunks)} chunks")

# Preview first chunk
print(f"\nFirst chunk preview:\n{text_chunks[0][:200]}...\n")

# -------------------------------
# 4. Create embeddings for all chunks
# -------------------------------
print("Creating embeddings...")
embeddings = model.encode(text_chunks).tolist()
print(f"Created {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")

# -------------------------------
# 5. Upsert into Pinecone
# -------------------------------
print("Uploading to Pinecone...")
batch_size = 100  # Process in batches to avoid memory issues

for i in range(0, len(text_chunks), batch_size):
    batch_chunks = text_chunks[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size]
    
    # Create vectors with metadata
    vectors = []
    for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
        vectors.append({
            "id": f"chunk_{i+j}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i+j,
                "source": "pdf_document"
            }
        })
    
    # Upsert batch
    index.upsert(vectors=vectors)
    print(f"Uploaded batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")

print("Data upload complete!")

# Wait for index to be ready for queries
print("Waiting for index to be ready for queries...")
time.sleep(5)

# Check index stats
stats = index.describe_index_stats()
print(f"Index stats: {stats}")

# -------------------------------
# 6. Query the PDF content
# -------------------------------
def search_pdf(query: str, top_k: int = 5):
    """Search the PDF content"""
    print(f"\nQuery: '{query}'")
    
    try:
        # Create query embedding
        q_emb = model.encode([query]).tolist()[0]
        print(f"Query embedding created with {len(q_emb)} dimensions")
        
        # Perform search
        result = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        
        print(f"Found {len(result['matches'])} results:")
        
        if not result["matches"]:
            print("No matches found!")
            return
            
        for i, match in enumerate(result["matches"], 1):
            print(f"\n{i}. Score: {match['score']:.4f}")
            print(f"   ID: {match['id']}")
            text = match['metadata']['text']
            if len(text) > 200:
                print(f"   Text: {text[:200]}...")
            else:
                print(f"   Text: {text}")
                
    except Exception as e:
        print(f"Error during search: {e}")

# Example queries - modify these based on your PDF content
# More specific queries based on your sample text
search_pdf("simple PDF file")
search_pdf("demonstration sentences")
search_pdf("document clean")
search_pdf("text format")

# Also try broader queries
search_pdf("What is this about?")
search_pdf("PDF")
