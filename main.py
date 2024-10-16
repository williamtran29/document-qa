from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.schema import Document
import faiss
import os
import uuid
import io
import json
import numpy as np
import logging
import fitz
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Embedding and Query API",
    description="This API allows users to upload PDF files, embed their content into a FAISS index, and query the embedded content using natural language questions.",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware
# Allow CORS requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings and FAISS storage
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
faiss_indices = {}
texts_store = {}

DB_FAISS_PATH = 'vectorstore/db_faiss'
TEXTS_STORE_PATH = 'vectorstore/texts_store'
TMP_PATH = 'tmp'

if not os.path.exists(TMP_PATH):
    os.makedirs(TMP_PATH)
    
if not os.path.exists(DB_FAISS_PATH):
    os.makedirs(DB_FAISS_PATH)

if not os.path.exists(TEXTS_STORE_PATH):
    os.makedirs(TEXTS_STORE_PATH)

def load_knowledgeBase():
    db = faiss.read_index(DB_FAISS_PATH)
    return db

def get_openai_instance():
    """Return an OpenAI instance with the API key."""
    return OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def parse_pdf(file) -> str:
    """Extracts text from a PDF file."""
    doc = fitz.open(file)
    output = []
    for page in doc:
        text = page.get_text()
        if not text:
            try:
                text = page.get_textpage_ocr(language='eng').extractText()
            except RuntimeError as e:
                print(e)
        if text:
            output.append(text)
    return "\n\n".join(output)

def embed_text(text: str, pdf_id: str):
    """Split the text and embed it in a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors).astype('float32')
    
    # Assuming the dimension of the embeddings is known (e.g., 1536 for OpenAI's embeddings)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    index.add(vectors)
    
    # Save the index to disk
    faiss.write_index(index, os.path.join(DB_FAISS_PATH, f"{pdf_id}.index"))
    
    # Save the texts to disk
    with open(os.path.join(TEXTS_STORE_PATH, f"{pdf_id}.json"), 'w') as f:
        json.dump(texts, f)

    return index, texts

@app.post("/upload", summary="Upload PDF and Embed Content", description="Uploads a PDF file, extracts its content, embeds it, and stores it in a FAISS index.")
async def upload_pdf(file: UploadFile = File(...), pdf_id: str = Form(None)):
    if pdf_id is None:
        pdf_id = str(uuid.uuid4())
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        content = await file.read()
        temp_file_path = os.path.join(TMP_PATH, f"{uuid.uuid4()}.index")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        
        text = parse_pdf(temp_file_path)
        os.remove(temp_file_path)
        
        logger.info(f"Extracted text length: {len(text)}")
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from the PDF.")
        
        index, texts = embed_text(text, pdf_id)
        logger.info(f"Index and texts created for pdf_id: {pdf_id}")
        
        faiss_indices[pdf_id] = index
        texts_store[pdf_id] = texts
        return {"message": "PDF uploaded and embedded successfully.", "pdf_id": pdf_id}
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class QueryPayload(BaseModel):
    pdf_id: str
    question: str

@app.post("/query", summary="Query Embedded PDF Content", description="Queries the embedded content of a previously uploaded PDF using a natural language question.")
async def query_pdf(payload: QueryPayload):
    pdf_id = payload.pdf_id
    question = payload.question
    
    if not pdf_id or not question:
        raise HTTPException(status_code=400, detail="Both 'pdf_id' and 'question' are required in the request body.")
    
    index_path = os.path.join(DB_FAISS_PATH, f"{pdf_id}.index")
    texts_path = os.path.join(TEXTS_STORE_PATH, f"{pdf_id}.json")
    
    if not os.path.exists(index_path) or not os.path.exists(texts_path):
        raise HTTPException(status_code=400, detail="Invalid PDF ID or PDF has not been uploaded.")

    try:
        # Load the FAISS index from disk
        index = faiss.read_index(index_path)
        
        # Load the texts from disk
        with open(texts_path, 'r') as f:
            texts = json.load(f)
        
        # For simplicity, using the centroid of the query embedding for similarity search
        query_vector = embeddings.embed_query(question)
        D, I = index.search(np.array([query_vector]).astype('float32'), k=5)
        
        # Retrieve the original chunks based on the indices `I`
        docs = [Document(page_content=texts[i]) for i in I[0] if i != -1]
        
        chain = load_qa_chain(get_openai_instance(), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        if not answer:
            return {"answer": "No answer could be generated from the provided content."}
        return {"answer": answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
