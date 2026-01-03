from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

# - Load raw PDF(s)
DATA_PATH = r"C:\Projects\AI_medical_chatbot\data"
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents

documents = load_pdf(DATA_PATH)
#print("length of documents:", len(documents))
# Step 2 - Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks = create_chunks(documents)
#print("length of text_chunks:", len(text_chunks))
    
# Step 3 - Create Vector Embeddings
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = get_embeddings()

# Step 4 - Store embeddings in FAISS
DB_FAISS_PATH = r"C:\Projects\AI_medical_chatbot\vectorstore\db_faiss"
os.makedirs(DB_FAISS_PATH, exist_ok=True)
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
