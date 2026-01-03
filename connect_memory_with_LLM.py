import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Step 1 - Setup LLM 
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
huggingface_repo_id = "meta-llama/Llama-3.1-8B-Instruct"

def load_llm(repo_id):
    llm_base = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=TOKEN 
    )
    chat_llm = ChatHuggingFace(llm=llm_base)
    return chat_llm

# Step 2 - Define Modern System Prompt
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise and do not use small talk.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Step 3 - Load Database
DB_FAISS_PATH = r"C:\Projects\AI_medical_chatbot\vectorstore\db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS with safe deserialization
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Step 4 - Create the RAG Chain (The Modern Way)
llm = load_llm(huggingface_repo_id)

# This creates the part of the chain that handles document combination
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# This combines the retriever and the doc chain into a final RAG chain
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Step 5 - Invoke
user_query = input("Write query here: ")
# Modern chains use 'input' instead of 'query'
response = rag_chain.invoke({"input": user_query})

print("\n--- RESULT ---")
print(response["answer"])

print("\n--- SOURCE DOCUMENTS ---")
for doc in response["context"]:
    print(f"- {doc.metadata.get('source', 'Unknown source')}")