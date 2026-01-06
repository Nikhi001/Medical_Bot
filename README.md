# Medical Chatbot (Med-Bot)

## Description

This repository contains a **Retrieval-Augmented Medical Chatbot** that answers user queries using **medical PDF documents** as its knowledge source.

Instead of generating answers without context, the chatbot retrieves relevant information from documents and uses it to generate accurate, context-aware responses using a Large Language Model (LLM).

---
## Overview

Med-Bot enables users to ask medical questions and receive responses based strictly on uploaded medical PDFs.
Instead of generating answers blindly, the chatbot retrieves relevant content from documents and uses it as context for answer generation.

---

### What this project contains

- **main.py** – Starts the medical chatbot (app/chat loop).
- **connect_memory_with_LLM.py** – Connects vector memory with the LLM for retrieval.
- **create_memory_for_llm.py** – Creates embeddings and builds the FAISS vector store.
- **requirement.txt** – Project dependencies
- **.gitignore** – Ignores venv, cache, and unnecessary files in Git.
- **data/** – Stores source medical PDFs.
    - *The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf* – Main knowledge document.
- **vectorstore/** – Stores vector database.
    - **db_faiss/**
        - *index.faiss* – FAISS embedding index.
        - *index.pkl* – Metadata and document–chunk mapping.

---

## Technology Stack

| Component            | Tool        |
| -------------------- | ----------- |
| Programming Language | Python      |
| LLM Framework        | LangChain   |
| Model Provider       | HuggingFace |
| Language Model       | Mistral     |
| Vector Database      | FAISS       |
| User Interface       | Streamlit   |
| IDE                  | VS Code     |

---

## Project Structure

```
medical-chatbot/
│
├── data/
│   └── medical_docs/        # Medical PDF input files
│
├── vector_store/
│   └── faiss_index/         # Stored FAISS embeddings
│
├── ingest.py                # PDF loading and embedding creation
├── chain.py                 # RAG pipeline (Retriever + LLM)
├── app.py                   # Streamlit chatbot application
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md
```

---

## Installation and Setup

### Step 1: Python Version

Verify Python installation:

```
python --version
```

Required version: **Python 3.9 or higher**

---

### Step 2: Install Dependencies

Install all required libraries:

```
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Create a file named `.env` in the project root directory.

Add the following line:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

This token is required to access HuggingFace models and embeddings.

---

### Step 4: Run the Application

Start the chatbot interface:

```
streamlit run app.py
```

---

### Step 5: Access the Application

Open a browser and navigate to:

```
http://localhost:8501
```

---

## Usage Instructions

1. Place medical PDF files inside:

```
data/medical_docs/
```

2. Run the ingestion process to create embeddings
3. Launch the Streamlit application
4. Enter medical questions in the chatbot interface
5. The chatbot responds using only the document content

---

## System Workflow

### Phase 1: Vector Database Creation

* Load medical PDF documents
* Split text into chunks
* Generate vector embeddings
* Store embeddings in FAISS

---

### Phase 2: LLM and Memory Integration

* Initialize the Mistral LLM using HuggingFace
* Connect FAISS as the retriever
* Build the LangChain RAG pipeline

---

### Phase 3: Chatbot User Interface

* Load the FAISS index from disk
* Accept user queries
* Retrieve relevant document chunks
* Generate responses using the LLM

---

## Future Enhancements

* User authentication
* Document upload via UI
* Support for multiple documents
* Unit testing for RAG pipelines
* Metadata-based document filtering
* Cloud deployment (AWS, Azure, GCP)

---

## Disclaimer

This chatbot does **not** provide medical advice.
It should not be used for diagnosis or treatment decisions.

Always consult qualified healthcare professionals.

---

## Contributing

Contributions are welcome.
Fork the repository and submit a pull request with proper documentation.

---

## License

This project is licensed under the **MIT License**.

---

## Author

Nikhil
AI / ML Developer

---

### Next logical beginner-friendly additions (optional)

If you want, next we can add:

* “What happens when I run ingest.py”
* “How FAISS works in simple terms”
* “Common beginner errors and fixes”
* “One-command setup for first-time users” 
