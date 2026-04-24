# Document Q&A System (FAISS Powered)

## Project Overview

This project is a Document Question Answering System developed using Streamlit and FAISS. It allows users to upload a TXT or PDF file and ask questions based on the content of the document.

The system retrieves relevant information using semantic similarity and provides concise answers extracted from the document.

---

## Features

* Upload TXT or PDF documents which are paragraph based
* Automatic text extraction and preprocessing
* Document chunking for efficient processing
* Fast similarity search using FAISS
* Semantic understanding using Sentence Transformers
* Chat-based interface for user interaction
* Extraction of the most relevant answer from the document

---

## Technologies Used

* Python
* Streamlit
* FAISS (Facebook AI Similarity Search)
* Sentence Transformers (all-MiniLM-L6-v2)
* NumPy
* PyPDF2
* Scikit-learn

---

## Project Structure

```
document-qa-system/
│
├── app.py
├── requirements.txt
├── README.md
```

---

## Working Methodology

1. The user uploads a TXT or PDF file.
2. The system extracts and cleans the text.
3. The text is divided into smaller chunks.
4. Each chunk is converted into vector embeddings.
5. A FAISS index is created for efficient similarity search.
6. The user submits a query.
7. The system retrieves the most relevant chunk.
8. The most relevant sentence is extracted and displayed as the answer.

---

## How to Run the Project

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
streamlit run app.py
```

---

## Deployment

This application has been deployed using streamlit cloud platform

---

## Example Queries

* What is Artificial Intelligence?
* What is Machine Learning?

---

## Requirements

The project dependencies are:-
streamlit
sentence-transformers
faiss-cpu
numpy
PyPDF2
torch
torchvision
scikit-learn
---

## Learning Outcomes

* Understanding of Retrieval-Augmented Generation (RAG) concepts
* Working with vector embeddings and similarity search
* Building an interactive AI-based application
* Deployment using Streamlit

---

## Future Enhancements

* Improve answer generation using advanced language models
* Highlight answers within the document
* Support multiple file uploads
* Add summarization capabilities

---
