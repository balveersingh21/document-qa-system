import streamlit as st
import PyPDF2
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Document Q&A (FAISS)", layout="wide")
st.title("Document Q&A AI")

# LOAD MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# SESSION STATE

if "history" not in st.session_state:
    st.session_state.history = []

if "processed" not in st.session_state:
    st.session_state.processed = False


# EXTRACT TEXT

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text


# CLEAN TEXT

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


# SPLIT TEXT

def split_text(text, chunk_size=150):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# BUILD FAISS INDEX

@st.cache_data
def build_faiss(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


# SEARCH

def search(question, index, chunks, k=3):
    q_emb = model.encode([question])
    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, k)

    results = [chunks[i] for i in indices[0]]
    return results
    
def extract_best_sentence(question, chunk):
    sentences = [s.strip() for s in chunk.split(".") if s.strip()]

    q_emb = model.encode([question])
    s_emb = model.encode(sentences)

    sims = cosine_similarity(q_emb, s_emb)[0]
    best_idx = sims.argmax()

    return sentences[best_idx]

# FILE UPLOAD

uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])

if uploaded_file and not st.session_state.processed:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)
        text = clean_text(text)

        chunks = split_text(text)
        index, embeddings = build_faiss(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.processed = True

    st.success(f"Processed {len(chunks)} chunks")


# CHAT UI

st.subheader("Ask Questions")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

question = st.chat_input("Ask your question...")
# ANSWER

if question and uploaded_file:
    with st.chat_message("user"):
        st.write(question)

    chunks = st.session_state.chunks
    index = st.session_state.index

    results = search(question, index, chunks, k=1)

    if not results:
        answer = "❌ Answer not found in document."
    else:
        answer = extract_best_sentence(question, results[0])

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.history.append((question, answer))


# CLEAR CHAT
if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.processed = False
    st.rerun()
