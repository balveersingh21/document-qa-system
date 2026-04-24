import streamlit as st
import PyPDF2
import re
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Document Q&A AI", layout="wide")
st.title("📄 Document Q&A AI (Gemini Powered)")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gemini_model = genai.GenerativeModel("gemini-pro")
    return embed_model, gemini_model

embed_model, gemini_model = load_models()

if "history" not in st.session_state:
    st.session_state.history = []

if "processed" not in st.session_state:
    st.session_state.processed = False

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@st.cache_data
def get_embeddings(chunks):
    return embed_model.encode(chunks)

def get_relevant_chunks(question, chunks, embeddings, top_k=5):
    q_emb = embed_model.encode([question])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file and not st.session_state.processed:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)
        text = clean_text(text)
        chunks = split_text(text)
        embeddings = get_embeddings(chunks)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.processed = True
    st.success(f"✅ Document processed into {len(chunks)} chunks!")

st.subheader("💬 Ask Questions")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

question = st.chat_input("Ask something from your document...")

if question and uploaded_file:
    with st.chat_message("user"):
        st.write(question)

    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings

    context_chunks = get_relevant_chunks(question, chunks, embeddings)
    context = " ".join(context_chunks)

    with st.spinner("Thinking..."):
        prompt = f"""
You are a helpful AI assistant.

Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer clearly:
"""

        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()

    if not answer:
        answer = "I couldn't find a clear answer in the document."

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.history.append((question, answer))

if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.processed = False
    st.rerun()
