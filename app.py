import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

import re
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------- ENV CONFIG -------------------------------
load_dotenv()

HISTORY_FILE = "history.json"

# --------------------------- HISTORY HELPERS --------------------------

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

history = load_history()

# --------------------------- UI CONFIG -------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide",
    page_icon="📄"
)

# Clean minimal UI
st.markdown("""
<style>
    .section-tag {
        background-color:#eef0f3;
        padding:6px 12px;
        border-radius:12px;
        margin:3px;
        display:inline-block;
        font-size:14px;
    }
</style>
""", unsafe_allow_html=True)


st.title("📄 AI Resume Analyzer")
st.write("Upload → Extract → Vector Search → LLM Analysis")

# --------------------------- SIDEBAR HISTORY --------------------------
st.sidebar.header("🕒 History")

if len(history) == 0:
    st.sidebar.caption("No history yet.")
else:
    for idx, item in enumerate(history):
        st.sidebar.write(f"**{idx+1}. Q:** {item['q']}")
        st.sidebar.caption(f"Response length: {len(item['a'])} chars")

# -------------------- Utility Functions ------------------------------

def extract_text_from_pdf(tmp_path):
    reader = PdfReader(tmp_path)
    s = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            s += "\n" + t
    return s.strip()

def split_into_sections(text):
    keywords = [
        "education","projects","skills","experience","work experience",
        "certifications","achievements","internship","courses","languages",
        "summary","profile","objective","extracurricular","publications","interests"
    ]
    clean = text.lower()
    positions = {}

    for k in keywords:
        m = re.search(r"\b" + re.escape(k) + r"\b", clean)
        if m:
            positions[k] = m.start()

    if not positions:
        return {"full_resume": text}

    sorted_pos = sorted(positions.items(), key=lambda x: x[1])
    sections = {}

    for i in range(len(sorted_pos)):
        name, start = sorted_pos[i]
        end = sorted_pos[i+1][1] if i+1 < len(sorted_pos) else len(text)
        sections[name] = text[start:end].strip()

    return sections

def init_vectordb(persist_dir="my_chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="resume_sections",
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectordb

def store_sections(vectordb, sections):
    docs, metas, ids = [], [], []
    for i, (name, content) in enumerate(sections.items()):
        docs.append(content)
        metas.append({"section": name})
        ids.append(f"sec_{i}")
    vectordb.add_texts(texts=docs, metadatas=metas, ids=ids)
    vectordb.persist()

def retrieve(vectordb, query, k=3):
    return vectordb.similarity_search(query, k=k)

def build_chain():
    llm = ChatGroq(model="llama-3.1-8b-instant")

    prompt = PromptTemplate(
        template=(
            "I will provide you a question and a resume document.\n"
            "Analyze the resume and provide:\n"
            "- Accurate ATS score\n"
            "- Detailed reasoning\n"
            "- Answer to the question\n\n"
            "Question: {question}\n\n"
            "Document: {document}"
        ),
        input_variables=["question", "document"],
    )
    return prompt | llm | StrOutputParser()

# -------------------------- MAIN UI ---------------------------------

uploaded = st.file_uploader("📤 Upload your Resume (PDF)", type=["pdf"])
question = st.text_input("💬 Enter your Question")

if st.button("🔍 Analyze Resume", use_container_width=True):

    if not uploaded:
        st.error("Please upload a PDF first.")
        st.stop()

    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    # Save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    # Extract text
    text = extract_text_from_pdf(tmp_path)

    st.subheader("📄 Resume Preview")
    st.code(text[:2000] + ("..." if len(text) > 2000 else ""))

    # Split sections
    sections = split_into_sections(text)
    st.subheader("🗂 Detected Resume Sections")
    st.markdown("".join([f"<span class='section-tag'>{s}</span>" 
                         for s in sections.keys()]), 
                unsafe_allow_html=True)

    # Vector DB
    vectordb = init_vectordb("my_chroma_db")
    store_sections(vectordb, sections)

    retrieved = retrieve(vectordb, question, k=3)

    st.subheader("📌 Retrieved Relevant Sections")
    for r in retrieved:
        st.markdown(f"### {r.metadata['section'].upper()}")
        st.write(r.page_content[:800] + ("..." if len(r.page_content) > 800 else ""))

    # AI Analysis
    with st.spinner("🤖 Running AI Analysis..."):
        chain = build_chain()
        doc_context = "\n\n".join(
            [f"SECTION ({r.metadata['section']}):\n{r.page_content}" for r in retrieved]
        )

        try:
            output = chain.invoke({"question": question, "document": doc_context})

            st.subheader("🧠 AI Output")
            st.write(output)

            history.append({"q": question, "a": output})
            save_history(history)

        except Exception as e:
            st.error("LLM Error: " + str(e))
