# app.py — HYPE RAG v1.0 (2025 Edition)
# The most visually stunning, blazing-fast, production-grade RAG chatbot you've ever seen.

import os
import time
import io
from pathlib import Path
from typing import List, Dict, Optional

import torch
import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# =========================
# 🎨 PAGE CONFIG + THEMING
# =========================
st.set_page_config(
    page_title="🔥 HYPE RAG — AI Knowledge Chatbot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for hype styling
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    .stButton>button { background: linear-gradient(45deg, #f093fb, #f5576c); color: white; border-radius: 12px; font-weight: bold; }
    .stTextInput>div>div>input { background: rgba(255,255,255,0.1); color: white; border: 1px solid #f093fb; }
    .stMarkdown { color: #f0f0f0; }
    .sidebar .sidebar-content { background: #1a1a2e; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 HYPE RAG — Chat with Your Docs")
st.caption("Powered by FAISS + Transformers + LangChain + 💥 Pure Hype")

# =========================
# 🔐 SECRETS / TOKENS
# =========================
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN:
    st.warning("⚠️ No Hugging Face token found. Set `HF_TOKEN` in secrets or env.")
    st.info("Public models (e.g., TinyLlama) will still work.")

# =========================
# ⚙️ SIDEBAR SETTINGS
# =========================
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
    st.header("🎛️ HYPE Controls")

    embedding_model_name = st.selectbox(
        "🧠 Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "thenlper/gte-small"
        ],
        index=0,
        help="What powers your semantic search"
    )

    llm_model_name = st.text_input(
        "🤖 LLM Model (Hugging Face)",
        value="google/gemma-2-9b-it",
        help="Try: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'microsoft/Phi-3-mini-4k-instruct', 'google/gemma-2-9b-it'"
    )

    inference_mode = st.radio(
        "⚡ Inference Mode",
        options=["local-4bit", "local-full", "huggingface-api"],
        index=0,
        help="Local 4-bit = fast & light. Full = high quality. API = no GPU needed."
    )

    temperature = st.slider("🌡️ Temperature", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("🎯 Top-p", 0.5, 1.0, 0.92, 0.01)
    max_new_tokens = st.slider("📏 Max New Tokens", 64, 1024, 384, 32)
    k_retrieval = st.slider("🔍 Top-K Chunks", 1, 8, 3, 1)

    show_chunks = st.toggle("💎 Show Retrieved Chunks", value=False)

    st.divider()
    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    persist_dir = st.text_input("💾 FAISS Save Folder", value="faiss_hype_index")
    rebuild_index = st.toggle("🔄 Rebuild Index on Upload", value=True)

# =========================
# 🧰 UTILITIES
# =========================
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                texts.append(text.strip())
        except Exception as e:
            st.warning(f"⚠️ Could not extract text from page: {e}")
            continue
    return "\n\n".join(texts)

def _read_txt(file_bytes: bytes) -> str:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = file_bytes.decode(enc)
            if len(text.strip()) > 10:
                return text.strip()
        except Exception:
            continue
    return ""

def load_documents(files) -> List[Dict]:
    docs = []
    for f in files or []:
        with st.spinner(f"📄 Reading {f.name}..."):
            content = f.read()
            if f.name.lower().endswith(".pdf"):
                text = _read_pdf(content)
            else:
                text = _read_txt(content)
            if text:
                docs.append({"source": f.name, "text": text})
                st.success(f"✅ Loaded {f.name} ({len(text)} chars)")
    return docs

def chunk_documents(raw_docs: List[Dict], chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = []
    for doc in raw_docs:
        texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(texts):
            if len(chunk.strip()) > 50:
                chunks.append({
                    "source": doc["source"],
                    "chunk_id": f"{doc['source']}#{i}",
                    "text": chunk.strip()
                })
    return chunks

@st.cache_resource(show_spinner=False)
def build_or_load_faiss(chunks: List[Dict], embeddings, persist_path: Optional[str]):
    if not chunks:
        return None

    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    if persist_path:
        path = Path(persist_path)
        if not path.exists() or rebuild_index:
            with st.spinner("🧠 Building FAISS index..."):
                vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
                path.mkdir(parents=True, exist_ok=True)
                vs.save_local(str(path))
                st.success(f"✅ Saved FAISS index to `{persist_path}`")
        else:
            with st.spinner(f"📂 Loading FAISS index from `{persist_path}`..."):
                vs = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vs

# =========================
# 🤖 LLM LOADING (HYPE MODE)
# =========================
@st.cache_resource(show_spinner="🚀 Loading HYPE LLM (this may take a minute)...")
def load_local_model(model_name: str, token: str | None, mode: str = "4bit"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set chat template if missing
        if not tokenizer.chat_template:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
            )

        # Configure quantization
        if mode == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:  # full precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

        return tokenizer, model, model.device

    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.exception(e)
        st.stop()

# =========================
# 🧠 EMBEDDINGS + VECTOR STORE
# =========================
embeddings = get_embeddings(embedding_model_name)
raw_docs = load_documents(uploaded_files)

if raw_docs:
    chunks = chunk_documents(raw_docs)
    vectorstore = build_or_load_faiss(chunks, embeddings, persist_dir or None)
    st.success(f"📚 Indexed {len(chunks)} chunks from {len(raw_docs)} files!")
elif persist_dir and Path(persist_dir).exists():
    with st.spinner("📂 Loading existing FAISS index..."):
        dummy_chunks = [{"text": "dummy", "source": "none", "chunk_id": "0"}]
        vectorstore = build_or_load_faiss(dummy_chunks, embeddings, persist_dir)
        st.success(f"📁 Loaded FAISS index from `{persist_dir}`.")
else:
    vectorstore = None
    st.info("📤 Upload PDFs or text files to build your knowledge base!")

# =========================
# 🚀 SETUP LLM BACKEND
# =========================
if "local" in inference_mode:
    mode = "4bit" if inference_mode == "local-4bit" else "full"
    tokenizer, model, device = load_local_model(llm_model_name, HF_TOKEN, mode)
    inference_client = None
    st.success(f"🟢 HYPE LLM loaded in {mode} mode on {str(device).upper()}")
else:
    inference_client = InferenceClient(model=llm_model_name, token=HF_TOKEN)
    tokenizer, model, device = None, None, "huggingface"
    st.info("☁️ Using Hugging Face Inference API — no local GPU needed!")

# =========================
# 💬 PROMPT ENGINEERING (HYPE PROMPT v2)
# =========================
def format_prompt(question: str, context_chunks: List[str], history: List[Dict] = None) -> str:
    context = "\n".join([f"📄 {c}" for c in context_chunks]) if context_chunks else "No relevant context found."

    system_prompt = """You are HYPEBOT 🤖 — a brilliant, enthusiastic AI assistant with deep knowledge from provided documents.
Answer questions accurately using the context below. If unsure, say "I don't know based on the provided context."
Be concise, engaging, and add a touch of personality!"""

    if inference_mode.startswith("local"):
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (optional enhancement)
        if history:
            for turn in history[-3:]:  # last 3 turns for context
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})
        
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        })
        
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        history_text = ""
        if history:
            for turn in history[-2:]:
                history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
        
        return f"""{system_prompt}

{history_text}
CONTEXT:
{context}

QUESTION: {question}

ANSWER: """

# =========================
# 🧠 GENERATE ANSWER (HYPE MODE)
# =========================
def generate_answer(prompt: str, temp: float, top_p: float, max_tokens: int) -> str:
    try:
        if inference_mode.startswith("local"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    num_return_sequences=1
                )
            
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            
            # Extract assistant response
            if "<|im_start|>assistant" in full_text:
                answer = full_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            else:
                answer = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            return answer.strip().replace("", "").replace("<|im_end|>", "").strip()

        else:  # Hugging Face API
            response = inference_client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop_sequences=["<|im_end|>", "User:", "Human:"]
            )
            return response.strip()

    except Exception as e:
        return f"💥 HYPE RAG Error: {str(e)}"

# =========================
# 🔍 RETRIEVAL
# =========================
def retrieve_chunks(query: str, k: int) -> List[Dict]:
    if not vectorstore:
        return []
    docs = vectorstore.similarity_search(query, k=k)
    return [
        {
            "text": d.page_content.strip(),
            "source": d.metadata.get("source", "unknown"),
            "chunk_id": d.metadata.get("chunk_id", "")
        }
        for d in docs if len(d.page_content.strip()) > 20
    ]

# =========================
# 💬 CHAT UI
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("💬 Ask HYPEBOT anything about your documents:", placeholder="E.g., Summarize the key points from my PDF...", label_visibility="collapsed")
with col2:
    ask_button = st.button("🚀", use_container_width=True)

if ask_button and user_input.strip():
    with st.spinner("🧠 HYPEBOT is thinking..."):
        # Retrieve
        retrieved = retrieve_chunks(user_input, k_retrieval)
        context_snippets = [r["text"] for r in retrieved]

        # Format prompt
        full_prompt = format_prompt(user_input, context_snippets, st.session_state.history)

        # Generate
        start_time = time.time()
        answer = generate_answer(full_prompt, temperature, top_p, max_new_tokens)
        latency = time.time() - start_time

        # Save to history
        st.session_state.history.append({
            "question": user_input,
            "answer": answer,
            "retrieved": retrieved,
            "latency": latency
        })

# Display chat history
for i, turn in enumerate(reversed(st.session_state.history)):
    with st.container(border=True):
        st.markdown(f"**🧑 YOU:** {turn['question']}")
        st.markdown(f"<div style='background:#2d2d3d; padding:15px; border-radius:10px; margin:10px 0;'><b>🤖 HYPEBOT:</b> {turn['answer']}</div>", unsafe_allow_html=True)
        st.caption(f"⏱️ {turn['latency']:.1f}s · 📚 {len(turn['retrieved'])} chunks · 🧠 {inference_mode.upper()}")

        if show_chunks and turn["retrieved"]:
            with st.expander("💎 View Retrieved Knowledge Fragments", expanded=False):
                for j, r in enumerate(turn["retrieved"], 1):
                    st.markdown(f"**Fragment {j}** — `{r['source']}`")
                    st.code(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""), language="text")

# =========================
# 📜 FOOTER
# =========================
st.divider()
st.caption("🔥 HYPE RAG v1.0 — The Ultimate Document Chatbot | Built with ❤️ using Streamlit + Transformers + FAISS")
st.caption(f"Model: `{llm_model_name}` | Embedding: `{embedding_model_name}` | Mode: `{inference_mode}`")

if st.button("🧹 Clear Chat History"):
    st.session_state.history = []
    st.rerun()
