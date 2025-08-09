import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import os
import subprocess
import hashlib
import pickle
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file.")
    st.stop()


model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=google_api_key
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)


st.set_page_config(page_title="YouTube Q&A Chat", page_icon="🎥", layout="centered")
st.title("🎥 YouTube Video Q&A (Chat Mode)")

captions_dir = "captions"
cache_dir = "cache"
os.makedirs(captions_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

qa_prompt = PromptTemplate(
    template="""
You are a precise Q&A assistant.
Use ONLY the transcript context to answer the question.
Answer in 1-3 sentences maximum, directly addressing the question.
If the answer is not in the transcript, say exactly: "I don't know (not in transcript)".

Transcript context:
{context}

Question: {question}

Answer:
""",
    input_variables=['context', 'question']
)


def vtt_to_text(vtt_path):
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    transcript = []
    for line in lines:
        if "-->" in line or line.strip().isdigit() or line.strip() == "":
            continue
        transcript.append(line.strip())
    return " ".join(transcript)

def load_or_create_vectorstore(video_url):
    video_id = hashlib.md5(video_url.encode()).hexdigest()
    faiss_path = os.path.join(cache_dir, f"{video_id}.faiss")
    meta_path = os.path.join(cache_dir, f"{video_id}.pkl")

    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6}), metadata.get("title", "Unknown Video")

    with st.spinner("🔄 Downloading subtitles..."):
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "--output", os.path.join(captions_dir, "%(title)s.%(ext)s"),
            video_url
        ]
        subprocess.run(cmd, check=True)

    vtt_file = None
    for file in os.listdir(captions_dir):
        if file.endswith(".en.vtt"):
            vtt_file = os.path.join(captions_dir, file)
            video_title = file.replace(".en.vtt", "")
            break
    if not vtt_file:
        st.error("❌ Subtitle file not found.")
        st.stop()

    clean_transcript = vtt_to_text(vtt_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([clean_transcript])

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"video_url": video_url, "title": video_title}, f)

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6}), video_title


if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]

# ---------------------------
# Video Input
# ---------------------------
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Load Video") and video_url:
    try:
        retriever, video_title = load_or_create_vectorstore(video_url)
        st.session_state.retriever = retriever
        st.session_state.video_title = video_title
        st.session_state.messages = []  # Reset chat
        st.success(f"✅ Video loaded: {video_title}")
    except subprocess.CalledProcessError:
        st.error("❌ Failed to download subtitles. Check the video URL.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ---------------------------
# Chat Interface
# ---------------------------
if st.session_state.retriever:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the video transcript..."):
        # Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("💬 Thinking..."):
                retrieved_docs = st.session_state.retriever.get_relevant_documents(prompt)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs).strip()
                prompt_text = qa_prompt.format(context=context_text, question=prompt)
                resp = model.invoke([HumanMessage(content=prompt_text)])
                answer_text = resp.content.strip()
                st.markdown(answer_text)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
