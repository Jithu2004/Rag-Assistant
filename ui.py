
import streamlit as st

from src.loader import load_youtube_transcript
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.rag import get_answer


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)


# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

.chat-box {
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Session state
# -------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_url" not in st.session_state:
    st.session_state.processed_url = None


# -------------------------------
# Application heading
# -------------------------------

st.title("🎥 YouTube RAG Assistant")

st.write(
    "Enter a YouTube URL, process the video, "
    "and ask questions about its content."
)

st.divider()


# -------------------------------
# YouTube URL input
# -------------------------------

st.subheader("📺 Process YouTube Video")

video_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("🚀 Process Video"):

    if not video_url.strip():

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            with st.spinner("Extracting YouTube transcript..."):

                documents = load_youtube_transcript(video_url)

            if not documents or not documents[0].page_content.strip():
                st.error("The video transcript is empty.")
                st.stop()

            transcript = documents[0].page_content

            st.success("Transcript extracted successfully!")

            # Show transcript information
            st.write("Transcript length:", len(transcript), "characters")

            with st.expander("View transcript preview"):
                st.write(transcript[:2000])

            with st.spinner("Splitting transcript into chunks..."):

                chunks = split_documents(documents)

            # Preserve original chunk order for summarization
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = index

            with st.spinner("Creating FAISS knowledge base..."):

                vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.processed_url = video_url
            st.session_state.messages = []

            st.success(
                f"Video processed successfully! "
                f"{len(chunks)} chunks created."
            )

        except Exception as error:

            st.error(f"Error processing video: {error}")


st.divider()


# -------------------------------
# Chat interface
# -------------------------------

st.subheader("💬 Ask Questions")

if st.session_state.vectorstore is None:

    st.info("First enter a YouTube URL and click Process Video.")

else:

    st.success("Your video knowledge base is ready.")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User question
    question = st.chat_input(
        "Ask something about the video..."
    )

    if question:

        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Analyzing video transcript..."):

                try:

                    answer = get_answer(
                        st.session_state.vectorstore,
                        question
                    )

                    st.markdown(answer)

                except Exception as error:

                    answer = f"Error generating answer: {error}"
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# -------------------------------
# Footer
# -------------------------------

st.divider()

st.caption(
    "Built with LangChain • FAISS • HuggingFace Embeddings • Ollama"
)
