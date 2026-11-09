# RAG Assistant 🎥

An AI-powered application that allows users to ask questions about YouTube videos using Retrieval-Augmented Generation (RAG).

## Features
- Extracts transcripts from YouTube videos.
- Answers questions about video content.
- Summarizes entire videos.
- Supports timestamp-based questions.
- Summarizes content between timestamps.
- Interactive chatbot interface.

## Technologies Used
- Python
- Streamlit
- LangChain
- YouTube Transcript API
- HuggingFace Embeddings
- FAISS
- Ollama
- Llama 3.2

## How It Works
1. Enter a YouTube video URL.
2. Extract the video transcript.
3. Process transcript segments and preserve timestamps.
4. Generate embeddings and store them in FAISS.
5. Retrieve relevant content based on user questions.
6. Generate answers using Llama 3.2.
7. Display answers through Streamlit.

## Installation

### 1. Clone the repository
git clone https://github.com/Jithu2004/Rag-Assistant.git
cd Rag-Assistant

### 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download Ollama model
ollama pull llama3.2

### 5. Run the application
streamlit run ui.py

## Example Questions
- What is this video about?
- What are the main points?
- What did the speaker say at 3:22?
- Summarize between 3:20 and 4:33.

