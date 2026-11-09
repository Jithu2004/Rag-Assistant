
import re
from langchain_ollama import ChatOllama


# Initialize local Llama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ---------------------------------
# Convert timestamp into seconds
# ---------------------------------

def to_seconds(timestamp):
    parts = [int(x) for x in timestamp.split(":")]

    if len(parts) == 2:
        return parts[0] * 60 + parts[1]

    if len(parts) == 3:
        return (
            parts[0] * 3600
            + parts[1] * 60
            + parts[2]
        )

    return None


# ---------------------------------
# Detect timestamp range
# ---------------------------------

def extract_time_ranges(question):

    match = re.search(
        r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:to|until|-)\s*(\d{1,2}:\d{2}(?::\d{2})?)',
        question.lower()
    )

    if not match:
        return None

    start_time = to_seconds(match.group(1))
    end_time = to_seconds(match.group(2))

    if end_time <= start_time:
        return None

    return start_time, end_time


# ---------------------------------
# Detect single timestamp
# ---------------------------------

def extract_timestamp(question):

    match = re.search(
        r'\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b',
        question
    )

    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))

    return hours * 3600 + minutes * 60 + seconds


# ---------------------------------
# Format seconds into MM:SS
# ---------------------------------

def format_time(seconds):

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes}:{remaining_seconds:02d}"


# ---------------------------------
# Get all transcript documents
# ---------------------------------

def get_all_documents(vectorstore):

    documents = list(
        vectorstore.docstore._dict.values()
    )

    documents.sort(
        key=lambda doc: doc.metadata.get("chunk_index", 0)
    )

    return documents


# ---------------------------------
# Summarize transcript in batches
# ---------------------------------

def summarize_batch(text):

    prompt = f"""
You are an expert YouTube video summarizer.

Read the transcript carefully.

Identify:
- Main topic
- Important points
- Key concepts
- Examples discussed
- Important conclusions

Use only the information in the transcript.
Do not invent information.

Transcript:
{text}

Summary:
"""

    response = llm.invoke(prompt)

    return response.content


# ---------------------------------
# Full video summary
# ---------------------------------

def summarize_entire_video(vectorstore):

    all_documents = get_all_documents(vectorstore)

    if not all_documents:
        return "No transcript content is available."

    batch_size = 8
    summaries = []

    for i in range(0, len(all_documents), batch_size):

        batch = all_documents[i:i + batch_size]

        batch_text = "\n\n".join(
            doc.page_content for doc in batch
        )

        summary = summarize_batch(batch_text)

        summaries.append(summary)

    # Combine summaries if necessary
    while len(summaries) > batch_size:

        new_summaries = []

        for i in range(0, len(summaries), batch_size):

            combined_text = "\n\n".join(
                summaries[i:i + batch_size]
            )

            new_summaries.append(
                summarize_batch(combined_text)
            )

        summaries = new_summaries

    final_context = "\n\n".join(summaries)

    final_prompt = f"""
You are a helpful YouTube video assistant.

Summarize the video using the transcript summaries below.

Include:
1. Main topic
2. Main points
3. Important concepts
4. Examples
5. Overall conclusion

Use simple language.
Do not invent information.

Transcript summaries:
{final_context}

Video summary:
"""

    return llm.invoke(final_prompt).content


# ---------------------------------
# Timestamp range summary
# ---------------------------------

def summarize_time_range(vectorstore, question):

    time_range = extract_time_ranges(question)

    if not time_range:
        return None

    start_time, end_time = time_range

    all_documents = get_all_documents(vectorstore)

    matching_documents = [
        doc for doc in all_documents
        if "start" in doc.metadata
        and "end" in doc.metadata
        and doc.metadata["start"] < end_time
        and doc.metadata["end"] > start_time
    ]

    if not matching_documents:

        return (
            f"No transcript content found between "
            f"{format_time(start_time)} and "
            f"{format_time(end_time)}."
        )

    context = "\n\n".join(
        f"[{format_time(doc.metadata['start'])}] "
        f"{doc.page_content}"
        for doc in matching_documents
    )

    prompt = f"""
You are a YouTube video assistant.

Summarize what the speaker discusses between
{format_time(start_time)} and {format_time(end_time)}.

Include:
1. Main topic discussed
2. Important points
3. Examples or explanations
4. Key takeaway

Use only the transcript provided.
Do not include information from outside this time range.
Do not invent information.

Transcript:
{context}

Summary:
"""

    return llm.invoke(prompt).content


# ---------------------------------
# Single timestamp question
# ---------------------------------

def answer_timestamp_question(vectorstore, question):

    target_time = extract_timestamp(question)

    if target_time is None:
        return None

    all_documents = get_all_documents(vectorstore)

    nearby_documents = [
        doc for doc in all_documents
        if "start" in doc.metadata
        and abs(doc.metadata["start"] - target_time) <= 30
    ]

    if not nearby_documents:

        return (
            f"No transcript segments found near "
            f"{format_time(target_time)}."
        )

    nearby_documents.sort(
        key=lambda doc: doc.metadata["start"]
    )

    context = "\n\n".join(
        f"[{format_time(doc.metadata['start'])}] "
        f"{doc.page_content}"
        for doc in nearby_documents
    )

    prompt = f"""
You are a YouTube transcript assistant.

The user wants to know what the speaker said
around {format_time(target_time)}.

Explain the relevant spoken content clearly.
Use only the transcript.
Do not invent dialogue.

Transcript:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt).content


# ---------------------------------
# Detect full video summary questions
# ---------------------------------

def is_summary_question(question):

    summary_words = [
        "what is the video about",
        "what are main points",
        "what are the main points",
        "main points",
        "key points",
        "summarize",
        "summary",
        "give me a summary",
        "explain the whole video",
        "main topic",
        "what does this video explain",
        "what did the speaker discuss",
        "explain this video",
        "important topics"
    ]

    question = question.lower().strip()

    return any(
        word in question for word in summary_words
    )


# ---------------------------------
# Main RAG function
# ---------------------------------

def get_answer(vectorstore, question):

    # 1. Timestamp range questions
    range_answer = summarize_time_range(
        vectorstore,
        question
    )

    if range_answer is not None:
        return range_answer

    # 2. Single timestamp questions
    timestamp_answer = answer_timestamp_question(
        vectorstore,
        question
    )

    if timestamp_answer is not None:
        return timestamp_answer

    # 3. Full video summary questions
    if is_summary_question(question):
        return summarize_entire_video(vectorstore)

    # 4. Normal question answering using FAISS
    documents = vectorstore.similarity_search(
        question,
        k=5
    )

    if not documents:
        return "I couldn't find relevant information in the video transcript."

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = f"""
You are an intelligent YouTube video assistant.

Answer the user's question using ONLY the transcript context.

Instructions:
1. Give a direct and relevant answer.
2. Explain concepts in simple language.
3. Do not make up information.
4. Do not introduce unrelated topics.
5. If the answer is not available in the context, say:
   "I couldn't find that information in the video transcript."

Transcript context:
{context}

User question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
