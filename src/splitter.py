
def split_documents(documents):
    """
    Keep YouTube transcript segments separate
    so timestamp-based questions work correctly.
    """

    chunks = []

    for index, doc in enumerate(documents):
        if not doc.page_content.strip():
            continue

        doc.metadata["chunk_index"] = index
        chunks.append(doc)

    return chunks
