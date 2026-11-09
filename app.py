from src.loader import load_youtube_transcript
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.rag import get_answer


def main():
    print("\n=== YouTube RAG Assistant ===\n")

    # Get YouTube URL
    url = input("Enter YouTube URL: ")

    print("\nExtracting transcript...")

    try:
        # 1. Load YouTube transcript
        documents = load_youtube_transcript(url)

        # 2. Split transcript into chunks
        chunks = split_documents(documents)

        # 3. Create FAISS vector store
        print("Creating knowledge base...")
        vectorstore = create_vectorstore(chunks)

        print("\nKnowledge base is ready!")
        print("Ask questions about the video.")
        print("Type 'exit' to stop.\n")

        # Question-answer loop
        while True:
            question = input("Ask a question: ")

            if question.lower() == "exit":
                print("\nGoodbye!")
                break

            if not question.strip():
                print("Please enter a question.")
                continue

            answer = get_answer(vectorstore, question)

            print("\nAnswer:")
            print(answer)
            print()

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()