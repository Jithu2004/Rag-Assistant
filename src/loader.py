
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs


def get_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    if parsed_url.hostname in ["www.youtube-nocookie.com", "youtube-nocookie.com"]:
        return parsed_url.path.strip("/").split("/")[0]

    return None


def load_youtube_transcript(url):
    video_id = get_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    documents = []

    for index, snippet in enumerate(transcript):
        start = float(snippet.start)
        duration = float(snippet.duration)
        end = start + duration

        documents.append(
            Document(
                page_content=snippet.text,
                metadata={
                    "source": url,
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "chunk_index": index
                }
            )
        )

    return documents
