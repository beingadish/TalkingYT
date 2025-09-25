from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

def SplitVideoTranscript(transcript: str) -> list[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(separators=[" ", "  ", "\n", "\n\n"], chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        return chunks
    except Exception as e:
        print(f"Exception Occured = {e}")