from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents.base import Document
from langchain.vectorstores import FAISS

def GenerateEmbeddingFromDocuments(docs: list[Document]) -> FAISS:
    """
    This function is used to generate the Embedding from the given `List[Document]`
    """
    try:
        embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vector_store = FAISS.from_documents(docs,embedder)
        return vector_store
    except Exception as E:
        print(f"An Exception occured in Embedder. E = {E}")