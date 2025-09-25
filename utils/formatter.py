from langchain_core.documents import Document


def format_context(docs: list[Document]) -> str:
    context = "/n/n".join(doc.page_content for doc in docs)
    return context