from dotenv import load_dotenv
from indexing.document_load import FetchTranscript
from indexing.splitter import SplitVideoTranscript
from langchain.retrievers import MultiQueryRetriever
from indexing.embedder import GenerateEmbeddingFromDocuments
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from utils.formatter import format_context
from assistant.prompt import AssistantPrompt
from utils.parser import parser

load_dotenv()

vid = "OYvlznJ4IZQ"

# Fetch Transcripts
transcript = FetchTranscript(vid)

# Split them into documents
docs = SplitVideoTranscript(transcript)

# Create Embedding from the Docs & Store it to VectorStore

vector_store = GenerateEmbeddingFromDocuments(docs)

# Creating A retriever from the Vector Store

base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True
)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_context),
    'question': RunnablePassthrough()
})


ask = parallel_chain | AssistantPrompt() | llm | parser()

question = "What is vectorization? & What is Model Context Protocol?"

answer = ask.invoke(question)

print(answer)