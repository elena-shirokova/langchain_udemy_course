import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone, Chroma
import pinecone
import chromadb

from config import INDEX_NAME, DATA_PATH

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRON"],
)

def ingest_docs():
    loader = ReadTheDocsLoader(path=DATA_PATH)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs-test", "https:/")
        doc.metadata.update({"source": new_url})
    
    embeddings = OpenAIEmbeddings()
    
    print(f"Going to add {len(documents)} to Pinecone")
    # vectorstore = Chroma(INDEX_NAME, embeddings)
    # create simple ids
    ids = [str(i) for i in range(1, len(documents) + 1)]
    # new_client = chromadb.EphemeralClient()
    # add data
    # Chroma.from_documents(documents, embeddings, collection_name=INDEX_NAME)
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()