import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002")

CHROMA_PERSIST_DIR = os.environ["CHROMA_PERSIST_DIRECTORY"]
CHROMA_COLLECTION = os.environ["CHROMA_COLLECTION"]
print(f"Starting up Chroma client with persist_directory={CHROMA_PERSIST_DIR}")

client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PERSIST_DIR))

print(f"Chroma collection={CHROMA_COLLECTION}")

collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=openai_ef)
print(collection.peek())
print(len(collection))

def get_albums_for_query(query):
    results = collection.query(query_texts=[query], n_results=5)
    albums = []

    for metadata in results["metadatas"][0]:
        album = {
            "title": metadata["title"],
            "artist": metadata["artist"],
        }
        albums.append(album)

    return albums