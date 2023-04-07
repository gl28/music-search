import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002")

client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet",
            persist_directory="embeddings_data"))

collection = client.get_or_create_collection(name="top_5000_reviews", embedding_function=openai_ef)

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