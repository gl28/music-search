import sqlite3
import os
import openai
import time
import datetime
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
SQLITE_DB_NAME = "../data/reviews_database.sqlite"

conn = sqlite3.connect(SQLITE_DB_NAME)
cursor = conn.cursor()

# load concatenated review text
cursor.execute('''
    SELECT compressed_content.reviewId, compressed_content.complete_content, reviews.artist, reviews.title, reviews.score, reviews.pub_date
    FROM compressed_content
    INNER JOIN reviews ON compressed_content.reviewId = reviews.reviewId
''')
rows = cursor.fetchall()

documents = []
metadatas = []
ids = []

for row in rows:
    review_id, content, artist, title, score, pub_date = row
    if not content or not review_id or not content or not artist or not title or not score or not pub_date:
        continue

    documents.append(content)
    metadatas.append({"artist": artist, "title": title, "score": score, "pub_date": pub_date})
    ids.append(str(review_id))

conn.close()


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002")

client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet",
             persist_directory="../embeddings_data"))

collection = client.get_or_create_collection(name="compressed_reviews", embedding_function=openai_ef)

# embedding and persisting the reviews in chunks seems to work better and is easier to monitor the progress
chunk_size = 500
start = collection.count()
total_items = len(documents)
start_time = time.time()

for i in range(start, total_items, chunk_size):
    elapsed_time = time.time() - start_time
    elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
    print(f"Index {i}/{total_items}. Elapsed time: {elapsed_time_str}.")

    print("Starting " + str(i))
    collection.add(documents=documents[i:i+chunk_size], ids=ids[i:i+chunk_size], metadatas=metadatas[i:i+chunk_size])
    print("Done with " + str(i))
    print("Persisting...")
    client.persist()
    print("Done persisting.")
    print("Current total items " + str(collection.count()))