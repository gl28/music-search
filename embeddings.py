import os
import openai
import pandas as pd
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAT_GPT_FILTERING_ENABLED = False

print("Loading album data from disk...")

reviews_df = pd.read_pickle("data/pitchfork_reviews_data.pkl")

print("Finished loading")
print(reviews_df.info())

def get_albums_for_query(query):
    initial_candidates = search_reviews(reviews_df, query, 20)
    formatted_candidates_string = ""

    for index, row in initial_candidates.iterrows():
        formatted_candidates_string += f'{index+1}. {row["complete_content"]}\n'

    filtered_results_indices = []
    
    if CHAT_GPT_FILTERING_ENABLED:
        filtered_results_indices = get_final_results(query, formatted_candidates_string)
    else:
        filtered_results_indices = range(1, 20)

    albums = []

    for i in filtered_results_indices:
        album = {
            "title": initial_candidates.loc[i, "title"],
            "artist": initial_candidates.loc[i, "artist"],
            "description": initial_candidates.loc[i, "complete_content"]
        }
        albums.append(album)

    return albums

# use dot product on album embeddings to find relevant candidates
def search_reviews(df, query, n=20):
   print(f"Getting initial candidates for query: {query}")
   query_embedding = get_embedding(query)
   df['similarities'] = df.embedding.apply(lambda x: np.dot(x, query_embedding))
   res = df.sort_values('similarities', ascending=False).head(n).copy()
   res = res.reset_index(drop=True)
   return res

# call OpenAI to get an embedding for a string
def get_embedding(text):
   print(f"Getting embedding for: {text}")
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=EMBEDDINGS_MODEL)['data'][0]['embedding']

# call chatGPT and ask it to narrow down the final results
def get_final_results(query, candidates):
    print("Getting final results from ChatGPT")
    prompt = f"""A user searched for music with the following query: "{query}."
    In response they received the following list of descriptions, each one corresponding to an album. Each album begins with a number (the index) and then continues with the description:
    {candidates}

    Please pick the top 8 that you think best represent their search query. Format your response as a Python list, providing only the indices of the albums. For example: [album_index_1, album_index_2]
    """
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "user", "content": prompt}
            ]
        )
    print(gpt_response)
    final_results_list = eval(gpt_response.choices[0]["message"]["content"].strip())
    return final_results_list