import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json

from openApiKey import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=openai.api_key)
def find_most_similar_text(query, embeddings_file='embeddings.json', chunks=None):
    # Read embeddings from the file
    with open(embeddings_file, 'r') as f:
        embeddings = json.load(f)

    # Generate embedding for the query
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    ).data[0].embedding

    # Compute cosine similarity between query embedding and stored embeddings
    cos_sim = cosine_similarity([query_embedding], embeddings)
    most_similar_index = np.argmax(cos_sim)  # Index of the most similar embedding
    print("Similarity score:", most_similar_index)
    # Retrieve the corresponding text
    # if chunks is not None and most_similar_index < len(chunks):
    #     most_similar_text = chunks[most_similar_index]
    #     print("Most similar text:", most_similar_text)
    #     return most_similar_text
    # else:
    #     print("Chunks not provided or index out of range")
    #     return None