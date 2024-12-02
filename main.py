import json
import openai
import tiktoken
from charset_normalizer import detect

import re
from nltk.corpus import stopwords
from langdetect import detect

from Query import find_most_similar_text
from openApiKey import OPENAI_API_KEY
from ReadIsdaFile import process_pdf_file
import os

# print(os.getenv("OPENAI_API_KEY"))

def read_config_and_process_file():
    # Read the JSON configuration file
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Extract the file path
    file_path = config.get('file_path')

    if file_path:
        # Call the function to process the file in `ReadIsdaFile.py`
        return process_pdf_file(file_path)
    else:
        raise ValueError("File path not found in configuration.")



def split_text_into_chunks(text, max_tokens=8192):
    encoding = tiktoken.get_encoding("cl100k_base")  # Tokenizer for GPT models
    tokens = encoding.encode(text)

    # Split tokens into chunks that do not exceed the max_tokens limit
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    # Decode the chunks back into text
    chunk_texts = [encoding.decode(chunk) for chunk in chunks]
    return chunk_texts

def clean_text(text):
    # Step 1: Check for language (optional)
    if detect(text) != 'en':
        return ""  # Skip non-English text

    # Step 2: Remove URLs, emails, and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags

    # Step 3: Remove special characters, numbers, and short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join([word for word in text.split() if len(word) > 2])  # Remove short words

    # Step 4: Convert to lowercase and remove stop words
    stop_words = set(stopwords.words('english'))  # Corrected line
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Function to get embeddings for text chunks
def get_embeddings_for_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        chunk = clean_text(chunk)
        try:
            # Get embeddings from OpenAI API
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            # Extract embedding from the response
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing chunk: {e}")
    return embeddings

def store_embeddings_to_file(embeddings, file_name='embeddings.json'):
    try:
        with open(file_name, 'w') as f:
            json.dump(embeddings, f)
        print(f"Embeddings successfully stored in {file_name}")
    except Exception as e:
        print(f"Error storing embeddings to file: {e}")

def main():
    # Step 1: Read and process the file
    if os.path.exists("embeddings.json") and os.path.getsize("embeddings.json") > 0:
        with open('embeddings.json', 'r') as f:
            embeddings = json.load(f)
            query = "What are the market conventions for USD swaps?"
            find_most_similar_text(query, chunks=None)
    else:
        processed_text = read_config_and_process_file()

        # Step 2: Embed the processed text
        chunks = split_text_into_chunks(processed_text)
        embedding = get_embeddings_for_chunks(chunks)
        store_embeddings_to_file(embedding)
        query = "What are the market conventions?"
        find_most_similar_text(query, chunks=None)
    # Output the result, or use it further as needed
    #print("Text Embedding:", embedding)


if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=openai.api_key)
    main()
