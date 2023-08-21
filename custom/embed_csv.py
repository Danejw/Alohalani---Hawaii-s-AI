import os
from dotenv import load_dotenv
import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.embeddings import OpenAIEmbeddings
import tenacity

# set up logging
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Load .env file
# Load .env file from the root of the project
env = load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Retrieve API key
API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded correctly
if not API_KEY:
    raise ValueError("API Key not found!")

openai.api_key = API_KEY



# Implement exponential backoff, try 10 times, wait 1 second after the first failure,
# then 2 seconds, then 4 seconds, then 8 seconds, then 16 seconds, then 32 seconds, and so on
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=2, min=1, max=60),
    stop=tenacity.stop_after_attempt(10),
    after=tenacity.after_log(logger, logging.DEBUG)
)
def embed_text(file_name: str, text_content: str, embeddings_model: OpenAIEmbeddings) -> tuple:
    return file_name, embeddings_model.embed_documents([text_content])[0]

def create_empty_embeddings_df():
    '''Create text and embeddings dataframe'''
    embeddings_df = pd.DataFrame(columns=['text', 'embeddings'])
    return embeddings_df

def add_text_to_df(csv_path, embeddings_df):
    '''Add text to dataframe'''
    df = pd.read_csv(csv_path, encoding='utf-8')  # Remove the 'errors' parameter
    for index, row in df.iterrows():
        header = row['Header'] if not pd.isna(row['Header']) else ''  # Check for NaN in "Header"
        paragraph = row['Paragraph'] if not pd.isna(row['Paragraph']) else ''  # Check for NaN in "Paragraph"
        content = header + '\n\n' + paragraph
        embeddings_df.loc[f'{os.path.basename(csv_path)}_row{index}'] = [content, None]
    return embeddings_df


def embed_concurrent(csv_path, embeddings_df, embeddings_model, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(embed_text, file_name, embeddings_df.loc[file_name]['text'], embeddings_model): file_name
            for file_name in embeddings_df.index
        }
        for future in as_completed(futures):
            file_name, embedding = future.result()
            embeddings_df.at[file_name, 'embeddings'] = embedding

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process and embed text content from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to process')

    args = parser.parse_args()

    # Create an empty dataframe
    embeddings_df = create_empty_embeddings_df()

    # Add text from the specified CSV file to the dataframe
    embeddings_df = add_text_to_df(args.csv_file, embeddings_df)

    print(embeddings_df.head())
    print(embeddings_df.shape)

    # Initialize the OpenAI embeddings model
    embeddings_model = OpenAIEmbeddings()

    # Embed the text content
    embed_concurrent(args.csv_file, embeddings_df, embeddings_model)

    # Save the embeddings dataframe to a CSV file
    embeddings_df.to_csv(f"{args.csv_file}_embeddings.csv")

    # Save the embeddings dataframe to a JSON file
    embeddings_df.to_json(f"{args.csv_file}_embeddings.json")
