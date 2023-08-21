from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from openai.embeddings_utils import cosine_similarity
import tiktoken
import openai

import pandas as pd
import os
import random
from termcolor import colored
import numpy as np



# Load .env file
env = load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Retrieve API key
API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded correctly
if not API_KEY:
    raise ValueError("API Key not found!")

openai.api_key = API_KEY



def load_embeddings_df(csv_path):
    '''load embeddings dataframe from the specified CSV file'''
    embeddings_df = pd.read_csv(csv_path, index_col=0)
    # Convert the string representation of lists to actual lists
    embeddings_df['embeddings'] = embeddings_df['embeddings'].apply(lambda x: np.array(eval(x)))
    return embeddings_df

def embed_item(content: str, embeddings_model: OpenAIEmbeddings) -> tuple:
    return embeddings_model.embed_documents([content])[0]

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    text = str(text)
    return len(encoding.encode(text))


def search_embeddings(df, query, n=3, pprint=True, n_lines=1):
    embedding = embed_item(query, embeddings_model)
    df['similarities'] = df.embeddings.apply(lambda x: cosine_similarity(x, embedding))
    # print(df.head())

    res = df.sort_values('similarities', ascending=False).head(n)
    # random colors for termcolor
    colors = ['green', 'yellow', 'blue', 'magenta']
    if pprint:
        prev_color = None
        for r in res.iterrows():            
            color = random.choice(colors)
            while color == prev_color:
                color = random.choice(colors)
            prev_color = color
            # print n_lines of the context
            print(colored(r[1].text.split('\n')[:n_lines], color))
            print()
            print('Similarity:', r[1].similarities)
            print()
            print('-' * 50)
            print()
    return res


embeddings_model = OpenAIEmbeddings()
# print(embeddings_df.head())


class GPTChat:
    def __init__(self):
        self.messages = [{'role': 'system', 'content': "Hello, I'm a Hawaii made assitant that can answer your questions about Hawaii, Hawaiian history and culture. Respond with the spirit of Aloha using hawaiin words and phrases."}]

    def take_user_input(self):
        user_input = input("You: ")
        return user_input

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def clear_messages(self):
        '''clears all messages except the system message'''
        self.messages = [self.messages[0]]

    def get_gpt_response(self, user_input):
        self.add_message('user', user_input)
        
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=self.messages,
            temperature=0,
            stream=True
        )

        responses = ""

        for chunk in response:
            response_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if response_content:
                responses += response_content
                print(response_content, end='', flush=True)

        self.add_message('assistant', responses)

        return responses
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process and embed text content from a CSV file.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file to process')

    args = parser.parse_args()

    # Create an empty dataframe
    embeddings_df = load_embeddings_df(args.csv_path)    # Pass the CSV file path as an argument


    lc_bot = GPTChat()

    while True:
        query = input("Enter a query: ")
        related_code = search_embeddings(embeddings_df, query, n=5, pprint=True, n_lines=1)
        related_code_text = '\n'.join(related_code.text.values)  # Update to 'text' column
        print(count_tokens(related_code_text))
        lc_bot.add_message('user', 'return only the content which answers the user question based on the context provided')
        lc_bot.add_message('user', f'context:\n{related_code_text}')
        response = lc_bot.get_gpt_response(query)
        
        lc_bot.add_message('user', query)
        lc_bot.add_message('assistant', response)
        
        print(f"\n\ntoken count of all messages: {count_tokens(lc_bot.messages)}")
        #lc_bot.clear_messages()