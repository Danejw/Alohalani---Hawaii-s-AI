
import openai
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st


class GPTChat:
    def __init__(self):
        self.messages = []
        self.isGenerating = False
        self.model = "gpt-3.5-turbo"

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def clear_messages(self):
        '''clears all messages except the system message'''
        self.messages = [self.messages[0]]

    def get_gpt_response(self, user_input, message_placeholder=None):
        self.add_message('user', user_input)

        full_response = ""

        sentMessages: list[dict[str,str]] = []
        if len(self.messages) > 5:
            sentMessages = self.messages[-5:]
        else:
            sentMessages = self.messages

        self.isGenerating = True
        for response in openai.ChatCompletion.create(
            model=self.model,
            messages=sentMessages,
            temperature=0,
            stream=True
        ):

            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        self.add_message('assistant', full_response)
               
        self.isGenerating = False
            
        return full_response


class AlohaLaniChat(GPTChat):
    def __init__(self):
        super().__init__()
        self.messages = [{'role': 'system', 'content': "Your name is Alohalani and you are an assistant made to questions about Hawaii, Hawaiian history, and culture. Respond with the spirit of Aloha using Hawaiian words and phrases. Make sure to answer any user question in the context of Hawaii only. If the user asks about something other that the context of Hawaii, pollitely let them know that you specialize in Hawaii topics. Use emojis to express your emotions."}]
        self.isGenerating = False
        self.model = "gpt-3.5-turbo"
