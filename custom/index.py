from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from gpt import GPTChat

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from gpt import Embeddings
import os

# Vector Dance
# Create an index
# Create a Retriever from that index
# Create a question answering chain
# Ask questions!

# Load the document
raw_documents = TextLoader('custom\\text\\Hawaii.txt', autodetect_encoding=True).load()
#raw_documents = TextLoader('custom\\text\\HawaiiHistory.txt').load()

print(raw_documents)

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)

documents = text_splitter.split_documents(raw_documents)

print(documents[0].page_content)

#embed each chunk and load it into the vector store.
db = Chroma.from_documents(documents, Embeddings().model)

llm = OpenAI(openai_api_key="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")
# llm = ChatOpenAI(temperature=0, openai_api_key="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")
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=db.as_retriever(), llm=llm
# )

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


query = "How did Queen Liluokalani respond to the annexation of Hawaii?"



#unique_docs = retriever_from_llm.get_relevant_documents(query = query)
#print (unique_docs)

lc_bot = GPTChat()
import streamlit as st

while True:
    query = input("Enter a query: ")
    
    # Create a retiriever from the vector store
    retriever = db.similarity_search_with_relevance_scores(query, k=10)
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    #result = qa.run(query, )
    lc_bot.add_message('user', 'return only the content which answers the user question based on the context provided')
    lc_bot.add_message('user', f'context:\n{retriever}')
    
    message_placeholder = st.empty()
    
    response = lc_bot.get_gpt_response(query, message_placeholder)
        
    lc_bot.add_message('user', query)
    lc_bot.add_message('assistant', response)

    print(response)


    #print(f"\n\ntoken count of all messages: {count_tokens(lc_bot.messages)}")
    #lc_bot.clear_messages()