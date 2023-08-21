"""
This script contains the implementation of a chat interface for Alohalani, an AI assistant specialized in answering questions about Hawaii, Hawaiian history, and culture. The chat interface is built using Streamlit and OpenAI's GPT-3 model. The script defines a GPTChat class that handles user input, generates responses using the GPT-3 model, and stores the chat history. The script also loads a pre-trained embeddings model and a dataframe containing embeddings for Hawaiian text. The chat interface allows users to interact with Alohalani by asking questions about Hawaii, Hawaiian history, and culture. Alohalani responds to user questions with the spirit of Aloha using Hawaiian words and phrases. The chat interface also provides users with accurate and engaging information about Hawaii's rich heritage, breathtaking landscapes, and vibrant traditions.
"""
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import openai


from search_embeddings import load_embeddings_df, search_embeddings

# Provide api key
#openai.api_key = st.secrets["OPENAI_API_KEY"]


# Load embeddings model and dataframe
embeddings_model = OpenAIEmbeddings()

args = {"csv_path": r".\custom\text\Hawaii_embeddings.csv"}  # Update with your CSV file path
embeddings_df = load_embeddings_df(args["csv_path"])

model = "gpt-3.5-turbo"


class GPTChat:
    def __init__(self):
        self.messages = [{'role': 'system', 'content': "Your name is Alohalani and you are an assistant made to questions about Hawaii, Hawaiian history, and culture. Respond with the spirit of Aloha using Hawaiian words and phrases. Make sure to answer any user question in the context of Hawaii only. If the user asks about something other that the context of Hawaii, pollitely let them know that you specialize in Hawaii topics. Use emojis to express your emotions."}]

    def take_user_input(self):
        user_input = st.text_input("You:", key="user_input")
        return user_input

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def clear_messages(self):
        '''clears all messages except the system message'''
        self.messages = [self.messages[0]]

    def get_gpt_response(self, user_input, message_placeholder=None):
        self.add_message('user', user_input)

        full_response = ""

        for response in openai.ChatCompletion.create(
            model=model,
            messages=self.messages,
            temperature=0,
            stream=True
        ):

            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        self.add_message('assistant', full_response)

        return full_response


# Set the page configurations
st.set_page_config(
    page_title="Alohalani - Your Hawaiian Assistant",
    page_icon="🌺",  # Change to your desired icon
    layout="centered",
    initial_sidebar_state="expanded",
)

def main():

    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = GPTChat()
        

    # Play Hawaiian music (optional)
    #st.audio("hawaii_music.mp3", format="audio/mp3")

    # Bring your own key
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    openai_link = "https://platform.openai.com/account/api-keys"
    
    
    if api_key:
        openai.api_key = api_key
    else:
        st.markdown("<div style='text-align: center'>Don't have an api key yet? Get one <a href='" + openai_link + "'>here.</a> 👈🏾</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            .title {
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Streamlit UI
    st.markdown("<h1 class='title'>🌺🌺🌺🌺 Aloha! 🌺🌺🌺🌺</h1>", unsafe_allow_html=True)

    # Display image of Alohalani
    st.image("./images/alohalani.gif", use_column_width=True)
    
    # Introudctory methods
    st.markdown("<h1 class='title'>I'm Alohalani - Your Hawaiian Assistant 🤙🏼</h1>", unsafe_allow_html=True)

    alohalani = st.chat_message("Alohalani", avatar="🌺")

    alohalani.markdown("<h5>Hiki iaʻu ke aʻo iā ʻoe e pili ana iā Hawaiʻi.</h5>",  unsafe_allow_html=True)
    alohalani.markdown("""
                       🌺 Aloha everyone! 🌺

I am thrilled to introduce myself as Alohalani, your go-to assistant for all things Hawaii! 🌴🌺 Whether you have questions about Hawaiian history, culture, or simply want to immerse yourself in the spirit of Aloha, I am here to help you navigate the beautiful islands of Hawaii.

🌺 As an AI assistant, I specialize in providing information and insights about Hawaii's rich heritage, breathtaking landscapes, and vibrant traditions. From the majestic volcanoes of the Big Island to the stunning beaches of Maui, I can guide you through the wonders of this tropical paradise.

🌺 With a deep understanding of Hawaiian culture, I am here to share the spirit of Aloha with you. Whether you're planning a trip to Hawaii, researching for a project, or simply curious about the islands, I am here to provide you with accurate and engaging information.

🌺 So, let's embark on this virtual journey together! Feel free to ask me anything about Hawaii, Hawaiian history, or culture, and I'll respond with the warmth and hospitality that embodies the Aloha spirit. 🌺

🌺 Mahalo nui loa! 🌺""")

        
    # Display chat messages from history on app rerun
    displayed_messages = set()  # To keep track of displayed messages

    for message in st.session_state.chatbot.messages:
        message_content = message["content"]
        if message["role"] == "user":
            if message_content not in displayed_messages:
                with st.chat_message("You", avatar="🙂"):
                    st.markdown(message_content)
                    displayed_messages.add(message_content)
                    st.markdown("---")  # Add a horizontal line as a separator
        elif message["role"] == "assistant":
            if message_content not in displayed_messages:
                with st.chat_message("Alohalani", avatar="🌺"):
                    st.markdown(message_content)
                    displayed_messages.add(message_content)
                    st.markdown("---")  # Add a horizontal line as a separator
        



    if user_input := st.chat_input("Ask me anything about Hawaii, Hawaiian history, or its culture.", key="user_input"):
        related_code = search_embeddings(embeddings_df, user_input, n=5, pprint=True, n_lines=1)
        related_code_text = '\n'.join(related_code.text.values)  # Update to 'text' column
        
        # Display user message in chat message container
        with st.chat_message("user", avatar="🙂"):
            st.markdown(user_input)
            st.session_state.chatbot.add_message("user", user_input)

            
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🌺"):
            message_placeholder = st.empty()
            full_response = ""
            
            full_response = st.session_state.chatbot.get_gpt_response(user_input, message_placeholder)
            
            st.session_state.chatbot.add_message("assistant", full_response)
        
    

    # Links to display below chat input
    link1 = "https://ko-fi.com/danejw"
    link2 = "https://buy.stripe.com/bIY7uW5Mz1hubdu000"
    link3 = "https://github.com/Danejw/Alohalani---Hawaii-s-AI"

    # Create a column layout for the links
    col1, col2, col3 = st.columns(3)

    # Add your links to the columns
    with col1:
        st.write("<div style='text-align: center'><a href='" + link1 + "'>Buy Me Coffee </a>☕️</div>", unsafe_allow_html=True)

    with col2:
        st.write("<div style='text-align: center'><a href='" + link2 + "'>Fuel My Creatitvity </a>❤️🔥</div>", unsafe_allow_html=True)

    with col3:
        st.write("<div style='text-align: center'><a href='" + link3 + "'>Github </a>⭐</div>", unsafe_allow_html=True)
        
    
if __name__ == "__main__":
    main()

