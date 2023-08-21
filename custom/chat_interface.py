import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import openai


from search_embeddings import load_embeddings_df, search_embeddings

# Load env file
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Load embeddings model and dataframe
embeddings_model = OpenAIEmbeddings()

args = {"csv_path": r".\custom\text\en_wikipedia_org_wiki_Hawaii.csv_embeddings.csv"}  # Update with your CSV file path
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
        
        
    # Change button colors to Hawaiian theme
    st.markdown(
        """
        <style>
            .stButton>button {
                background-color: #007BFF; /* Hawaiian color */
                color: white;
            }
            body {
                background-color: #ffffff; /* Replace with your desired color */
            }
            .title {
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Play Hawaiian music (optional)
    #st.audio("hawaii_music.mp3", format="audio/mp3")


    # Streamlit UI
    st.markdown("<h1 class='title'>🌺🌺🌺🌺 Aloha! 🌺🌺🌺🌺</h1>", unsafe_allow_html=True)

    # Display image of Alohalani
    st.image("./images/alohalani.gif", use_column_width=True)

    st.markdown("<h1 class='title'>I'm Alohalani - Your Hawaiian Assistant</h1>", unsafe_allow_html=True)

    alohalani = st.chat_message("Alohalani", avatar="🌺")

    alohalani.markdown("<h5>Hiki iaʻu ke aʻo iā ʻoe e pili ana iā Hawaiʻi.</h5>",  unsafe_allow_html=True)
    alohalani.markdown("I can teach you about Hawaii.")

        
    # Display chat messages from history on app rerun
    displayed_messages = set()  # To keep track of displayed messages

    for message in st.session_state.chatbot.messages:
        message_content = message["content"]
        if message["role"] == "user":
            if message_content not in displayed_messages:
                with st.chat_message("You", avatar="🙂"):
                    st.markdown(message_content)
                    displayed_messages.add(message_content)
        elif message["role"] == "assistant":
            if message_content not in displayed_messages:
                with st.chat_message("Alohalani", avatar="🌺"):
                    st.markdown(message_content)
                    displayed_messages.add(message_content)


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
        
        
if __name__ == "__main__":
    main()

