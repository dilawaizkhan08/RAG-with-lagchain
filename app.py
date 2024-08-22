import streamlit as st
from langchain.memory import ConversationBufferMemory  # Import ConversationBufferMemory
from main import get_response, memory  # Import the existing memory object and get_response function
from langchain.schema import HumanMessage, AIMessage  # Import message types

# Ensure that memory is correctly initialized
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to display messages in chat format
def display_message(role, message):
    """Display a message with different styling for user and assistant."""
    if role == "user":
        st.markdown(f"<div style='text-align: right; color: blue;'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: green;'>{message}</div>", unsafe_allow_html=True)

def main():
    st.title("RAG Chatbot")

    # Display chat history
    if st.session_state.memory.chat_memory.messages:
        st.write("**Chat History:**")
        for msg in st.session_state.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                display_message("user", msg.content)
            elif isinstance(msg, AIMessage):
                display_message("assistant", msg.content)

    # Create a chat input box for user questions
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add the user input to chat history and display it
        st.session_state.memory.chat_memory.add_user_message(user_input)
        display_message("user", user_input)

        # Get the response from the RAG system
        response = get_response(user_input)
        
        # Add the response to chat history and display it
        st.session_state.memory.chat_memory.add_ai_message(response)
        display_message("assistant", response)

if __name__ == "__main__":
    main()
