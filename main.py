import os
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the language model with Groq API
llm = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="llama3-70b-8192")

# Load text from a file
file_path = "dummy.txt"  # Replace with your actual file path

with open(file_path, "r") as file:
    text = file.read()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

# Initialize embeddings using HuggingFace model
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}  # Force CPU usage
encode_kwargs = {'normalize_embeddings': False}

embeddings1 = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Create a Chroma vector store
vector_store = Chroma.from_texts(chunks, embeddings1)

# Prepare conversation memory and prompt templates
general_system_template = """
You are a helpful assistant. If the answer to the user's question is not found in the provided context, 
respond with: 'The information you requested is not available in the provided context.'

----
chat history = {chat_history}
----
context = {context}
----
human question =  {question}
----
"""

general_user_template = "Question:```{question}```"

messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]
aqa_prompt = ChatPromptTemplate.from_messages(messages)

# Initialize retriever and conversation chain
retriever = vector_store.as_retriever(k=1)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": aqa_prompt}, verbose=False
)

def get_response(question):
    return conversation.run({'question': question})
























# import os
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.document_loaders import TextLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain.chains import create_retrieval_chain
# from langchain import hub
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )

# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Access the Groq API key
# API_KEY = os.getenv("GROQ_API_KEY")
# # Disable parallelism for tokenizers to avoid deadlocks
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Initialize the language model with Groq API
# llm = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="llama3-70b-8192")

# # Load text from a file
# file_path = "dummy.txt"  # Replace with your actual file path

# with open(file_path, "r") as file:
#     text = file.read()

# # Split the text into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
# chunks = text_splitter.split_text(text)

# # Initialize embeddings using HuggingFace model
# modelPath = "sentence-transformers/all-MiniLM-l6-v2"
# model_kwargs = {'device': 'cpu'}  # Force CPU usage
# encode_kwargs = {'normalize_embeddings': False}

# embeddings1 = HuggingFaceEmbeddings(
#     model_name=modelPath,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# # Create a Chroma vector store
# vector_store = Chroma.from_texts(chunks, embeddings1)

# # Prepare conversation memory and prompt templates
# general_system_template = """
# You are a helpful assistant. If the answer to the user's question is not found in the provided context, 
# respond with: 'The information you requested is not available in the provided context.'

# ----
# chat history = {chat_history}
# ----
# context = {context}
# ----
# human question =  {question}
# ----
# """

# general_user_template = "Question:```{question}```"

# messages = [
#     SystemMessagePromptTemplate.from_template(general_system_template),
#     HumanMessagePromptTemplate.from_template(general_user_template)
# ]
# aqa_prompt = ChatPromptTemplate.from_messages(messages)

# # Initialize retriever and conversation chain
# retriever = vector_store.as_retriever(k=1)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# conversation = ConversationalRetrievalChain.from_llm(
#     llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": aqa_prompt}, verbose=False
# )

# def interactive_conversation():
#     print("You can start asking questions. Type 'exit' to quit.")
#     while True:
#         question = input("Your question: ")
#         if question.lower() == 'exit':
#             print("Exiting the conversation.")
#             break
#         response = conversation.run({'question': question})
#         print(f"Answer: {response}")

# if __name__ == "__main__":
#     interactive_conversation()
