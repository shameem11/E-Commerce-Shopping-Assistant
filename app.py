import os
from dotenv import load_dotenv
from Src.retrivel_generation import get_session_history,Retriver_Generstion,Chat_Generation
from Src.data_ingestion import Data_Ingestion
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from flask import Flask, render_template, request, jsonify

chat_history = []  # to store previous chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


data_ingestion = Data_Ingestion('done')
vector_store = data_ingestion

# Generate the RAG chain
Conversational_rag_chain = Chat_Generation(vector_store)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    session_id = request.json.get('session_id')

    # Get the session history
    session_history = get_session_history(session_id)

    # Format the context properly
    context = "\n".join([f"{doc.metadata['product_name']}: {doc.page_content}" for doc in vector_store.documents])

    # Invoke the RAG chain with the user query and context
    response = Conversational_rag_chain.invoke({"input": user_message, "chat_history": session_history.messages, "context": context})

    # Update the session history with the new interaction
    session_history.add_user_message(user_message)
    session_history.add_ai_message(response['answer'])

    return jsonify({"response": response['answer']})

if __name__ == "__main__":
    app.run(debug=True)
    



