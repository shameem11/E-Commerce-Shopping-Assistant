import os
import uuid
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from Src.retrivel_generation import get_session_history, Retriver_Generstion, Chat_Generation
from Src.data_ingestion import Data_Ingestion
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq

chat_history = []  # to store previous chat histories
store = {}

# Initialize Data Ingestion
data_ingestion = Data_Ingestion('done')
vector_store = data_ingestion

# Generate the RAG chain
Conversational_rag_chain = Chat_Generation(vector_store)

app = Flask(__name__)

# Helper function to generate a new session ID
def generate_session_id():
    return str(uuid.uuid4())

@app.route('/')
def index():
    # Generate a new session ID for each visit (e.g., upon refreshing the page)
    session_id = generate_session_id()
    return render_template('index.html', session_id=session_id)

@app.route('/chat', methods=['POST'])
def chat():
    # Get user message and session ID from the request
    user_message = request.json.get('message')
    session_id = request.json.get('session_id')

    if not user_message or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400

    # Get the session history for the current session
    session_history = get_session_history(session_id)

    # Invoke the RAG chain with the user message and session ID
    response = Conversational_rag_chain.invoke(
        {"input": user_message},  # Use the user message here
        config={"configurable": {"session_id": session_id}}  # Use the actual session_id from request
    )

    # Update the session history with the new interaction
    session_history.add_user_message(user_message)
    session_history.add_ai_message(response['answer'])

    # Return the response as JSON
    return jsonify({"response": response['answer']})

if __name__ == "__main__":
    app.run(debug=True)
