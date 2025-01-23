import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from Src.data_ingestion import Data_Ingestion

api_key = os.getenv('Gork_API_KEY')

# groq model
llm_Model = ChatGroq(temperature=0.5, groq_api_key=api_key, model_name="mixtral-8x7b-32768")

chat_history = []  # to store previous chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# retriver generation from astra vector db
def Retriver_Generstion(vector_store):
    Retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ("""
    You are an intelligent eCommerce assistant designed to reformulate questions for clarity and context. Your task is to analyze the chat history and the latest user question to determine whether the question depends on prior context. Follow these steps:

    1. **Context-Dependent Questions**:  
     - If the latest user question references or relies on information from the chat history, rewrite it as a standalone question by incorporating the necessary context from the history.  
     - Ensure the reformulated question is complete, concise, and easy to understand.  

    2. **Standalone Questions**:  
      - If the latest user question is already self-contained and does not require additional context from the chat history, return it without making any changes.  

    3. Maintain the focus on eCommerce-related queries and ensure the reformulated question is relevant to the product or shopping context.

    **Output**: A concise, clear, and self-contained question, ready for further processing.
     """)


    Retriever_template = ChatPromptTemplate.from_messages(
        [("system",retriever_prompt),
         MessagesPlaceholder(variable_name='chat_history'),
         ("human", "{input}")]
    )

    # Create history-aware retriever
    Vector_Retriever = create_history_aware_retriever(llm_Model, Retriever, Retriever_template)

    return Vector_Retriever

def Chat_Generation(vector_store):
    # Create the retriever
    Vector_Retriever = Retriver_Generstion(vector_store)

    # Define the system prompt for the e-commerce assistant
    system_prompt = """
    You are an eCommerce assistant. Your primary task is to recommend products based on user queries. You should adhere to the following guidelines:

    1. **Relevance**: Always ensure your responses are highly relevant to the query and product catalog.
    2. **Politeness**: If the requested product is not available in the database, suggest an alternative product that fits the user's query. Be polite and guide the user toward other options without frustration.
    3. **Product Scope**: Avoid discussing topics unrelated to the product catalog. Stick to information about the products, such as features, specifications, and recommendations.
    4. **Follow-up Questions**: Use follow-up questions to clarify ambiguities if needed (e.g., 'What features are you looking for in a gaming laptop?').
    5. **Concise Responses**: Provide concise and informative answers, and avoid over-explaining.

    If a specific product requested by the user is unavailable, say something like:

    - "I couldn't find that specific product in our catalog. However, here are some similar options that might interest you:"
    - "It looks like that product isn't available right now. Would you like to explore similar products?"


   **Context**:  
    {context}

   **User Input**:  
    {input}

   **Your Answer**:  
   """


    output_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # System prompt
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
            ("human", "{input}")  # Placeholder for the user's input]
        ]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm_Model, output_prompt)
    rag_chain = create_retrieval_chain(Vector_Retriever, question_answer_chain)

    # Create the conversational RAG chain
    Conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return Conversational_rag_chain

if __name__ == "__main__":
    # Assuming 'Data_Ingestion' provides a vector store
    data_ingestion = Data_Ingestion('done')
    vector_store = data_ingestion

    # Generate the RAG chain
    Conversational_rag_chain = Chat_Generation(vector_store)

    # Invoke the RAG chain with user input
    response = Conversational_rag_chain.invoke(
        {"input": "clear the chat history"},
        config={"configurable": {"session_id": "abc125"}}
    )

    # Print the response answer
    print(response['answer'])
