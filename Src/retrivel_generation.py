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
from data_ingestion import Data_Ingestion

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

    retriver_prompt = ("""You are an intelligent E-commerce assistant designed to reformulate questions. Your task is to analyze a chat history and the latest user question to determine if the question depends on prior context. Follow these steps:
    1. If the latest user question references context from the chat history, rewrite it as a standalone question by incorporating the necessary context from the history.
    2. If the latest user question is already standalone, return it without any changes.
    3. Ensure the reformulated question is concise, clear, and self-contained.
    """)

    Retriever_template = ChatPromptTemplate.from_messages(
        [("system", retriver_prompt),
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
    You are an eCommerce assistant. Your primary task is to recommend products based on user queries.
    Use the following guidelines:
    1. Analyze product titles and reviews to provide recommendations.
    2. Ensure your responses are highly relevant to the query and product context.
    3. Avoid discussing topics unrelated to the product catalog.
    4. Provide concise and informative answers about the products.
    5. If the user asks for a specific product type (e.g., 'gaming laptop'), recommend only products that match the description (e.g., gaming laptops).
    * Use follow-up questions to clarify ambiguities if needed (e.g., 'What features are you looking for in a gaming laptop?').
    * Avoid recommending unrelated products unless explicitly requested by the user.

    context:
    {context}

    input:
    {input}

    Your_Answer:
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
        config={"configurable": {"session_id": "abc124"}}
    )

    # Print the response answer
    print(response['answer'])
