import pandas as pd 
import numpy as np 
import os 
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_astradb import AstraDBVectorStore
from data_conveter import Data_Converter



ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRADB_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRADB_KEYSPACE')
Hugging_API = os.getenv('Hugging_Face_API')


embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=Hugging_API,model_name="sentence-transformers/all-mpnet-base-v2")


def Data_Ingestion(Status):
    # Initialize the vector store
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name='Amazon_RAG',
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE
    )

    storage = Status

    # Check if storage is None
    if storage is None:
        # Convert data 
        Docs = Data_Converter()
        # Add documents to the vector store
        Interst_To_DB = vector_store.add_documents(Docs)
        return vector_store, Interst_To_DB
    else:
        # If storage is not None, return the vector store
        return vector_store
    


   

