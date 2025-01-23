# E-commerce Recommendation System with Chat-Based Interaction

## Project Overview

This project implements a chat-based e-commerce recommendation system using LangChain and Groq. The system is designed to recommend relevant products to users based on their queries. By leveraging vector databases and a retrieval-based approach, the assistant can provide highly relevant product suggestions while adhering to politeness and ensuring the focus remains within the product catalog.

## Technologies Used

- **LangChain**: A framework to create modular and chainable NLP components.
- **Groq**: A language model for processing and generating conversational responses.
- **Vector Database**:DataStax Astra DB Used for storing and retrieving product data based on user queries.


## Key Features

- **Product Recommendation**: The assistant recommends products based on user queries by analyzing product titles and reviews.
- **Polite Alternative Suggestions**: If a requested product isn't available, the assistant suggests alternative products, politely guiding users toward similar items.
- **Context-Aware Responses**: The system incorporates chat history to provide more contextually relevant answers, ensuring the user feels heard.
- **E-commerce Focus**: Strictly adheres to the product catalog and avoids discussing topics outside the scope of eCommerce.

## Screenshots

![Screenshot 1](https://github.com/shameem11/E-Commerce-Shopping-Assistant/blob/main/Media/Screenshot%20-1.png)
![Screenshot 2](https://github.com/shameem11/E-Commerce-Shopping-Assistant/blob/main/Media/Screenshot%20-2.png)
![Screenshot 3](https://github.com/shameem11/E-Commerce-Shopping-Assistant/blob/main/Media/Screenshot-3.png)

## Installation

1.Clone the repository:

 https://github.com/shameem11/E-Commerce-Shopping-Assistant.git**
   
2.create Repo

To create a new Conda environment with Python 11, run the following command:

 **conda create --name GenerativeAI python==11 -y**
 
**3.Install dependencies**:

pip install -r requirments.txt

