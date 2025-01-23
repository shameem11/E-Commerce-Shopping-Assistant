import pandas as pd 
import numpy as np 
import re

from langchain_core.documents import Document

def Data_Converter():
    Data = pd.read_csv('/home/mshameem/Documents/Generative_AI /Data/flipkart_reviews_dataset.csv')

    Data_set = Data[['product_title','review']]


    Review_list = []

    for index, row in Data_set.iterrows():
    #   Clean the review text by removing special characters
        cleaned_review = re.sub(r'[^A-Za-z0-9\s]', '', row['review'])
        cleaned_review1 = re.sub(r'[^A-Za-z0-9\s]', '', row['product_title'])


     # Remove the specific character 'e'
        cleaned_review = re.sub(r'e', '', cleaned_review)
        cleaned_review2 = re.sub(r'e', '', cleaned_review1)

        cleaned_review = re.sub(r'\s+', ' ', cleaned_review)  # Replace multiple spaces with a single space
        cleaned_review = cleaned_review.strip()

        object = {
        'product_name': row['product_title'],
        'Review': cleaned_review2
     }

        Review_list.append(object)  # Append to Review list


        Docs = []

    for Review in Review_list:
       metadata = {'product_name':Review['product_name']}
       page_content = Review['Review']

       doc = Document(page_content=page_content,metadata=metadata)
       Docs.append(doc)



    