from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
documents = [
    "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities such as burning fossil fuels and deforestation.",
    "Nepal is known for its rich culture and natural beauty, attracting tourists with destinations like Pokhara, Lumbini, and the Himalayan trekking routes.",
    "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data and make predictions without being explicitly programmed.",
    "Digital health technologies, including telemedicine and wearable devices, are transforming modern healthcare by improving accessibility and early diagnosis.",
    "Prastuti considers herself as someone who has big aspirations and wants to improve herself everyday. Someone who is passionate about literature, technology and travel. She believes in connection and believes that she can contribute towards creating an interconnected world through her passion."
]

query = "Tell me about Prastuti!"
documents_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], documents_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query:", query)
print("Best Match:", documents[index])
print("Similarity Score:", score)


'''
#for/using OpenAI Embedding Model
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimension = 300)
documents = [
    "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities such as burning fossil fuels and deforestation."
    "Nepal is known for its rich culture and natural beauty, attracting tourists with destinations like Pokhara, Lumbini, and the Himalayan trekking routes."
    "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data and make predictions without being explicitly programmed."
    "Digital health technologies, including telemedicine and wearable devices, are transforming modern healthcare by improving accessibility and early diagnosis."
    "Prastuti considers herself as someone who has big aspirations and wants to improve herself everyday. Someone who is passionate about literature, technology and travel. She believes in connection and believes that she can contribute towards creating interconnected world through her passion."
]

query = "Tell me about Prastuti!"
documents_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], documents_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score:", score)
'''