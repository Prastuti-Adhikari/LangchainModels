from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
documents = [
    "My name is Prastuti Adhikari!",
    "One day I will be the best programmer in the world!",
    "Who am I without my aspirations?"
    ]
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimension=32)
result = embedding.embed_documents(documents)
print(str(result))