from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text = "My name is Prastuti Adhikari"
#same process for documents embedding as well
vector = embedding.embed_query(text)
print(str(vector))
