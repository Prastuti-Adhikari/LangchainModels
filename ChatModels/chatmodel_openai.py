#for/from openAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model='gpt-4', temperature=0.2, max_completion_tokens=10)
result = model.invoke("What is the capital city of Nepal?")
print(result.content)

