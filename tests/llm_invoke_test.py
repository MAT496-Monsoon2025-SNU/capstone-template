
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
print("OPENAI key present:", bool(os.getenv("OPENAI_API_KEY")))
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
resp = llm.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Say hello in one word.")])
print("LLM resp type:", type(resp))
print("LLM resp content (first 200 chars):", getattr(resp,"content",str(resp))[:200])
