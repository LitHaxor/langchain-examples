from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI

import langchain
langchain.debug = True

load_dotenv()

llm = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

retriever = db.as_retriever()


chain = RetrievalQA.from_chain_type(
    llm= llm,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.invoke("what is an interesting fact about english?") 

print(result)