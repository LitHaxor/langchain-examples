from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

openai = OpenAI()
embeddings = OpenAIEmbeddings()

loader = TextLoader('resources/facts.txt')

docs = loader.load_and_split(
    text_splitter=text_splitter
)


# Semeantic Search
## Embeddings

# emb = embeddings.embed_query("Hi there?")

# Vector store
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)


# for doc in docs:
#     print(doc.page_content)
#     print('\n')

results = db.similarity_search_with_score("what is an interesting fact about english?", k=5)


for result in results:
    print("\n")
    print("score: ",result[1])
    print(result[0].page_content)