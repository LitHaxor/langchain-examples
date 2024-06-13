from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriver(BaseRetriever):
    """
    This class is a redundant filter retriever that uses the Chroma vector store

    Args:
        BaseRetriever ([type]): [description]
        chroma ([type]): [description]
    """
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate embeddings for the query
        emb = self.embeddings.embed_query(query)
        # take the embedding and feed them into the vector store
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        return []