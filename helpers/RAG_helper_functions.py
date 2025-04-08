from retrievers.VectorRetriever import VectorRetriever
from retrievers.GraphRetriever import GraphRetriever
from retrievers.CombinedRetriever import CombinedRetriever


def get_retriever(method, texts, metadata, index, embedder, device, G=None):
    if method == 'vector':
        return VectorRetriever(texts, metadata, index, embedder, device)
    elif method == 'graph':
        return GraphRetriever(G, embedder, index, device)
    elif method == 'combined':
        vector_retriever = VectorRetriever(texts, metadata, index, embedder, device)
        graph_retriever = GraphRetriever(G, embedder, index, device)
        return CombinedRetriever(vector_retriever, graph_retriever, G)
    else:
        raise ValueError("Unknown retrieval method specified.")
