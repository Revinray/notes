from .Retriever import Retriever
import networkx as nx
import numpy as np

class GraphRetriever(Retriever):
    def __init__(self, G, embedder, index, device):
        self.G = G
        self.embedder = embedder
        self.index = index
        self.device = device

    def retrieve(self, query, k=5, return_chunks=True):
        query_embedding = self.embedder.encode([query], convert_to_tensor=True, device=self.device)
        D, I = self.index.search(query_embedding.cpu().numpy(), 1)
        most_similar_node = I[0][0]

        # Perform graph traversal and keep track of the path
        traversal_nodes = []
        neighbors = nx.single_source_dijkstra_path_length(self.G, most_similar_node, cutoff=k)
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1])[:k]
        results = []
        for neighbor, _ in sorted_neighbors:
            traversal_nodes.append(neighbor)
            result = {
                'text': self.G.nodes[neighbor]['text'],
                'metadata': self.G.nodes[neighbor]['metadata']
            }
            if return_chunks:
                print(f"Chunk: {result['text']} | Metadata: {result['metadata']}")
            results.append(result)
        indices = np.array(traversal_nodes)
        return results, indices
