from .Retriever import Retriever
import networkx as nx
import numpy as np

class CombinedRetriever(Retriever):
    def __init__(self, vector_retriever, graph_retriever, G):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.G = G

    def retrieve(self, query, k=5, return_chunks=True):
        # First retrieve using vector retriever
        vector_results, vector_indices = self.vector_retriever.retrieve(query, k=k, return_chunks=False)
        # Map retrieved texts to node indices
        text_to_node = {self.G.nodes[i]['text']: i for i in self.G.nodes}
        node_indices = [text_to_node.get(res['text']) for res in vector_results if res['text'] in text_to_node]
        combined_results = []
        combined_indices = []

        # Add vector results to combined_results and combined_indices
        for res, idx in zip(vector_results, vector_indices):
            combined_results.append(res)
            combined_indices.append(idx)

        # Use node indices to get graph neighbors
        for node_index in node_indices:
            if node_index is not None:
                neighbors = nx.single_source_dijkstra_path_length(self.graph_retriever.G, node_index, cutoff=k)
                for neighbor in neighbors:
                    neighbor_text = self.G.nodes[neighbor]['text']
                    neighbor_metadata = self.G.nodes[neighbor]['metadata']
                    # Check if neighbor_text is already in combined_results
                    if not any(res['text'] == neighbor_text for res in combined_results):
                        combined_results.append({
                            'text': neighbor_text,
                            'metadata': neighbor_metadata
                        })
                        combined_indices.append(neighbor)
        if return_chunks:
            for res in combined_results:
                print(f"Chunk: {res['text']} | Metadata: {res['metadata']}")
        indices = np.array(combined_indices)
        results = combined_results
        return results, indices
