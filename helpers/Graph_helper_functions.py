import networkx as nx

def build_graph(documents, corpus_embeddings, index, k_neighbors=5, similarity_threshold=0.8):
    """
    Builds a graph where nodes are document chunks and edges represent semantic similarity.

    Parameters:
    - documents: List of document objects.
    - corpus_embeddings: Tensor of embeddings corresponding to the documents.
    - index: FAISS index built from the corpus embeddings.
    - k_neighbors: Number of nearest neighbors to consider for each node.
    - similarity_threshold: Threshold for adding an edge between nodes.

    Returns:
    - G: A NetworkX graph with nodes and edges added.
    """
    G = nx.Graph()

    # Add nodes to the graph
    for i, doc in enumerate(documents):
        G.add_node(i, text=doc.page_content, metadata=doc.metadata)

    # Add edges based on semantic similarity
    for i in range(len(corpus_embeddings)):
        # For each node, find its k nearest neighbors
        D, I = index.search(corpus_embeddings[i].cpu().numpy().reshape(1, -1), k_neighbors)
        for j, dist in zip(I[0], D[0]):
            if i != j:
                # Calculate similarity
                similarity = 1 - dist
                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=similarity)

    return G
