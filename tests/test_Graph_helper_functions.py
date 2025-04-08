import unittest
import torch
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from helpers.Graph_helper_functions import build_graph

class TestGraphHelperFunctions(unittest.TestCase):
    def setUp(self):
        # Sample documents
        self.documents = [
            Document(page_content="This is a test document.", metadata={"id": 1}),
            Document(page_content="Another test document.", metadata={"id": 2}),
            Document(page_content="Completely different content.", metadata={"id": 3})
        ]
        self.texts = [doc.page_content for doc in self.documents]
        self.metadata = [doc.metadata for doc in self.documents]

        # Embedding model
        self.device = torch.device("cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        self.corpus_embeddings = self.embedder.encode(self.texts, convert_to_tensor=True, device=self.device)

        # FAISS index
        self.index = faiss.IndexFlatL2(self.corpus_embeddings.shape[1])
        self.index.add(self.corpus_embeddings.cpu().numpy())

    def test_build_graph(self):
        G = build_graph(self.documents, self.corpus_embeddings, self.index, k_neighbors=2, similarity_threshold=0.5)
        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(len(G.nodes), len(self.documents))
        # Verify that at least one edge is added
        self.assertGreater(len(G.edges), 0)
        # Check node attributes
        for i in G.nodes:
            self.assertIn('text', G.nodes[i])
            self.assertIn('metadata', G.nodes[i])

if __name__ == '__main__':
    unittest.main()
