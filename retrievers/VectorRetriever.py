from .Retriever import Retriever
import numpy as np

class VectorRetriever(Retriever):
    def __init__(self, texts, metadata, index, embedder, device):
        self.texts = texts
        self.metadata = metadata
        self.index = index
        self.embedder = embedder
        self.device = device

    def retrieve(self, query, k=5, return_chunks=True):
        query_embedding = self.embedder.encode([query], convert_to_tensor=True, device=self.device)
        D, I = self.index.search(query_embedding.cpu().numpy(), k)
        indices = I[0]
        results = []
        for i in indices:
            result = {
                'text': self.texts[i],
                'metadata': self.metadata[i]
            }
            if return_chunks:
                print(f"Chunk: {result['text']} | Metadata: {result['metadata']}")
            results.append(result)
        return results, indices
