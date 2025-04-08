# OLDER VERSION

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import time
import warnings
import torch
import numpy as np
from datetime import datetime

import streamlit as st
from gpt4all import GPT4All

from helpers.RAG_helper_functions import get_retriever
from helpers.GraphVisualization_helper_functions import visualize_graph
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

config = load_config()

def load_preprocessed_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed_data.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

data = load_preprocessed_data()
texts = data["texts"]
metadata = data["metadata"]
corpus_embeddings = data["corpus_embeddings"]
index = data["index"]
G = data["G"]

device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

model_name = config["embedding_model"]
model_name = os.path.join(os.path.dirname(__file__), "..", model_name)
embedder = SentenceTransformer(model_name).to(device)

# Load GPT4All model
text_generation_model_path = os.path.join(os.path.dirname(__file__), "..", config["text_generation_model"])
generator = GPT4All(model_name=text_generation_model_path,
                    allow_download=False,
                    n_ctx=config.get("max_input_length", 512))

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def generate_response(query, use_conversation_history=False):
    retriever = get_retriever(
        config["retrieval_method"], texts, metadata, index, embedder, device, G=G
    )
    retrieved_docs, indices = retriever.retrieve(query, k=10, return_chunks=False)

    if isinstance(indices, (list, np.ndarray)):
        indices = torch.tensor(indices).to(device)
    else:
        indices = indices.to(device)

    if len(retrieved_docs) != len(indices):
        st.error("Mismatch between retrieved documents and indices lengths.")
        return

    query_embedding = embedder.encode([query], convert_to_tensor=True, device=device)
    retrieved_embeddings = corpus_embeddings[indices].to(device)
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding, retrieved_embeddings
    )
    sorted_indices = torch.argsort(similarities, descending=True)
    sorted_indices_list = sorted_indices.cpu().numpy().tolist()

    retrieved_docs = [retrieved_docs[i] for i in sorted_indices_list]
    indices = indices[sorted_indices]

    top_k = min(config.get("top_k", 3), len(retrieved_docs))
    retrieved_docs = retrieved_docs[:top_k]
    indices = indices[:top_k]

    input_texts = [doc['text'] for doc in retrieved_docs]
    context_text = "\n".join(input_texts)

    prompt_template = (
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: {today_date}\n\n"
        "{system_prompt}\n\n"
        "{question}\n"
    )

    today_date = datetime.now().strftime("%d %b %Y")
    system_prompt = (
        "You are an assistant that provides concise and accurate answers based on the provided context.\n"
        "Context:\n{context}"
    ).format(context=context_text)

    prompt = prompt_template.format(
        today_date=today_date,
        system_prompt=system_prompt,
        question=query
    )

    response = generator.generate(prompt, max_tokens=config.get("max_output_length", 256))

    if len(retrieved_docs) == 0:
        return

    return indices, response, retrieved_docs

st.title("Document Retrieval and Generation")

retrieval_method = st.selectbox(
    "Select retrieval method",
    ["vector", "graph", "combined"],
    index=["vector", "graph", "combined"].index(config.get("retrieval_method", "vector"))
)

if 'retrieval_method' not in st.session_state or st.session_state.retrieval_method != retrieval_method:
    st.session_state.retrieval_method = retrieval_method
    with st.spinner("Updating retrieval method..."):
        pass

for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Enter your query"):
    st.session_state.conversation.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.spinner("Searching for information..."):
        result = generate_response(query)
        if result:
            indices, response, retrieved_docs = result
            traversal_nodes = [int(i) for i in indices.cpu().numpy().tolist()]
    
            with st.chat_message("model"):
                full_response = "".join(response_generator(response))
                st.markdown(full_response)
    
            st.session_state.conversation.append({
                "role": "model",
                "content": full_response,
                "retrieved_docs": retrieved_docs,
                "traversal_nodes": traversal_nodes if config.get("retrieval_method") in ['graph', 'combined'] else None
            })
    
            if config.get("retrieval_method") in ['graph', 'combined']:
                visualize_graph(G, traversal_nodes)
    
                with st.expander("Show Retrieved Chunks and Metadata"):
                    for doc in retrieved_docs:
                        doc_markdown = f"""
                                        ### Document {doc.get('id', 'N/A')}
                                        **Metadata:**
                                        - **Source:** {doc.get('source', 'N/A')}
                                        - **Page:** {doc.get('page', 'N/A')}

                                        **Content:**
                                        {doc.get('text', '')}
                                        """
                        st.markdown(doc_markdown)