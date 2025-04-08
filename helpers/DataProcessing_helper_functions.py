from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd

# Function to generate metadata
def generate_metadata(row):
    metadata = row.drop('content').to_dict()
    return metadata

# Function to split DataFrame into chunks
def split_dataframe(dataframe, chunk_size=200, chunk_overlap=20):
    """
    Splits the dataframe into smaller chunks using the specified chunk size and overlap.

    Parameters:
    - dataframe: The pandas DataFrame containing the data.
    - chunk_size: The size of each chunk.
    - chunk_overlap: The number of overlapping characters between chunks.

    Returns:
    - A pandas DataFrame with the split chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = [
        Document(page_content=row['content'], metadata=generate_metadata(row))
        for _, row in dataframe.iterrows()
    ]
    chunks = text_splitter.split_documents(documents)
    chunk_data = [
        {
            "text": chunk.page_content,
            **chunk.metadata
        }
        for chunk in chunks
    ]
    return pd.DataFrame(chunk_data)
