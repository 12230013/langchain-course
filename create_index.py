import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]

    if pc.has_index(index_name):
        raise RuntimeError(
            f"Index '{index_name}' already exists. Choose a new INDEX_NAME in .env "
            "to create a standard vector index without a document schema."
        )

    pc.create_index(
        name= index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine", #how similar two vectors are
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created")
