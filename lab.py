#1. load the vector store
vector_store = InMemoryStore.load(
    "react_store.json", embeddings
)

#2 embed a question
query_vector = embeddings.embed_query(query)

#3 search by meaning
docs = vector_store.similarity_search(
    query, k=3
)

#4 compare scores
for q in [query, "best pizza in Naples"]:
    results = (
        vector_store.similarity_search_with_score(q, k=5)
    )

#filter by page
docs = vector_store.similarity_search(
    query, k=3,
    filter= lambda_doc: (
        doc.metadata.get("page") == 0
    )
)

#6 make a retriever
retriever = vector_store.as_retriever(
    search_kwargs= {"k": 3}
)
docs = retriever.invoke(query)