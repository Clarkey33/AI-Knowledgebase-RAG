def map2embeddings(data, embedding_model):
    """Map a list of texts to their embeddings using the provided embedding model"""
    
    # Initialize an empty list to store embeddings
    embeddings = []

    # Iterate over each text in the input data list
    no_texts = len(data)
    print(f"Mapping {no_texts} pieces of information")
    for i in tqdm(range(no_texts)):
        # Get embeddings for the current text using the provided embedding model
        embeddings.append(get_embedding(data[i], embedding_model))
    
    # Return the list of embeddings
    return embeddings