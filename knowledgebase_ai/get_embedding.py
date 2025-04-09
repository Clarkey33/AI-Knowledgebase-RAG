def get_embedding(text, embedding_model):
    """Get embeddings for a given text using the provided embedding model"""
    
    # Encode the text to obtain embeddings using the provided embedding model
    embedding = embedding_model.encode(text, show_progress_bar=False)
    
    # Convert the embeddings to a list of floats and return
    return embedding.tolist()