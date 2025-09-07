import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# File paths for saving/loading embeddings and index
EMBEDDINGS_PATH = "data/embeddings.pkl"
INDEX_PATH = "data/faiss_index.bin"

def save_embeddings_and_index(embeddings, texts, index):
    """Save embeddings and texts to a pickle file and FAISS index to a binary file."""
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump({'texts': texts, 'embeddings': embeddings}, f)
    faiss.write_index(index, INDEX_PATH)

def load_embeddings_and_index():
    """Load embeddings, texts, and FAISS index from files."""
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
        index = faiss.read_index(INDEX_PATH)
        return data['texts'], data['embeddings'], index
    return None, None, None

def main():
    print("\n\n=========== Vector Search using FAISS ===========\n")

    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Try to load existing embeddings and index
    texts, embeddings, index = load_embeddings_and_index()

    # If embeddings or index don't exist, compute and save them
    if texts is None or embeddings is None or index is None:
        texts = [
            # People & Family
            "kid", "parent", "mother", "father", "child", "son", "daughter", "aunt", "uncle", "grandparent",
            "brother", "sister", "family", "friend", "cousin", "infant", "toddler",
            
            # Vehicles & Transportation
            "car", "bus", "bicycle", "airplane", "motorcycle", "train", "truck", "van", "scooter",
            
            # Car Brands (expanded)
            "nissan", "toyota", "honda", "ford", "chevrolet", "bmw", "mercedes", "audi", "tesla",
            
            # Generic Vehicle Terms
            "vehicle", "automobile", "transportation",
            
            # Names (expanded)
            "Habib", "Amiya", "Riya", "Ankita", "Sarah", "John", "Maria", "David",
            
            # Concepts
            "relationship", "bond", "journey", "travel", "engine", "wheel", "road"
        ]
        embeddings = model.encode(texts)
        
        # Normalize embeddings to use with L2 for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dimension = embeddings.shape[1]
        
        # Use a flat L2 index, which now effectively performs a cosine search
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save the normalized embeddings and the new index
        save_embeddings_and_index(embeddings, texts, index)
        print("\nComputed and saved embeddings and index.")
    else:
        print("\nLoaded embeddings and index from disk.")


    print("\nData: ", texts)

    while True:
        user_input = input("\nEnter a query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Query search
        query_embedding = model.encode(user_input)
        
        # Normalize the query embedding as well
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)

        print("\nQuery:", user_input)
        print("\nClosest matches:", [texts[i] for i in I[0]])
        # The distances (D) are now L2 distances, not cosine similarity scores
        # You can convert them back to cosine scores if needed: score = 1 - D^2 / 2
        print("Distances (L2):", D[0])
        

if __name__ == "__main__":
    main()