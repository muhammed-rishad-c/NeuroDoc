# src/components/vectorization_storage.py - ADD THESE IMPROVEMENTS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple

# KEEP your original function for backward compatibility
def get_sentence_embedding(chunk):
    """ORIGINAL: Your existing function (kept for compatibility)"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(chunk, show_progress_bar=True)
    return model, embedding

# KEEP your original function for backward compatibility  
def create_faiss_index(embeddings):
    """ORIGINAL: Your existing function (kept for compatibility)"""
    embeddings = np.array(embeddings).astype('float32')
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# ADD these improved functions
def get_sentence_embeddings_batch(chunks: List[str], model_name: str = 'all-MiniLM-L6-v2', 
                                 batch_size: int = 32) -> Tuple[SentenceTransformer, np.ndarray]:
    
    if not chunks:
        return None, np.array([])
    
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
   
    valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
    if len(valid_chunks) != len(chunks):
        print(f"Warning: Filtered out {len(chunks) - len(valid_chunks)} empty chunks")
    
    if not valid_chunks:
        return model, np.array([])
    
    print(f"Creating embeddings for {len(valid_chunks)} chunks...")
    embeddings = model.encode(
        valid_chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT: Better similarity search
    )
    
    embeddings = embeddings.astype('float32')
    print(f"Embeddings created: shape {embeddings.shape}")
    
    return model, embeddings

def create_optimized_faiss_index(embeddings: np.ndarray, similarity_type: str = 'cosine') -> faiss.Index:
    
    if embeddings.size == 0:
        raise ValueError("Empty embeddings array")
    
    embeddings = np.array(embeddings).astype('float32')
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    
    d = embeddings.shape[1]
    n = embeddings.shape[0]
    
    print(f"Creating FAISS index for {n} vectors of dimension {d}")
    
    if similarity_type == 'cosine':
        
        index = faiss.IndexFlatIP(d)  
        faiss.normalize_L2(embeddings)
        print("Using cosine similarity (Inner Product with normalization)")
    else:
       
        index = faiss.IndexFlatL2(d)
        print("Using Euclidean distance (L2)")
    
    index.add(embeddings)
    
    print(f"Index created with {index.ntotal} vectors")
    return index


