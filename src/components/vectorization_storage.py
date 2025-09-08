from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def get_sentence_embedding(chunk):
    model=SentenceTransformer('all-MiniLM-L6-v2')
    embedding=model.encode(chunk,show_progress_bar=True)
    
    return model,embedding

def create_faiss_index(embeddings):
    embeddings=np.array(embeddings).astype('float32')
    d=embeddings.shape[1]
    index=faiss.IndexFlatL2(d)
    
    index.add(embeddings)
    
    return index

