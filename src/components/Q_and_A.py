from transformers import pipeline
import numpy as np
from src.components.storage_manager import get_faiss_data
import re

# The model and index will be loaded once when this module is imported
print("Loading FAISS index and model from disk...")
try:
    index, model, chunks = get_faiss_data()
except FileNotFoundError as e:
    print(e)
    index, model, chunks = None, None, None

def find_similar_chunks(query, k=5, min_score=0.3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k*2)

    valid_indices = []
    for i, distance in enumerate(distances[0]):
        if distance >= min_score: 
            valid_indices.append(indices[0][i])
    
    return np.array([valid_indices[:k]])

def generating_answer(query, indices):
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = " ".join([chunk for chunk in relevant_chunks if chunk.strip()])
 
    qa_pipeline = pipeline("question-answering", 
                          model="microsoft/DialoGPT-medium") 
   
    result = qa_pipeline(question=query, context=context[:4000])  
    return result['answer']

