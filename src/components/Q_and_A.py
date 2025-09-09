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

def find_similar_chunks(query, k=5):
    """
    Finds the top k most similar chunks to a given query using the loaded index.
    """
    if model is None or index is None:
        raise RuntimeError("FAISS index or model not loaded. Please run the training script first.")
        
    query_embedding = model.encode([query]).astype('float32')
    
    distances, indices = index.search(query_embedding, k)
    
    return indices

def generating_answer(query, indices):
    """
    Generates an answer from a query and relevant text chunks.
    """
    if chunks is None:
        raise RuntimeError("Text chunks not loaded. Please run the training script first.")
        
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    # Filter out empty chunks before joining
    context = " ".join([chunk for chunk in relevant_chunks if chunk.strip()])
    
    # Use a more robust QA model for better performance
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    # The pipeline returns a dictionary with the answer
    # This is the corrected line where the question and context are passed
    result = qa_pipeline(question=query, context=context)
    
    return result['answer']

