
from src.components.text_extract import extract_text_from_pdf, chunk_text_by_sentences  # Use improved chunking
from src.components.vectorization_storage import create_optimized_faiss_index, get_sentence_embeddings_batch  # Improved functions
from src.components.storage_manager import save_faiss_data
import time
import os

def validate_chunks(chunks):
    valid_chunks = []
    for i, chunk in enumerate(chunks):
        if chunk and chunk.strip():
            cleaned_chunk = chunk.strip()
            if len(cleaned_chunk) > 20: 
                valid_chunks.append(cleaned_chunk)
        else:
            print(f"Warning: Skipping empty chunk at index {i}")
    
    return valid_chunks



if __name__ == "__main__":
    print("Starting RAG Training Pipeline...")
    start_time = time.time()
    
    pdf_path = "data/KTU S7 Mod 2 Artificial Intelligence PDF Notes.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at {pdf_path}")
        print("Please check the file path and try again.")
        exit(1)
    
    print(f"Extracting text from: {pdf_path}")
    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"✓ Text extracted successfully. Length: {len(text)} characters")
    except Exception as e:
        print(f"ERROR: Failed to extract text from PDF: {str(e)}")
        exit(1)
    
    print("\nChunking text...")
    try:
        chunks = chunk_text_by_sentences(text, max_chunk_size=800, overlap_sentences=2)
        
        
        chunks = validate_chunks(chunks)
        print(f"✓ Created {len(chunks)} valid chunks")
        
        if len(chunks) == 0:
            print("ERROR: No valid chunks created. Check your PDF content.")
            exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed to chunk text: {str(e)}")
        exit(1)
    
    print("\nCreating embeddings...")
    try:
        model, embeddings = get_sentence_embeddings_batch(
            chunks, 
            model_name='all-MiniLM-L6-v2',  # You can change this to a better model
            batch_size=32
        )
        print(f"✓ Embeddings created. Shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"ERROR: Failed to create embeddings: {str(e)}")
        exit(1)
    
    print("\nCreating FAISS index...")
    try:
        index = create_optimized_faiss_index(embeddings, similarity_type='cosine')
        print(f"✓ FAISS index created with {index.ntotal} vectors")
        
    except Exception as e:
        print(f"ERROR: Failed to create FAISS index: {str(e)}")
        exit(1)
    
    print("\nSaving model and data...")
    try:
        save_faiss_data(index=index, model=model, chunks=chunks)
        print("✓ Model and data saved successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to save model and data: {str(e)}")
        exit(1)
    
    total_time = time.time() - start_time
   
    print(f"\n🎉 Training completed successfully in {total_time:.2f} seconds!")
    print("\nYou can now run your query script to test the RAG system.")
    
    