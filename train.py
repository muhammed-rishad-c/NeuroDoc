# IMPROVED train.py following your exact structure

from src.components.text_extract import extract_text_from_pdf, chunk_text_by_sentences  # Use improved chunking
from src.components.vectorization_storage import create_optimized_faiss_index, get_sentence_embeddings_batch  # Improved functions
from src.components.storage_manager import save_faiss_data
import time
import os

def validate_chunks(chunks):
    """Validate and clean chunks before processing"""
    valid_chunks = []
    for i, chunk in enumerate(chunks):
        if chunk and chunk.strip():
            # Clean the chunk
            cleaned_chunk = chunk.strip()
            if len(cleaned_chunk) > 20:  # Minimum meaningful length
                valid_chunks.append(cleaned_chunk)
        else:
            print(f"Warning: Skipping empty chunk at index {i}")
    
    return valid_chunks

def print_training_stats(chunks, embeddings, index):
    """Print useful statistics about the training data"""
    print("\n" + "="*50)
    print("TRAINING STATISTICS")
    print("="*50)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1] if embeddings.size > 0 else 0}")
    print(f"Total vectors in index: {index.ntotal}")
    
    # Chunk statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    word_counts = [len(chunk.split()) for chunk in chunks]
    
    print(f"Average chunk length: {sum(chunk_lengths) / len(chunks):.0f} characters")
    print(f"Average words per chunk: {sum(word_counts) / len(chunks):.0f} words")
    print(f"Shortest chunk: {min(chunk_lengths)} characters")
    print(f"Longest chunk: {max(chunk_lengths)} characters")
    
    print("\nFirst 3 chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i+1}: {chunk[:100]}...")
    print("="*50)

if __name__ == "__main__":
    print("Starting RAG Training Pipeline...")
    start_time = time.time()
    
    # Step 1: Extract text from PDF
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
    
    # Step 2: Chunk the text (IMPROVED)
    print("\nChunking text...")
    try:
        # Use improved sentence-based chunking (you'll need to add this to text_extract.py)
        chunks = chunk_text_by_sentences(text, max_chunk_size=800, overlap_sentences=2)
        
        # Fallback to your original method if the above doesn't exist yet
        # chunks = chunk_text_by_paragraph(text, chunk_size=1000, chunk_overlap=200)
        
        # Validate and clean chunks
        chunks = validate_chunks(chunks)
        print(f"✓ Created {len(chunks)} valid chunks")
        
        if len(chunks) == 0:
            print("ERROR: No valid chunks created. Check your PDF content.")
            exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed to chunk text: {str(e)}")
        exit(1)
    
    # Step 3: Create embeddings (IMPROVED)
    print("\nCreating embeddings...")
    try:
        # Use improved batch processing
        model, embeddings = get_sentence_embeddings_batch(
            chunks, 
            model_name='all-MiniLM-L6-v2',  # You can change this to a better model
            batch_size=32
        )
        print(f"✓ Embeddings created. Shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"ERROR: Failed to create embeddings: {str(e)}")
        exit(1)
    
    # Step 4: Create FAISS index (IMPROVED)
    print("\nCreating FAISS index...")
    try:
        # Use improved index with cosine similarity
        index = create_optimized_faiss_index(embeddings, similarity_type='cosine')
        print(f"✓ FAISS index created with {index.ntotal} vectors")
        
    except Exception as e:
        print(f"ERROR: Failed to create FAISS index: {str(e)}")
        exit(1)
    
    # Step 5: Save the trained model and data
    print("\nSaving model and data...")
    try:
        save_faiss_data(index=index, model=model, chunks=chunks)
        print("✓ Model and data saved successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to save model and data: {str(e)}")
        exit(1)
    
    # Final statistics and timing
    total_time = time.time() - start_time
    print_training_stats(chunks, embeddings, index)
    
    print(f"\n🎉 Training completed successfully in {total_time:.2f} seconds!")
    print("\nYou can now run your query script to test the RAG system.")
    
    