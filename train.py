from src.components.text_extract import extract_text_from_pdf,chunk_text_by_paragraph
from src.components.vectorization_storage import create_faiss_index,get_sentence_embedding
from src.components.storage_manager import save_faiss_data

if __name__=="__main__":
    text=extract_text_from_pdf("data/KTU S7 Mod 2 Artificial Intelligence PDF Notes.pdf")
    #print(text)
    chunks=chunk_text_by_paragraph(text)
    model,embedding=get_sentence_embedding(chunks)
    
    print(chunks,model)
    index=create_faiss_index(embedding)
    print(index)
    save_faiss_data(index=index,model=model,chunks=chunks)