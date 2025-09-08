from src.components.text_extract import extract_text_from_pdf,chuck_text_by_paragraph
from src.components.vectorization_storage import create_faiss_index,get_sentence_embedding


if __name__=="__main__":
    text=extract_text_from_pdf("data/TheLittlePrince.pdf")
    #print(text)
    chunks=chuck_text_by_paragraph(text)
    model,embedding=get_sentence_embedding(chunks)
    
    #print(embedding)
    index=create_faiss_index(embedding)
    print(index)
    
    