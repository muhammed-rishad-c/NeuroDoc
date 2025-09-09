import fitz
import re

def extract_text_from_pdf(pdfpath):
    doc=fitz.open(pdfpath)
    text=""
    
    for page in doc:
        text+=page.get_text()
        
    doc.close()
    return text



def chunk_text_by_paragraph(text, chunk_size=500, chunk_overlap=50):
    """
    Splits a long text string into fixed-size chunks with overlap.
    A more robust approach for RAG pipelines.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks