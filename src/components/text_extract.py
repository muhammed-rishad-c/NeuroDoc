import fitz
import re

def extract_text_from_pdf(pdfpath):
    doc=fitz.open(pdfpath)
    text=""
    
    for page in doc:
        text+=page.get_text()
        
    doc.close()
    return text

def chuck_text_by_paragraph(text):
    normalized_text=re.sub(r'\n','[PARAGRAPH_BREAK]',text)
    paragraph=normalized_text.split('[PARAGRAPH_BREAK]')
    
    return [p.strip() for p in paragraph if p.strip()]