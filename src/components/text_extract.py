
import fitz
import re
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """IMPROVED: Better PDF text extraction with cleaning"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            
     
            cleaned_page_text = clean_pdf_text(page_text)
            
            text += f"\n\n--- Page {page_num + 1} ---\n{cleaned_page_text}"
            
        doc.close()
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def clean_pdf_text(text):
    """Clean common PDF text extraction artifacts"""
    if not text:
        return ""
    

    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    

    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text) 

    text = re.sub(r'[^\w\s.,!?;:()\-"\'%$]', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def chunk_text_by_paragraph(text, chunk_size=1000, chunk_overlap=200):
    """IMPROVED: Your original function with better parameters"""
    if not text.strip():
        return []
    
    text = re.sub(r'\s+', ' ', text.strip())
    
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def chunk_text_by_sentences(text, max_chunk_size=800, overlap_sentences=2):
    """NEW: Better chunking using sentence boundaries"""
    if not text.strip():
        return []
    
    sentences = sent_tokenize(text)
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = ""
    current_sentences = []
    
    for sentence in sentences:
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(potential_chunk) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            if len(current_sentences) > overlap_sentences:
                overlap_start = len(current_sentences) - overlap_sentences
                overlapping_sentences = current_sentences[overlap_start:]
                current_chunk = " ".join(overlapping_sentences) + " " + sentence
                current_sentences = overlapping_sentences + [sentence]
            else:
                current_chunk = sentence
                current_sentences = [sentence]
        else:
            current_chunk = potential_chunk
            current_sentences.append(sentence)
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks