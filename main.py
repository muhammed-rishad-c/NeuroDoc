from src.components.text_extract import extract_text_from_pdf,chuck_text_by_paragraph


if __name__=="__main__":
    text=extract_text_from_pdf("data/TheLittlePrince.pdf")
    #print(text)
    chucks=chuck_text_by_paragraph(text)
    
    for c in chucks:
        print(c)