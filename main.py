from src.components.storage_manager import get_faiss_data
from src.components.Q_and_A import find_similar_chunks,generating_answer


if __name__=="__main__":
    
    
    
    query="What is breadth first search"
    
    indices=find_similar_chunks(query=query)
    answer=generating_answer(query=query,indices=indices)
    print(f"this is answer ",answer)
    
    
    
    
    