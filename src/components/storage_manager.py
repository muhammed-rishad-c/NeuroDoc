import faiss
import pickle
import os

def save_faiss_data(index,model,chunks):
    faiss.write_index(index,"faiss_index.faiss")
    
    dirname=os.path.dirname="pickle_data"
    os.makedirs(dirname,exist_ok=True)
    
    data_to_pickle={
        "model":model,
        "chunks":chunks
    }
    
    with open('pickle_data/faiss_data.pkl','wb') as f:
        pickle.dump(data_to_pickle,f)
        
    print("faiss index , model, chunks are all saved ")
    
    
def get_faiss_data(filename="pickle_data/faiss_data.pkl"):
    if not os.path.exists("faiss_index.faiss"):
        raise FileNotFoundError("faiss_index.faiss is not found please run training script at first")
    
    index=faiss.read_index("faiss_index.faiss")
    
    if not os.path.exists(filename):
        raise FileNotFoundError("file is not found faise_data.pkl save first you moron")
    
    with open(filename,'rb') as f:
        data=pickle.load(f)
        
    model=data['model']
    chunks=data['chunks']
    
    print("all data are retrieved smoothly ")
    
    return index,model,chunks
    