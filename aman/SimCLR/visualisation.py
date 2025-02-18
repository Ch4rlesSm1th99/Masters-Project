
import torch
from models import SimCLR, BaseEncoder

#loading model weights:
def load_model(path):
    #instantiating base encoder and model
    base_encoder = BaseEncoder(input_channels=1)
    model  = SimCLR(base_encoder, projection_dim=128, temperature=0.5, device='cpu')
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_embeddings(model,)





def compute_similarity()



model_path = r"C:\Users\aman\OneDrive - University of Southampton\Desktop\Year 4\MPhys Project\Lo\Masters-Project\aman\SimCLR\best_model.pth"
load_model(model_path)
