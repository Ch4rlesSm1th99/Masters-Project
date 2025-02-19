import torch
from models import SimCLR, BaseEncoder
from torch.utils.data import DataLoader
from data_loader import SuperDARNDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

dataset = SuperDARNDataset(
    h5_file_path=r"C:\Users\aman\Desktop\MPhys Data\Data\test.h5",  
    negative_value=-9999,
    apply_augmentations=False  
)

def collate_fn(batch):
    data_list = [item[0] for item in batch if item[0] is not None]
    if len(data_list) == 0:
        raise ValueError("No valid data found in the batch.")
    return torch.stack(data_list, dim=0)

data_loader = DataLoader(
    dataset,
    batch_size=500,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)

# loading model weights
def load_model(path, device="cuda"):
    base_encoder = BaseEncoder(input_channels=1)
    model = SimCLR(base_encoder, projection_dim=128, temperature=0.5, device=device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device)
    model.eval()
    return model

def get_embeddings(model, data_loader, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            print("Got batch of size:", batch.size())
            data = batch.unsqueeze(1)
            data = data.to(device)
            emb = model.encoder(data)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def visualize_neighbours(embs, dataset, segment_names, k=3):
    negative_value = -9999

    #calculating cosine similarity matrix
    embs = F.normalize(embs, p=2, dim=1)
    similarity_matrix = torch.mm(embs, embs.T)

    N = embs.shape[0]

    #select some anchors and visualise their neighbours  
    num_anchors = min(5, N)  
    anchor_indices = np.random.choice(N, num_anchors, replace=False)

    for anchor_idx in anchor_indices:
        #gets similarity scores for the anchor, making the anchor -9999
        sim_scores = similarity_matrix[anchor_idx].clone()
        sim_scores[anchor_idx] = -9999

        #sorts the similarity scores in descending order and returns the indices
        top_k_indices = torch.argsort(sim_scores, descending=True)[:k]

         #extracts the power values for the anchor from the dataset
        anchor_data, *_ = dataset[anchor_idx]
        anchor_data = anchor_data.numpy()

        #gets vmax and vmin for plotting 
        valid_mask = (anchor_data != negative_value)
        vmin, vmax = anchor_data[valid_mask].min(), anchor_data[valid_mask].max()

        #plotting the anchor
        fig, axs = plt.subplots(1, k + 1, figsize=(15, 5), sharey=True)

       
        im = axs[0].imshow(
            np.ma.masked_where(anchor_data == negative_value, anchor_data).T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )
        axs[0].set_title(f"Anchor: {segment_names[anchor_idx]}")
        axs[0].set_xlabel("Time Steps")
        axs[0].set_ylabel("Magnetic Latitude")
        fig.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)

        #plotting neighbours
        for j, nn_idx in enumerate(top_k_indices):
            neighbor_data, *_ = dataset[nn_idx]
            neighbor_data = neighbor_data.numpy()

            im2 = axs[j + 1].imshow(
                np.ma.masked_where(neighbor_data == negative_value, neighbor_data).T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )
            axs[j + 1].set_title(f"Neighbour {j+1}: {segment_names[nn_idx]}")
            axs[j + 1].set_xlabel("Time Steps")
            fig.colorbar(im2, ax=axs[j + 1], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = r"C:\Users\aman\OneDrive - University of Southampton\Desktop\Year 4\MPhys Project\Lo\Masters-Project\aman\SimCLR\best_model.pth"
    model = load_model(model_path, device)

    embeddings = get_embeddings(model, data_loader, device)
    print("Embeddings shape:", embeddings.shape)
    segment_names = list(range(len(embeddings)))
    visualize_neighbours(embeddings, dataset, segment_names, k=3)

if __name__ == "__main__":
    main()
