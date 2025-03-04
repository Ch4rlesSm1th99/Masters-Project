import os
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader

from charles.self_supervised_approach.models import BaseEncoder
from charles.self_supervised_approach.data_loader import SuperDARNDataset, contrastive_collate_fn


def load_model(path, device="cpu"):
    base_encoder = BaseEncoder(input_channels=1)
    checkpoint = torch.load(path, map_location=device)

    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }

    base_encoder.load_state_dict(encoder_state_dict)
    base_encoder.to(device)
    base_encoder.eval()
    return base_encoder


def get_embeddings(model, data_loader, dataset_type="test", device="cpu"):
    """
    Computes embeddings of all samples in dataloader if they are unsaved otherwise it loads them from directory
    """
    save_dir = r"/charles/data/embeddings"
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise ValueError(f"{save_dir} exists but is not a directory.")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{dataset_type}.npy")

    # load embeddings if available
    if os.path.exists(save_path):
        embs = np.load(save_path)
        return embs

    # else, compute embeddings
    embeddings = []
    with torch.no_grad():
        for batch_tuple in data_loader:
            batch, *_ = batch_tuple  # NOTE: NO AUGS HERE
            batch = batch.unsqueeze(1).to(device)  # shape: [batch_size, 1, H, W]
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    embs = np.vstack(embeddings)
    np.save(save_path, embs)
    print(f"Extracted {embs.shape[0]} {dataset_type} embeddings and saved to {save_path}")
    return embs


def compute_uniformity_metric(embeddings, sample_size=1000):
    """
    Computes the uniformity metric defined as:

      U = log ( (1 / (N*(N-1))) * sum{i != j} exp(-2*abs[ f(x_i)-f(x_j) ]^2))

    where embeddings are first L2 normalized so that they lie on the unit hypersphere.
    To reduce computation, a random subsample of size sample_size is used if the number of embeddings is large.
    """
    N = embeddings.shape[0]
    if N > sample_size:
        idx = np.random.choice(N, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        N = sample_size

    # L2 normalise
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)

    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    squared_dists = 2 - 2 * similarity_matrix       # for unit vectors, abs(a - b)^2 = 2 - 2*(a dot b)

    # mask diagonals to stop self comparison
    mask = torch.ones_like(squared_dists, dtype=torch.bool)
    mask.fill_diagonal_(False)

    # compute exponential part of U and then av over all samples used
    exp_neg_sq = torch.exp(-2 * squared_dists)
    avg_exp = exp_neg_sq[mask].mean().item()
    uniformity = np.log(avg_exp)
    return uniformity, N


def main():
    device = "cpu"
    test_data_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\test.h5"
    model_weights_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"

    dataset = SuperDARNDataset(
        h5_file_path=test_data_path,
        negative_value=-9999,
        apply_augmentations=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=contrastive_collate_fn
    )

    model = load_model(model_weights_path, device)
    embeddings = get_embeddings(model, data_loader, dataset_type="test", device=device)
    print(f"Extracted {embeddings.shape[0]} embeddings with shape {embeddings.shape}")

    uniformity_score, samples_used = compute_uniformity_metric(embeddings, sample_size=10000)
    print(f"Uniformity metric: {uniformity_score:.4f}")
    print(f"Uniformity computed using {samples_used} samples")

if __name__ == "__main__":
    main()
