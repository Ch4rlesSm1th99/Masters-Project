import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from charles.self_supervised_approach.models import BaseEncoder
from charles.self_supervised_approach.data_loader import SuperDARNDataset, contrastive_collate_fn


def load_model(path, device="cpu", weights_only=False):
    base_encoder = BaseEncoder(input_channels=1)
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }
    base_encoder.load_state_dict(encoder_state_dict)
    base_encoder.to(device)
    base_encoder.eval()
    return base_encoder


def get_alignment_metric(embeddings1, embeddings2):
    """
    Alignment Metric is the Euclidean distance between postivie pairs averaged over entire dataset
    """
    # double check the shapes
    if embeddings1.shape != embeddings2.shape:
        raise ValueError("The two sets of embeddings are not the same shape.")

    # apply L2 normalisation --> good alignment should be near zero opposite should be close to root 2
    embeddings1_normalized = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_normalized = F.normalize(embeddings2, p=2, dim=1)

    # compute Euclidean distance
    differences = embeddings1_normalized - embeddings2_normalized
    distances = torch.norm(differences, p=2, dim=1)
    return distances.mean().item()


def main():
    device = "cpu"

    test_data_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5"
    model_weights_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"

    #  two augmented views returned
    dataset = SuperDARNDataset(
        h5_file_path=test_data_path,
        negative_value=-9999,
        apply_augmentations=True
    )
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=contrastive_collate_fn
    )

    model = load_model(model_weights_path, device=device)

    alignment_scores = []
    total_samples = 0

    # loop over dataloader
    for batch_data, segment_names, _, _ in tqdm(data_loader, desc="Processing Batches"):

        x1, x2 = torch.chunk(batch_data, 2, dim=0)      # splits two sets of views in two

        x1 = x1.to(device)
        x2 = x2.to(device)

        x1 = x1.unsqueeze(1)  # adds extra dim for what model expects
        x2 = x2.unsqueeze(1)

        with torch.no_grad():
            emb1 = model(x1)
            emb2 = model(x2)

        # compute alignment for batch
        batch_alignment = get_alignment_metric(emb1, emb2)
        alignment_scores.append(batch_alignment)
        total_samples += x1.shape[0]

    # compute average of the average batch alignments
    overall_alignment = np.mean(alignment_scores)
    print(f"Overall Alignment Metric (average Euclidean distance): {overall_alignment:.4f}")

# good alignment should be near zero opposite should be close to root 2
if __name__ == "__main__":
    main()
