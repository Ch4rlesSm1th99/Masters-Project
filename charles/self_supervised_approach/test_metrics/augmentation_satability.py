import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from charles.self_supervised_approach.models import BaseEncoder
from charles.self_supervised_approach.data_loader import SuperDARNDataset, contrastive_collate_fn


def load_model(path, device="cpu"):
    base_encoder = BaseEncoder(input_channels=1)
    checkpoint = torch.load(path, map_location=device,weights_only=True)

    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }

    base_encoder.load_state_dict(encoder_state_dict)
    base_encoder.to(device)
    base_encoder.eval()
    return base_encoder


def compute_augmentation_stability(model, dataset, n_samples=100, n_augs=5, device="cpu"):
    """
    For a random subset of n_samples from the dataset, generate n_augs augmented versions,
    compute embeddings, and then compute the average pairwise Euclidean distance between embeddings.
    Lower values indicate higher stability (ie the model produces similar embeddings under augmentations).
    """
    stability_scores = []

    # randomly select samples from dataset to aug
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)

    for idx in indices:
        embeddings = []
        for _ in range(n_augs):
            sample = dataset[idx]
            aug_sample = sample[0]

            # from [75, 30] to [1, 1, 75, 30] to fit expected input
            x = aug_sample.unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(x)
            embeddings.append(emb.squeeze(0))

        # stack embeddings, shape --> [n_augs, embedding_dim]
        embeddings = torch.stack(embeddings, dim=0)

        # compute pairwise Euclidean distances among the n_augs embeddings
        pairwise_distances = []
        for i in range(n_augs):
            for j in range(i + 1, n_augs):
                d = torch.norm(embeddings[i] - embeddings[j], p=2)
                pairwise_distances.append(d.item())

        if pairwise_distances:
            avg_distance = np.mean(pairwise_distances)
            stability_scores.append(avg_distance)

    overall_stability = np.mean(stability_scores) if stability_scores else float('nan')
    return overall_stability


def main():
    device = "cpu"
    train_data_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5"
    model_weights_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"

    dataset = SuperDARNDataset(
        h5_file_path=train_data_path,
        negative_value=-9999,
        apply_augmentations=True
    )

    model = load_model(model_weights_path, device=device)

    n_samples = 100  # number of samples to test
    n_augs = 5  # number of augmentations per sample
    stability_metric = compute_augmentation_stability(model, dataset, n_samples, n_augs, device)

    print(f"Augmentation Stability Metric (ie average pair distance): {stability_metric:.4f}")


if __name__ == "__main__":
    main()
