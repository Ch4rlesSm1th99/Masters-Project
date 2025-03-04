import os
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def get_embeddings(model, data_loader, dataset_type="test", device="cpu"):
    """
    Computes embeddings for all samples in the provided dataloader.
    If embeddings have already been computed and saved, they are loaded from disk.
    """
    save_dir = r"/charles/data/embeddings"
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise ValueError(f"Error: {save_dir} exists but is not a directory.")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{dataset_type}.npy")

    if os.path.exists(save_path):
        embs = np.load(save_path)
        return embs

    embeddings = []
    with torch.no_grad():
        for batch_tuple in data_loader:
            batch, *_ = batch_tuple
            batch = batch.unsqueeze(1).to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    embs = np.vstack(embeddings)
    np.save(save_path, embs)
    print(f"Extracted {embs.shape[0]} {dataset_type} embeddings and saved to {save_path}")
    return embs


def visualize_clusters(embeddings_norm, cluster_labels, plot_noise=True):
    """
    Visualises clusters using PCA to reduce dimensionality to 2D. PCA takes the two most dominant
    features with principle component 2 being orthogonal to principle component 1 and plots them
    to visualise clusters.

    Parameters:
      embeddings_norm (np.array): L2 normalized embeddings
      cluster_labels (np.array): Cluster labels assigned by DBSCAN
      plot_noise (bool): Whether to plot noise points (label == -1)
    """
    # reduce dimensionality with PCA to 2 components
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_norm)

    plt.figure(figsize=(10, 8))

    # separate non-noise points and noise points
    noise_mask = cluster_labels == -1
    non_noise_mask = ~noise_mask

    # plot non-noise points
    scatter = plt.scatter(
        embeddings_2d[non_noise_mask, 0],
        embeddings_2d[non_noise_mask, 1],
        c=cluster_labels[non_noise_mask],
        cmap='viridis',
        s=10,
        alpha=0.7,
        label="Clusters"
    )

    # plot noise points if bool is True
    if plot_noise and np.any(noise_mask):
        plt.scatter(
            embeddings_2d[noise_mask, 0],
            embeddings_2d[noise_mask, 1],
            c='red',
            s=10,
            alpha=0.7,
            label="Noise"
        )

    plt.colorbar(scatter, label="Cluster label")
    plt.title("PCA Visualization of DBSCAN Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


def visualize_cluster_samples(dataset, cluster_labels, cluster, num_samples=5, negative_value=-9999):
    """
    Visualises a few original samples from the dataset that are assigned to a specific cluster.
    For inspection of trend found that the cluster for auroral events is most typically cluster
    1 but can change if varying epsilon and min number of samples.

    Parameters:
      dataset: dataset object (SuperDARNDataset object).
      cluster_labels (array-like): Array of cluster labels (from DBSCAN).
      cluster (int): The cluster label to visualize (0, 1, ...., num_clusters).
      num_samples (int): Number of samples to display from the cluster.
      negative_value (float): The value used in the dataset to indicate missing data.
    """
    # collect indicies from cluster
    indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
    if len(indices) == 0:
        print(f"No samples found for cluster {cluster}")
        return

    # randomly select a few for visual inspection
    selected_indices = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)

    for idx in selected_indices:
        data_tensor, _, data_unscaled, _, segment_name = dataset[idx]
        data_unscaled = data_unscaled.copy()
        valid_mask = data_unscaled != negative_value
        vmin, vmax = data_unscaled[valid_mask].min(), data_unscaled[valid_mask].max()

        plt.figure(figsize=(6, 4))
        plt.imshow(np.ma.masked_where(data_unscaled == negative_value, data_unscaled).T,
                   aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f"Cluster {cluster} Sample: {segment_name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Range Gates")
        plt.colorbar()
        plt.show()


def print_cluster_details(dataset, cluster_labels, embeddings_norm, cluster=1):
    """
    Prints details about the specified cluster ie metadata and size of cluster

    Parameters:
      dataset: dataset object (SuperDARNDataset object).
      cluster_labels (array-like): Array of cluster labels (from DBSCAN).
      embeddings_norm (np.array): L2 normalized embeddings
      cluster (int): The cluster label to visualize (0, 1, ...., num_clusters).
    """
    # collect indicies
    indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
    print(f"Cluster {cluster} has {len(indices)} samples.")

    if len(indices) == 0:
        print("No samples in this cluster.")
        return

    # compute the stats for the cluster
    cluster_embeddings = embeddings_norm[indices]
    mean_emb = np.mean(cluster_embeddings, axis=0)
    std_emb = np.std(cluster_embeddings, axis=0)
    print("Embedding statistics:")
    print(f"  Mean (first 10 components): {mean_emb[:10]}")
    print(f"  Std  (first 10 components): {std_emb[:10]}")

    # print meta data segment names and timestamps
    print("Sample segment details from this cluster:")
    with h5py.File(dataset.h5_file_path, 'r') as hf:
        sample_names = [dataset.segments[i] for i in indices[:10]]
        for name in sample_names:
            group = hf[name]
            start_time = group.attrs.get("start_time", "unknown")
            end_time = group.attrs.get("end_time", "unknown")
            print(f"  {name}: {start_time} to {end_time}")


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

    # L2 Normalization
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1).numpy()

    # -------------------------
    # DBSCAN Clustering
    # -------------------------
    eps = 0.2
    min_samples = 36
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings_norm)

    # -------------------------
    # Clustering Evaluation: Silhouette Score
    # -------------------------
    mask = cluster_labels != -1
    unique_clusters = np.unique(cluster_labels[mask])
    if len(unique_clusters) > 1:
        sil_score = silhouette_score(embeddings_norm[mask], cluster_labels[mask])
    else:
        sil_score = float('nan')

    num_clusters = len(unique_clusters)
    num_noise = np.sum(cluster_labels == -1)
    print(f"DBSCAN produced {num_clusters} clusters (ignoring noise) with {num_noise} noise points.")
    print(f"Silhouette Score (for non-noise points): {sil_score:.4f}")

    # visualize clusters with PCA
    visualize_clusters(embeddings_norm, cluster_labels, plot_noise=False)

    # visualize original samples from a particular cluster (normally cluster 1 for auroral events)
    cluster_to_inspect = 1  # Change this to any cluster label you wish to inspect
    visualize_cluster_samples(dataset, cluster_labels, cluster=cluster_to_inspect, num_samples=5)

    print_cluster_details(dataset, cluster_labels, embeddings_norm, cluster=1)


if __name__ == "__main__":
    main()
