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
import argparse
import csv

from charles.self_supervised_approach.models import BaseEncoder
from charles.self_supervised_approach.data_loader import SuperDARNDataset, contrastive_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Run clustering pipeline on train, val, or test dataset.")

    # choose dataset
    parser.add_argument('--run_set', type=str, default='train', choices=['train', 'val', 'test'],
                        help="Which dataset to run: train, val, or test.")

    # set data paths
    parser.add_argument('--train_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5",
                        help='Path to training H5 file.')
    parser.add_argument('--val_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5",
                        help='Path to validation H5 file.')
    parser.add_argument('--test_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\test.h5",
                        help='Path to test H5 file.')

    # clustering parameters for DBSCAN
    parser.add_argument('--min_samples', type=int, default=-1,
                        help="Manual min_samples for DBSCAN. If -1, dynamic scaling is used.")
    parser.add_argument('--baseline_size', type=int, default=9000,
                        help="Baseline dataset size for dynamic min_samples scaling.")
    parser.add_argument('--baseline_min_samples', type=int, default=30,
                        help="Baseline min_samples for dynamic scaling.")
    parser.add_argument('--adjust_factor', type=float, default=0.9,
                        help="Adjustment factor for dynamic min_samples scaling.")

    # plotting and dislplay options
    parser.add_argument('--plot_noise', action='store_true', default=False,
                        help="If set, noise points will be plotted in the PCA visualization.")
    parser.add_argument('--cluster_to_inspect', type=int, default=1,
                        help="Cluster label to inspect for random sample visualization.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help="Number of random samples to plot from the specified cluster.")

    # fine-tuning dataset saving option -- save pseudo labels for training classifier head
    parser.add_argument('--save_labels', action='store_true', default=False,
                        help="If set, save a CSV file with binary labels for fine-tuning.")
    parser.add_argument('--fine_tuning_dir', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels",
                        help="Directory to save the fine-tuning labeled dataset.")

    return parser.parse_args()


def load_model(path, device="cpu"):
    base_encoder = BaseEncoder(input_channels=1)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
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
    Visualises clusters using PCA to reduce dimensionality to 2D.
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_norm)

    plt.figure(figsize=(10, 8))
    noise_mask = cluster_labels == -1
    non_noise_mask = ~noise_mask

    scatter = plt.scatter(
        embeddings_2d[non_noise_mask, 0],
        embeddings_2d[non_noise_mask, 1],
        c=cluster_labels[non_noise_mask],
        cmap='viridis',
        s=10,
        alpha=0.7,
        label="Clusters"
    )

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
    """
    indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
    if len(indices) == 0:
        print(f"No samples found for cluster {cluster}")
        return

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
    Prints details about the specified cluster (metadata and size).
    """
    indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
    print(f"Cluster {cluster} has {len(indices)} samples.")

    if len(indices) == 0:
        print("No samples in this cluster.")
        return

    cluster_embeddings = embeddings_norm[indices]
    mean_emb = np.mean(cluster_embeddings, axis=0)
    std_emb = np.std(cluster_embeddings, axis=0)
    print("Embedding statistics:")
    print(f"  Mean (first 10 components): {mean_emb[:10]}")
    print(f"  Std  (first 10 components): {std_emb[:10]}")

    print("Sample segment details from this cluster:")
    with h5py.File(dataset.h5_file_path, 'r') as hf:
        sample_names = [dataset.segments[i] for i in indices[:10]]
        for name in sample_names:
            group = hf[name]
            start_time = group.attrs.get("start_time", "unknown")
            end_time = group.attrs.get("end_time", "unknown")
            print(f"  {name}: {start_time} to {end_time}")


def main():
    args = parse_args()
    device = "cpu"

    # choose the correct data path based on the run_set argument
    run_set = args.run_set.lower()
    if run_set == "train":
        data_path = args.train_h5_file_path
    elif run_set == "val":
        data_path = args.val_h5_file_path
    elif run_set == "test":
        data_path = args.test_h5_file_path
    else:
        raise ValueError("Invalid run_set specified.")

    model_weights_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"

    dataset = SuperDARNDataset(
        h5_file_path=data_path,
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
    embeddings = get_embeddings(model, data_loader, dataset_type=run_set, device=device)
    print(f"Extracted {embeddings.shape[0]} embeddings with shape {embeddings.shape}")

    # L2 normalsiation
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1).numpy()

    # compute the min_sampels for a cluster based on the size of the dataset (about 30/9000 * size)
    if args.min_samples > 0:
        dynamic_min_samples = args.min_samples
    else:
        dynamic_min_samples = int(args.baseline_min_samples * (len(dataset) / args.baseline_size) * args.adjust_factor)
    print(f"Using min_samples = {dynamic_min_samples} for DBSCAN (dynamic scaling)")

    # -------------------------
    # DBSCAN Clustering
    # -------------------------
    eps = 0.2
    dbscan = DBSCAN(eps=eps, min_samples=dynamic_min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings_norm)

    # -------------------------
    # Silhouette Score
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

    # visualise clusters with PCA
    visualize_clusters(embeddings_norm, cluster_labels, plot_noise=args.plot_noise)

    # inspect samples in the clusters
    visualize_cluster_samples(dataset, cluster_labels, cluster=args.cluster_to_inspect, num_samples=args.num_samples)

    print_cluster_details(dataset, cluster_labels, embeddings_norm, cluster=args.cluster_to_inspect)

    if args.save_labels:
        ft_dir = args.fine_tuning_dir
        os.makedirs(ft_dir, exist_ok=True)
        output_path = os.path.join(ft_dir, f"labels_{run_set}.csv")
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "segment_name", "label"])
            for i, seg_name in enumerate(dataset.segments):
                # label 1 if the sample's cluster matches the desired cluster, 0 otherwise.
                new_label = 1 if cluster_labels[i] == args.cluster_to_inspect else 0
                writer.writerow([i, seg_name, new_label])
        print(f"Saved fine-tuning labels to {output_path}")


if __name__ == "__main__":
    main()
