import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datetime import datetime
from torch.utils.data import DataLoader


from models import BaseEncoder
from data_loader import SuperDARNDataset, contrastive_collate_fn


def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return None

def filter_neighbor(
    anchor_beam,
    anchor_time,
    neighbor_beam,
    neighbor_time,
    filter_same_event=True,
    filter_within_1hr=False,
    filter_within_12hrs=False,
    filter_within_1day=False,
    filter_within_1month=False,
    filter_within_3months=False,
    filter_within_6months=False
):
    if not isinstance(anchor_time, datetime) or not isinstance(neighbor_time, datetime):
        pass
    else:
        # time diff in hours
        time_diff_hrs = abs((anchor_time - neighbor_time).total_seconds()) / 3600.0

        # exclusion condition for a specific time window option
        if filter_within_1hr and time_diff_hrs < 1:
            return True
        if filter_within_12hrs and time_diff_hrs < 12:
            return True
        if filter_within_1day and time_diff_hrs < 24:
            return True
        if filter_within_1month and time_diff_hrs < (24 * 30):
            return True
        if filter_within_3months and time_diff_hrs < (24 * 30 * 3):
            return True
        if filter_within_6months and time_diff_hrs < (24 * 30 * 6):
            return True

    # same event filter => exclude if same time but different beam, stops view of same event from different beams
    if filter_same_event and (anchor_time is not None) and (neighbor_time is not None):
        if anchor_time == neighbor_time and anchor_beam != neighbor_beam:
            return True

    # if no conditions are violated return the neighbour
    return False

def load_model(path, device="cpu"):
    base_encoder = BaseEncoder(input_channels=1)
    checkpoint = torch.load(path, map_location=device)

    # extract weights
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }

    # load model
    base_encoder.load_state_dict(encoder_state_dict)
    base_encoder.to(device)
    base_encoder.eval()

    return base_encoder

def get_embeddings(model, data_loader, dataset_type="test", device="cpu"):
    save_dir = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings"
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise ValueError(f"⚠️ Error: {save_dir} exists but is not a directory.")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{dataset_type}.npy")

    # if embeddings already exist, load them
    if os.path.exists(save_path):
        embs = np.load(save_path)
        return embs

    # if embeddings unavailable, compute embeddings
    embeddings = []
    with torch.no_grad():
        for batch_tuple in data_loader:
            batch, *_ = batch_tuple  # extract tensor
            batch = batch.unsqueeze(1).to(device)  # shape [batch_size, 1, H, W]
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    embs = np.vstack(embeddings)
    np.save(save_path, embs)
    print(f"Extracted {embs.shape[0]} {dataset_type} embeddings and saved to {save_path}")
    return embs


def visualize_single_anchor(
    embs,
    dataset,
    segment_names,
    h5_file_path,
    anchor_idx=0,
    k=1,
    filter_same_event=True,
    filter_within_1hr=False,
    filter_within_12hrs=False,
    filter_within_1day=False,
    filter_within_1month=False,
    filter_within_3months=False,
    filter_within_6months=False
):
    negative_value = -9999

    # convert embeddings to torch tensors
    embs_t = torch.tensor(embs, dtype=torch.float32)
    embs_t = F.normalize(embs_t, p=2, dim=1)

    # compute distance metric (cosine similarity)
    similarity_matrix = torch.mm(embs_t, embs_t.T)
    N = embs_t.shape[0]
    if anchor_idx < 0 or anchor_idx >= N:
        raise ValueError(f"anchor_idx {anchor_idx} out of range (0 to {N-1}).")

    # collect anchor metadata
    anchor_seg_name = segment_names[anchor_idx]
    with h5py.File(h5_file_path, "r") as hf:
        anchor_group = hf[anchor_seg_name]
        anchor_beam = anchor_group.attrs.get("beam_number", -1)
        anchor_start_str = anchor_group.attrs.get("start_time", "unknown")
        anchor_end_str = anchor_group.attrs.get("end_time", "unknown")

    anchor_dt = parse_time(anchor_start_str)

    print(f"\n--- Anchor (idx={anchor_idx}) ---")
    print(f"Segment Name: {anchor_seg_name}")
    print(f"Beam Number: {anchor_beam}")
    print(f"Start Time : {anchor_start_str}")
    print(f"End Time   : {anchor_end_str}")

    # get neighbours
    sim_scores = similarity_matrix[anchor_idx].clone()
    sim_scores[anchor_idx] = -9999  # Exclude itself
    neighbor_indices = torch.argsort(sim_scores, descending=True)

    filtered_indices = []
    with h5py.File(h5_file_path, "r") as hf:
        for nn_idx in neighbor_indices:
            if len(filtered_indices) >= k:
                break

            nn_seg_name = segment_names[nn_idx]
            nn_group = hf[nn_seg_name]
            nn_beam = nn_group.attrs.get("beam_number", -1)
            nn_start_str = nn_group.attrs.get("start_time", "unknown")
            nn_end_str = nn_group.attrs.get("end_time", "unknown")
            nn_dt = parse_time(nn_start_str)

            # Call filter_neighbor to decide if we exclude
            exclude = filter_neighbor(
                anchor_beam=anchor_beam,
                anchor_time=anchor_dt,
                neighbor_beam=nn_beam,
                neighbor_time=nn_dt,
                filter_same_event=filter_same_event,
                filter_within_1hr=filter_within_1hr,
                filter_within_12hrs=filter_within_12hrs,
                filter_within_1day=filter_within_1day,
                filter_within_1month=filter_within_1month,
                filter_within_3months=filter_within_3months,
                filter_within_6months=filter_within_6months
            )
            if exclude:
                continue

            print(f"\n>>> Potential Neighbor (idx={nn_idx}): {nn_seg_name}")
            print(f"    Beam={nn_beam}, start={nn_start_str}, end={nn_end_str}")
            filtered_indices.append(nn_idx)

    # plot anchor
    anchor_norm, _, anchor_unscaled, _, _ = dataset[anchor_idx]
    anchor_unscaled = anchor_unscaled.copy()
    valid_mask = (anchor_unscaled != negative_value)
    vmin, vmax = anchor_unscaled[valid_mask].min(), anchor_unscaled[valid_mask].max()

    fig, axs = plt.subplots(1, k + 1, figsize=(15, 5), sharey=True)

    im0 = axs[0].imshow(
        np.ma.masked_where(anchor_unscaled == negative_value, anchor_unscaled).T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )
    axs[0].set_title(
        f"Anchor idx={anchor_idx}\n"
        f"{anchor_seg_name}\n"
        f"{anchor_start_str} - {anchor_end_str}"
    )
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Range Gates")
    plt.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)

    # plot neighbours
    for j, nn_idx in enumerate(filtered_indices):
        nn_norm, _, nn_unscaled, _, nn_seg_name = dataset[nn_idx]
        nn_unscaled = nn_unscaled.copy()

        with h5py.File(h5_file_path, "r") as hf:
            nn_group = hf[nn_seg_name]
            nn_start_str = nn_group.attrs.get("start_time", "unknown")
            nn_end_str = nn_group.attrs.get("end_time", "unknown")

        valid_mask_nn = (nn_unscaled != negative_value)
        im = axs[j + 1].imshow(
            np.ma.masked_where(nn_unscaled == negative_value, nn_unscaled).T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )
        axs[j + 1].set_title(
            f"NN idx={nn_idx}\n"
            f"{nn_seg_name}\n"
            f"{nn_start_str} - {nn_end_str}"
        )
        axs[j + 1].set_xlabel("Time Steps")
        plt.colorbar(im, ax=axs[j + 1], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    print("\n--- Done plotting ---\n")



test_data_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\test.h5"
model_weights_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"

def main():
    device = "cpu"

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

    embs = get_embeddings(model, data_loader, dataset_type="test", device=device)
    print(f"✅ Extracted {embs.shape[0]} embeddings with shape {embs.shape}")

    segment_names = dataset.segments

    anchor_idx = 0

    visualize_single_anchor(
        embs=embs,
        dataset=dataset,
        segment_names=segment_names,
        h5_file_path=test_data_path,
        anchor_idx=anchor_idx,
        k=1,
        filter_same_event=True,
        filter_within_1hr=False,
        filter_within_12hrs=True,
        filter_within_1day=False,
        filter_within_1month=False,
        filter_within_3months=False,
        filter_within_6months=False
    )

if __name__ == "__main__":
    main()
