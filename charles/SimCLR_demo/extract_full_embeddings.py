import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader

# Local imports
from charles.self_supervised_approach.models import BaseEncoder
from charles.self_supervised_approach.data_loader import SuperDARNDataset, contrastive_collate_fn


TRAIN_PATH = r"/Masters-Project/charles/data/train.h5"
VAL_PATH   = r"/Masters-Project/charles/data/val.h5"
TEST_PATH  = r"/Masters-Project/charles/data/test.h5"
MODEL_WEIGHTS = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"
EMBED_SAVE_DIR = r"/Masters-Project/charles/data/embeddings"


def extract_embeddings(h5_file_path, dataset_name, model, device="cpu", batch_size=500):
    """
    1. Creates a SuperDARNDataset from the given h5_file_path
    2. Loads it with a DataLoader
    3. Extracts embeddings using `model`
    4. Saves embeddings and segment names

    :param h5_file_path: Path to the dataset HDF5 file.
    :param dataset_name: String (e.g. 'train', 'val', 'test') from pre_processing.py.
    :param model: A loaded BaseEncoder in eval mode.
    """

    dataset = SuperDARNDataset(
        h5_file_path=h5_file_path,
        negative_value=-9999,
        apply_augmentations=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=contrastive_collate_fn
    )
    segment_names = dataset.segments

    # extract embeddings
    embeddings_list = []
    seg_name_list = []

    with torch.no_grad():
        for batch_tuple in data_loader:
            # shape: (batch_data, segment_names, raw_data, _)
            batch_data, seg_batch_names, _, _ = batch_tuple

            # shape: [batch_size, height, width]
            batch_data = batch_data.unsqueeze(1).to(device)  # [batch_size, 1, H, W]
            emb = model(batch_data)  # shape: [batch_size, feature_dim]
            embeddings_list.append(emb.cpu().numpy())

            seg_name_list.extend(seg_batch_names)

    all_embeddings = np.vstack(embeddings_list)  # shape [num_samples, feature_dim]

    # save check
    os.makedirs(EMBED_SAVE_DIR, exist_ok=True)
    emb_path = os.path.join(EMBED_SAVE_DIR, f"{dataset_name}.npy")
    seg_path = os.path.join(EMBED_SAVE_DIR, f"{dataset_name}_segments.npy")

    np.save(emb_path, all_embeddings)
    np.save(seg_path, np.array(seg_name_list, dtype=object))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load model
    base_encoder = BaseEncoder(input_channels=1)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    # extract wieghts
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }
    base_encoder.load_state_dict(encoder_state_dict)
    base_encoder.to(device)
    base_encoder.eval()

    # extract embeddings and save
    extract_embeddings(
        h5_file_path=TRAIN_PATH,
        dataset_name="train",
        model=base_encoder,
        device=device,
        batch_size=500
    )
    extract_embeddings(
        h5_file_path=VAL_PATH,
        dataset_name="val",
        model=base_encoder,
        device=device,
        batch_size=500
    )
    extract_embeddings(
        h5_file_path=TEST_PATH,
        dataset_name="test",
        model=base_encoder,
        device=device,
        batch_size=500
    )

if __name__ == "__main__":
    main()
