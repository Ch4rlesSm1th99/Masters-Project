import numpy as np
import torch
import h5py
from datetime import datetime

from charles.self_supervised_approach.models import BaseEncoder
from charles.self_supervised_approach.data_loader import SuperDARNDataset

# file paths --> original data + the output embeddings from running extract_full_embeddings
TRAIN_H5 = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5"
VAL_H5 = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5"
TEST_H5 = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\test.h5"

TRAIN_EMB = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings\train.npy"
TRAIN_SEG = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings\train_segments.npy"

VAL_EMB = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings\val.npy"
VAL_SEG = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings\val_segments.npy"

TEST_EMB = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings\test.npy"
TEST_SEG = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\embeddings\test_segments.npy"

MODEL_WEIGHTS = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth"


DATASET_PATHS = {
    "train": TRAIN_H5,
    "val":   VAL_H5,
    "test":  TEST_H5
}

# info for each set
subsets_info = [
    ("train", TRAIN_H5, TRAIN_EMB, TRAIN_SEG),
    ("val",   VAL_H5,   VAL_EMB,   VAL_SEG),
    ("test",  TEST_H5,  TEST_EMB,  TEST_SEG)
]


embeddings = None
segment_names = None
offsets = {}            # offsets help determine which set the index belongs to allowing it to open the correct h5 file when plotting
model = None

# init
def initialize():
    """
    Concatenate train/val/test embeddings + segments into 'all' arrays. Also load the model. The user should see
    a single set of data from whatever sets they loaded in from the output of pre_processing.py.
    """
    global embeddings, segment_names, offsets, model

    emb_list = []
    seg_list = []
    for subset_name, h5_path, emb_path, seg_path in subsets_info:
        e = np.load(emb_path)
        s = np.load(seg_path, allow_pickle=True)
        emb_list.append(e)
        seg_list.append(s)

    # cancatenate data
    embeddings = np.vstack(emb_list)         # shape: [N, features]
    segment_names = np.concatenate(seg_list) # shape: [N,]
    offsets["train_end"] = emb_list[0].shape[0]
    offsets["val_end"]   = offsets["train_end"] + emb_list[1].shape[0]
    offsets["test_end"]  = offsets["val_end"]   + emb_list[2].shape[0]

    total = embeddings.shape[0]

    # load model
    device = "cpu"
    base_encoder = BaseEncoder(input_channels=1)

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }
    base_encoder.load_state_dict(encoder_state_dict)
    base_encoder.to(device)
    base_encoder.eval()

    model = base_encoder

def parse_time(time_str):
    from datetime import datetime
    try:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
    except:
        return None

def find_anchor_by_time(requested_time_str, beam_number=None):
    req_dt = parse_time(requested_time_str)
    if req_dt is None:
        return 0  # fallback if parse fails

    best_idx = 0
    best_diff = float('inf')

    # store timestamps & beam numbers, helps more effecient file search
    segment_times = np.full(len(segment_names), np.nan)  # later stores start time timestamps for segments
    beam_numbers = np.full(len(segment_names), -1)  # store beam num for each segment

    # load all timestamps and beamnumbers
    for subset_name, h5_file in DATASET_PATHS.items():
        with h5py.File(h5_file, "r") as hf:
            for i, seg_name in enumerate(segment_names):
                try:
                    start_str = hf[seg_name].attrs.get("start_time", "unknown")
                    beam_num = hf[seg_name].attrs.get("beam_number", -1)  # store beam
                    seg_dt = parse_time(start_str)

                    if seg_dt:
                        segment_times[i] = (seg_dt - datetime(1970,1,1)).total_seconds()
                        beam_numbers[i] = beam_num  # assign beam number
                except KeyError:
                    continue  # ignore missing segments

    # convert requested time to timestamp
    req_timestamp = (req_dt - datetime(1970,1,1)).total_seconds()

    # filter the indices, if beam locking is ticked
    if beam_number is not None:
        valid_indices = np.where(beam_numbers == beam_number)[0]  # only keep beams of same number eg. all beam_6
    else:
        valid_indices = np.arange(len(segment_names))  # use all segments

    if len(valid_indices) == 0:
        return 0

    # find closest match
    abs_diffs = np.abs(segment_times[valid_indices] - req_timestamp)
    best_idx_local = np.nanargmin(abs_diffs)
    best_idx = valid_indices[best_idx_local]  # map back to global index

    return best_idx


def find_anchor_by_segment_name(segment_name_str):
    """
    If exact match is not found, do a fallback approach by same beam + closest segment ID.
    """
    # check for exact match
    if segment_name_str in segment_names:
        return np.where(segment_names == segment_name_str)[0][0]

    # fallback
    beam_prefix = segment_name_str.split("_segment_")[0]
    try:
        requested_num = int(segment_name_str.split("_segment_")[1])
    except:
        requested_num = None

    same_beam_indices = []
    for i, seg in enumerate(segment_names):
        if seg.startswith(beam_prefix):
            same_beam_indices.append(i)

    if not same_beam_indices:
        return 0

    if requested_num is None:
        # if no segment number => pick first
        return same_beam_indices[0]

    best_idx = same_beam_indices[0]
    best_diff = float('inf')
    for idx in same_beam_indices:
        seg_num = int(segment_names[idx].split("_segment_")[1])
        diff = abs(seg_num - requested_num)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx

    return best_idx

def filter_neighbors(anchor_idx, same_beam, time_exclude, k=5):
    """
    Return top-k neighbors by cosine similarity as distance metric (same used in model loss in training).
    If same_beam => only neighbors with the same beam_number.
    If time_exclude => skip neighbors within that time window.
    """
    emb_t = torch.tensor(embeddings, dtype=torch.float32)
    anchor_emb = emb_t[anchor_idx].unsqueeze(0)

    sim_scores = torch.nn.functional.cosine_similarity(anchor_emb, emb_t, dim=1)
    sim_scores[anchor_idx] = -9999  # exclude anchor itself

    sorted_indices = torch.argsort(sim_scores, descending=True)

    final_list = []
    for nn_idx in sorted_indices:
        neighbor_i = nn_idx.item()
        if same_beam:
            if same_beam:
                # determine subset and file path for both indices
                s1 = "train" if anchor_idx < offsets["train_end"] else \
                    "val" if anchor_idx < offsets["val_end"] else "test"
                s2 = "train" if neighbor_i < offsets["train_end"] else \
                    "val" if neighbor_i < offsets["val_end"] else "test"

                seg1 = segment_names[anchor_idx]
                seg2 = segment_names[neighbor_i]

                # open HDF5 files and compare beam numbers
                with h5py.File(DATASET_PATHS[s1], "r") as hf1, h5py.File(DATASET_PATHS[s2], "r") as hf2:
                    beam1 = hf1[seg1].attrs.get("beam_number", -1)
                    beam2 = hf2[seg2].attrs.get("beam_number", -1)

                if beam1 != beam2:
                    continue

        if time_exclude != "none":
            if within_time_window(anchor_idx, neighbor_i, time_exclude):
                continue
        final_list.append(neighbor_i)
        if len(final_list) >= k:
            break
    return final_list

def within_time_window(i1, i2, time_exclude):
    s1 = "train" if i1 < offsets["train_end"] else \
         "val" if i1 < offsets["val_end"] else "test"
    s2 = "train" if i2 < offsets["train_end"] else \
         "val" if i2 < offsets["val_end"] else "test"   # determine subset for both indices

    seg1 = segment_names[i1]
    seg2 = segment_names[i2]
    file1 = DATASET_PATHS[s1]
    file2 = DATASET_PATHS[s2]

    with h5py.File(file1, "r") as hf:
        dt1_str = hf[seg1].attrs.get("start_time", "unknown")
    with h5py.File(file2, "r") as hf:
        dt2_str = hf[seg2].attrs.get("start_time", "unknown")

    dt1 = parse_time(dt1_str)
    dt2 = parse_time(dt2_str)
    if dt1 is None or dt2 is None:
        return False

    max_hrs = {
        '1h': 1, '4h': 4, '12h': 12, '1d': 24, '1m': 24 * 30
    }.get(time_exclude, False)

    if not max_hrs:
        return False

    diff_hrs = abs((dt1 - dt2).total_seconds()) / 3600.0
    return diff_hrs < max_hrs


def get_plot_for(anchor_idx, neighbor_idx):
    # determine subset & local index for anchor
    if anchor_idx < offsets["train_end"]:
        sA, lA = "train", anchor_idx
    elif anchor_idx < offsets["val_end"]:
        sA, lA = "val", anchor_idx - offsets["train_end"]
    else:
        sA, lA = "test", anchor_idx - offsets["val_end"]

    # do same for neighbour
    if neighbor_idx < offsets["train_end"]:
        sB, lB = "train", neighbor_idx
    elif neighbor_idx < offsets["val_end"]:
        sB, lB = "val", neighbor_idx - offsets["train_end"]
    else:
        sB, lB = "test", neighbor_idx - offsets["val_end"]

    fileA = DATASET_PATHS[sA]
    fileB = DATASET_PATHS[sB]

    dsA = SuperDARNDataset(fileA, negative_value=-9999, apply_augmentations=False)
    dsB = SuperDARNDataset(fileB, negative_value=-9999, apply_augmentations=False)

    anchor_data_tuple = dsA[lA]    # (normed_data, None, unscaled_data, None, seg_name)
    neighbour_data_tuple = dsB[lB]

    return {
        "anchor_seg": anchor_data_tuple,
        "neighbor_seg": neighbour_data_tuple
    }
