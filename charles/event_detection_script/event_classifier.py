import os
import bz2
import pydarn
import datetime as dt
import numpy as np
import pandas as pd
import warnings
import h5py
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

#############################################
# Pre-processing: Build the HDF5 file
#############################################
def pre_processing():
    """
    Reads .bz2 files from a fixed debugging data directory,
    processes them for each beam (here using beam 0 only for simplicity),
    and saves all segments into a single HDF5 file with shape (30,75,1).
    """
    # Hard-coded settings
    data_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\1996"
    save_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\event_demo_saves"
    output_h5_file = os.path.join(save_directory, 'all_beams_selected_data.h5')
    negative_value = -9999
    min_valid_points = 100
    num_time_steps = 30
    beams_to_process = range(1)  # Adjust as needed

    def segment_and_select_closest_features(
            power_array,
            timestamps,
            segment_length='1H',
            num_time_steps=30,
            min_valid_points=100,
            negative_value=-9999
    ):
        """
        Splits data into time segments (e.g. 1-hour windows) and then selects
        num_time_steps time points from each segment by picking times closest to
        equally spaced desired timestamps.
        Returns a list of arrays of shape (30, 75, 1) and a list of start times.
        """
        segments = []
        segment_times = []
        delta = pd.Timedelta(segment_length)
        if len(timestamps) == 0:
            return segments, segment_times
        current_time = timestamps[0]
        end_time = timestamps[-1]
        while current_time < end_time:
            next_time = current_time + delta
            mask = (timestamps >= current_time) & (timestamps < next_time)
            segment_times_in_window = timestamps[mask]
            if np.any(mask):
                seg_power = power_array[mask]
                valid_points = np.sum(seg_power != negative_value)
                if valid_points >= min_valid_points:
                    print(f"Processing segment starting at {current_time} with {valid_points} valid points.")
                    desired_times = pd.date_range(
                        start=current_time,
                        end=next_time - pd.Timedelta('1ns'),
                        periods=num_time_steps
                    )
                    selected_indices = []
                    for desired_time in desired_times:
                        time_diffs = np.abs(segment_times_in_window - desired_time)
                        min_diff_idx = np.argmin(time_diffs)
                        selected_indices.append(min_diff_idx)
                    selected_indices = np.array(selected_indices)
                    selected_power = seg_power[selected_indices]  # shape (30,75)
                    selected_power = np.expand_dims(selected_power, axis=-1)  # shape (30,75,1)
                    segments.append(selected_power)
                    segment_times.append(current_time)
            current_time = next_time
        return segments, segment_times

    def load_and_preprocess_beam(
            data_directory,
            beam_number,
            negative_value,
            min_valid_points=100,
            num_time_steps=30
    ):
        """
        Reads all .bz2 files in data_directory, filters records for the given beam_number,
        and returns a list of (segment_data, start_time, stid_list) for that beam.
        """
        all_records = []
        files = [f for f in os.listdir(data_directory) if f.endswith('.bz2')]
        if not files:
            print("No .bz2 files found in the directory.")
            return []
        print(f"\n[Beam {beam_number}] Processing {len(files)} files...")
        for fitacf_file_name in files:
            fitacf_file = os.path.join(data_directory, fitacf_file_name)
            try:
                with bz2.open(fitacf_file, 'rb') as fp:
                    fitacf_stream = fp.read()
                sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
                fitacf_data = sdarn_read.read_fitacf()
                if not fitacf_data:
                    continue
                for record in fitacf_data:
                    if record.get('bmnum') != beam_number:
                        continue
                    if 'slist' not in record or len(record['slist']) == 0:
                        continue
                    try:
                        record_time = dt.datetime(
                            record['time.yr'],
                            record['time.mo'],
                            record['time.dy'],
                            record['time.hr'],
                            record['time.mt'],
                            record['time.sc'],
                            int(record['time.us'] / 1000)
                        )
                    except ValueError:
                        continue
                    common_data = {
                        'time': record_time,
                        'bmnum': record['bmnum'],
                        'stid': record['stid'],
                        'nrang': record['nrang'],
                        'frang': record['frang'],
                        'rsep': record['rsep'],
                    }
                    slist = record['slist']
                    for idx, gate in enumerate(slist):
                        gate_data = common_data.copy()
                        gate_data.update({
                            'range_gate': gate,
                            'p_l': record['p_l'][idx],
                        })
                        all_records.append(gate_data)
            except Exception as e:
                print(f"[Beam {beam_number}] Error reading {fitacf_file_name}: {e}")
        if not all_records:
            print(f"[Beam {beam_number}] No data collected.")
            return []
        df_beam = pd.DataFrame(all_records)
        df_beam['stid'] = df_beam['stid'].astype(int)
        df_beam['range_gate'] = df_beam['range_gate'].astype(int)
        df_beam.set_index(['time', 'range_gate'], inplace=True)
        df_beam.sort_index(inplace=True)
        df_beam = df_beam[~df_beam.index.duplicated(keep='first')]
        df_beam['p_l'] = df_beam['p_l'].fillna(negative_value)
        times = df_beam.index.get_level_values('time').unique()
        range_gates = np.arange(75)
        power_pivot = (df_beam['p_l']
                       .unstack(level='range_gate')
                       .reindex(index=times, columns=range_gates, fill_value=negative_value))
        power_pivot.fillna(negative_value, inplace=True)
        timestamps = pd.to_datetime(power_pivot.index)
        power_array = power_pivot.values
        segments, segment_times = segment_and_select_closest_features(
            power_array,
            timestamps,
            segment_length='1H',
            num_time_steps=num_time_steps,
            min_valid_points=min_valid_points,
            negative_value=negative_value
        )
        stid_list = df_beam['stid'].unique()
        results = []
        for seg_data, seg_start in zip(segments, segment_times):
            results.append((seg_data, seg_start, stid_list))
        del df_beam, power_pivot
        gc.collect()
        return results

    with h5py.File(output_h5_file, 'w') as hf_full:
        for beam_number in beams_to_process:
            beam_segments = load_and_preprocess_beam(
                data_directory,
                beam_number,
                negative_value,
                min_valid_points=min_valid_points,
                num_time_steps=num_time_steps
            )
            if not beam_segments:
                print(f"No valid segments or no data for beam {beam_number}.")
                continue
            beam_group = hf_full.create_group(f'beam_{beam_number}')
            for idx_seg, (seg_data, seg_start_time, stid_list) in enumerate(beam_segments):
                seg_group = beam_group.create_group(f'segment_{idx_seg}')
                seg_group.create_dataset('data', data=seg_data)
                start_time_dt = pd.to_datetime(seg_start_time).to_pydatetime()
                end_time_dt = start_time_dt + pd.Timedelta('1H')
                seg_group.attrs['start_time'] = start_time_dt.isoformat()
                seg_group.attrs['end_time'] = end_time_dt.isoformat()
                seg_group.attrs['beam_number'] = beam_number
                seg_group.attrs['stid_list'] = ','.join(str(x) for x in stid_list)
    print(f"\nAll segments saved to: {output_h5_file}")
    return output_h5_file

#############################################
# PyTorch Dataset for Inference (with Metadata)
#############################################
class SuperDARNInferenceDataset(Dataset):
    """
    Reads the HDF5 file containing segments (shape: (30,75,1)) and returns
    each sample as a tensor (1,30,75) along with its start timestamp.
    Normalizes valid (non--9999) points using global mean/std.
    """
    global_power_mean = 15.023935
    global_power_std = 9.644889
    negative_value = -9999

    def __init__(self, h5_file_path):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.all_paths = []
        with h5py.File(self.h5_file_path, 'r') as hf:
            for beam_name in hf.keys():
                if beam_name.startswith("beam_"):
                    for segment_name in hf[beam_name].keys():
                        self.all_paths.append((beam_name, segment_name))

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        beam_name, segment_name = self.all_paths[idx]
        with h5py.File(self.h5_file_path, 'r') as hf:
            seg_grp = hf[beam_name][segment_name]
            data = seg_grp["data"][:]  # shape: (30,75,1)
            data = np.squeeze(data, axis=-1)  # shape: (30,75)
            start_time = seg_grp.attrs['start_time']
        valid_mask = (data != self.negative_value)
        data[valid_mask] = ((data[valid_mask] - self.global_power_mean) /
                            self.global_power_std)
        data = torch.from_numpy(data).float().unsqueeze(0)  # shape: (1,30,75)
        return data, start_time

#############################################
# Model Definitions
#############################################
class BaseEncoder(nn.Module):
    """
    Base CNN encoder for extracting features from input data.
    """
    def __init__(self, input_channels=1):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return x

class FrozenEncoderBinaryClassifier(nn.Module):
    """
    Model with a frozen encoder and a binary classification head.
    """
    def __init__(self, encoder, embedding_dim=512, num_classes=1):
        super(FrozenEncoderBinaryClassifier, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            emb = self.encoder(x)
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)
        return logits


def load_full_model(model_path, device="cpu"):
    """
    Loads the full classifier (frozen encoder + binary head).
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = FrozenEncoderBinaryClassifier(BaseEncoder(input_channels=1), embedding_dim=512, num_classes=1)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def run_inference(model, h5_file_path, batch_size=8, device="cpu", threshold=0.5):
    """
    For each segment, if the probability > threshold, the start_time is recorded.
    saves the detected event timestamps to a CSV file. Returns the CSV file path.
    """
    dataset = SuperDARNInferenceDataset(h5_file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    event_timestamps = []
    with torch.no_grad():
        for batch_data, meta_batch in loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            probs = torch.sigmoid(logits)
            # Check each sample in the batch
            for prob, meta in zip(probs, meta_batch):
                if prob.item() > threshold:
                    event_timestamps.append(meta)
    if event_timestamps:
        print("Detected events at the following timestamps:")
        for ts in event_timestamps:
            print(ts)
    else:
        print("No events detected (all probabilities below threshold).")
    # Save to CSV file
    csv_path = os.path.join(os.path.dirname(h5_file_path), "detected_events.csv")
    df = pd.DataFrame({"timestamp": event_timestamps})
    df.to_csv(csv_path, index=False)
    print("Detected events saved to CSV at:", csv_path)
    return csv_path

#############################################
# Main: Preprocess, Load Model, and Inference
#############################################
def main():
    print("Starting data pre-processing...")
    h5_file = pre_processing()
    print(f"Pre-processing complete. Data saved at:\n{h5_file}")

    # print first segment for inspection
    with h5py.File(h5_file, 'r') as hf:
        beam_keys = list(hf.keys())
        if not beam_keys:
            print("No beams found in the HDF5 file.")
            return
        first_beam = beam_keys[0]
        segment_keys = list(hf[first_beam].keys())
        if not segment_keys:
            print(f"No segments found in {first_beam}.")
            return
        first_segment = segment_keys[0]
        segment_data = hf[first_beam][first_segment]['data'][:]
        print(f"\nFirst Segment from {first_beam}/{first_segment}:")
        print("Shape:", segment_data.shape)
        print("Sample Data (first 5 time steps):")
        print(segment_data[:5, :, 0])

    # load the full classifier model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\fine_tuned_classifier.pth"
    model = load_full_model(model_weights_path, device=device)
    print("Model loaded successfully.")

    # inference and save detected events to CSV.
    print("Running inference on preprocessed data...")
    csv_output = run_inference(model, h5_file, batch_size=8, device=device, threshold=0.5)
    print("Inference complete.")
    print("CSV file with detected events:", csv_output)

if __name__ == "__main__":
    main()
