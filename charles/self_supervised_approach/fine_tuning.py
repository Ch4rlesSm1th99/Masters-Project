import os
import csv
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from charles.self_supervised_approach.models import BaseEncoder, FrozenEncoderBinaryClassifier


class FineTuningDataset(Dataset):
    """
    Loads segments from an H5 file and their  labels (0 or 1) from a CSV.
    Applies the same global mean/std normalization for power data
    that you used in SuperDARNDataset.

    CSV must have columns: "index", "segment_name", "label".
    """
    # same global mean/std as in SuperDARNDataset
    global_power_mean = 15.023935
    global_power_std = 9.644889

    def __init__(self, h5_file_path, labels_csv, negative_value=-9999):
        self.h5_file_path = h5_file_path
        self.labels_csv = labels_csv
        self.negative_value = negative_value

        with h5py.File(self.h5_file_path, 'r') as hf:
            self.segments = list(hf.keys())
            # e.g., "segment_0", "segment_1", ...
            self.segments.sort(key=lambda x: int(x.split('_')[-1]))

        # read CSV to build a mapping of segment_name -> label
        self.labels = {}
        with open(self.labels_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.labels[row["segment_name"]] = int(row["label"])

        # filter segments to only those present in the CSV
        self.segments = [seg for seg in self.segments if seg in self.labels]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg_name = self.segments[idx]
        with h5py.File(self.h5_file_path, 'r') as hf:
            data = hf[seg_name]['data'][:]
            power_data = data[:, :, 0].astype(np.float32)

        # normalise power except padding vals of -9999
        valid_mask = (power_data != self.negative_value)
        power_data[valid_mask] = (
            (power_data[valid_mask] - self.global_power_mean)
            / self.global_power_std
        )

        power_tensor = torch.from_numpy(power_data).unsqueeze(0)

        label = self.labels[seg_name]
        return power_tensor, label


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy



def load_base_encoder(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    base_encoder = BaseEncoder(input_channels=1)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    base_encoder.load_state_dict(encoder_state_dict)

    base_encoder.to(device)
    base_encoder.eval()
    return base_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained encoder for event classification.")
    parser.add_argument('--h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5")
    parser.add_argument('--val_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5")
    parser.add_argument('--test_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\test.h5")

    parser.add_argument('--train_labels_csv', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels\labels_train.csv")
    parser.add_argument('--val_labels_csv', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels\labels_val.csv")
    parser.add_argument('--test_labels_csv', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels\labels_test.csv")

    parser.add_argument('--model_weights_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth",
                        help='Path to the pretrained base encoder weights.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the base encoder
    base_encoder = load_base_encoder(args.model_weights_path, device=device)

    # piece together classification model
    model = FrozenEncoderBinaryClassifier(base_encoder, embedding_dim=512, num_classes=1).to(device)

    # prepare the datasets with standardisation normalization mean 0 std 1
    train_dataset = FineTuningDataset(args.h5_file_path, args.train_labels_csv)
    val_dataset = FineTuningDataset(args.val_h5_file_path, args.val_labels_csv)
    test_dataset = FineTuningDataset(args.test_h5_file_path, args.test_labels_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.num_epochs} - Training Loss: {avg_train_loss:.4f}")

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch} - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    save_path = os.path.join(os.path.dirname(args.model_weights_path), "fine_tuned_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    main()
