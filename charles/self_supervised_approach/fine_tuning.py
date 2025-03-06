import os
import csv
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse


from charles.self_supervised_approach.models import FrozenEncoderBinaryClassifier


def load_model(path, device="cpu"):
    from charles.self_supervised_approach.models import BaseEncoder
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


class FineTuningDataset(Dataset):
    """
    Loads segments from an H5 file and their binary labels from a CSV.
    The CSV must have columns: "index", "segment_name", "label".
    Only segments that appear in the CSV are returned.
    """
    def __init__(self, h5_file_path, labels_csv, negative_value=-9999, transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.negative_value = negative_value

        with h5py.File(self.h5_file_path, 'r') as hf:
            self.segments = list(hf.keys())
            self.segments.sort(key=lambda x: int(x.split('_')[-1]))

        # map segment_name -> label from CSV.
        self.labels = {}
        with open(labels_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.labels[row["segment_name"]] = int(row["label"])

        # Filter segments to only those with labels.
        self.segments = [seg for seg in self.segments if seg in self.labels]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg_name = self.segments[idx]
        with h5py.File(self.h5_file_path, 'r') as hf:
            data = hf[seg_name]['data'][:]  # shape: (time_steps, range_gates, features)
            power_data = data[:, :, 0].astype(np.float32)
        if self.transform:
            power_data = self.transform(power_data)
        data_tensor = torch.from_numpy(power_data).unsqueeze(0)  # shape: [1, H, W]
        label = self.labels[seg_name]
        return data_tensor, label


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
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained encoder for event classification.")
    parser.add_argument('--run_set', type=str, default='train', choices=['train', 'val', 'test'],
                        help="Which dataset to fine-tune on: train, val, or test.")
    parser.add_argument('--h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5",
                        help='Path to the H5 file for training.')
    parser.add_argument('--val_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5",
                        help='Path to the H5 file for validation.')
    parser.add_argument('--test_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\test.h5",
                        help='Path to the H5 file for testing.')
    parser.add_argument('--train_labels_csv', type=str, default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels\labels_train.csv",
                        help='Path to the CSV file containing training labels.')
    parser.add_argument('--val_labels_csv', type=str, default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels\labels_val.csv",
                        help='Path to the CSV file containing validation labels.')
    parser.add_argument('--test_labels_csv', type=str, default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\fine_tuning_labels\labels_test.csv",
                        help='Path to the CSV file containing test labels.')
    parser.add_argument('--model_weights_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR_Weights\best_model.pth",
                        help='Path to the pretrained encoder weights.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of fine-tuning epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for fine-tuning.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_model(args.model_weights_path, device)
    model = FrozenEncoderBinaryClassifier(encoder, embedding_dim=512).to(device)

    train_dataset = FineTuningDataset(args.h5_file_path, args.train_labels_csv, negative_value=-9999)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    val_dataset = FineTuningDataset(args.val_h5_file_path, args.val_labels_csv, negative_value=-9999)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    test_dataset = FineTuningDataset(args.test_h5_file_path, args.test_labels_csv, negative_value=-9999)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Training loop.
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

        # eval on validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # eval on test
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch} - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Save the fine-tuned classifier.
    save_path = os.path.join(os.path.dirname(args.model_weights_path), "fine_tuned_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    main()
