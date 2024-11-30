import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import SuperDARNDataset, contrastive_collate_fn
from models import BaseEncoder, SimCLR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_h5_file_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5"
    val_h5_file_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5"

    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-3
    temperature = 0.5
    projection_dim = 128
    checkpoint_dir = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\checkpoints"

    augment_params = {
        'negative_value': -9999,
        'noise_strength': 0.02,
        'scale_range': (0.95, 1.05),
        'max_shift': 2,
        'max_removed_points': 25,
        'swap_prob': 0.1,
        'mask_prob': 0.1,
        'augment_probabilities': {
            'add_noise': 1.0,  # always apply
            'scale_data': 1.0,  # always apply
            'translate_y': 0.5,  # 50% chance
            'translate_x': 0.5,  # 50% chance
            'swap_adjacent_range_gates': 0.5,  # 50% chance
            'mask_data': 0.5   # 50% chance
        },
        'verbose': False
    }

    # Create datasets and data loaders
    train_dataset = SuperDARNDataset(
        train_h5_file_path,
        negative_value=-9999,
        apply_augmentations=True,
        augment_params=augment_params
    )

    val_dataset = SuperDARNDataset(
        val_h5_file_path,
        negative_value=-9999,
        apply_augmentations=True,
        augment_params=augment_params
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn,
        num_workers=4,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=contrastive_collate_fn,
        num_workers=4,
        drop_last=True
    )

    base_encoder = BaseEncoder(input_channels=1)
    model = SimCLR(
        base_encoder=base_encoder,
        projection_dim=projection_dim,
        temperature=temperature,
        device=device
    )
    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # training
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # save checkpoint
        if epoch % 10 == 0 or epoch == num_epochs:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (batch_data, _, _, _) in enumerate(data_loader):
        x_i, x_j = torch.chunk(batch_data, 2, dim=0)
        x_i = x_i.unsqueeze(1).to(device)
        x_j = x_j.unsqueeze(1).to(device)

        z_i = model(x_i)        # forward
        z_j = model(x_j)

        loss = model.compute_loss(z_i, z_j)

        optimizer.zero_grad()       # back
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    average_loss = total_loss / len(data_loader)
    return average_loss

def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data, _, _, _ in data_loader:
            x_i, x_j = torch.chunk(batch_data, 2, dim=0)
            x_i = x_i.unsqueeze(1).to(device)
            x_j = x_j.unsqueeze(1).to(device)

            z_i = model(x_i)        # forward
            z_j = model(x_j)

            loss = model.compute_loss(z_i, z_j)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"simclr_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
