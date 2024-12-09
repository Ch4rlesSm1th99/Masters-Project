import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse

from data_loader import SuperDARNDataset, contrastive_collate_fn
from models import BaseEncoder, SimCLR, SimCLR_topk_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a SimCLR model with configurable parameters.")

    parser.add_argument('--train_h5_file_path', type=str, default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5", help='Path to training H5 file.')
    parser.add_argument('--val_h5_file_path', type=str, default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5", help='Path to validation H5 file.')
    parser.add_argument('--checkpoint_dir', type=str, default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details\SimCLR", help='Directory to save checkpoints and logs.')

    # train params
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for NT-Xent loss.')
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension for SimCLR.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')

    # aug params
    parser.add_argument('--noise_strength', type=float, default=0.02, help='Noise strength for augmentations.')
    parser.add_argument('--scale_min', type=float, default=0.95, help='Minimum scale factor for scaling augmentation.')
    parser.add_argument('--scale_max', type=float, default=1.05, help='Maximum scale factor for scaling augmentation.')
    parser.add_argument('--max_shift', type=int, default=2, help='Max shift for translation augmentations.')
    parser.add_argument('--max_removed_points', type=int, default=25, help='Max number of points removed for augmentation.')
    parser.add_argument('--swap_prob', type=float, default=0.1, help='Probability of swapping adjacent range gates.')
    parser.add_argument('--mask_prob', type=float, default=0.1, help='Probability of masking data.')
    # probability of applying certain augs
    parser.add_argument('--translate_y_prob', type=float, default=0.5, help='Probability of translating data along Y-axis.')
    parser.add_argument('--translate_x_prob', type=float, default=0.5, help='Probability of translating data along X-axis.')
    parser.add_argument('--swap_adj_prob', type=float, default=0.5, help='Probability of swapping adjacent gates.')
    parser.add_argument('--mask_data_prob', type=float, default=0.5, help='Probability of masking data.')

    return parser.parse_args()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    log_file_path = os.path.join(args.checkpoint_dir, "training.log")
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        format='%(message)s',
        level=logging.INFO
    )
    logging.info("epoch,train_loss,val_loss,val_top1,val_top5,val_top10")

    best_val_loss = float('inf')        # early convergence checks
    epochs_without_improvement = 0

    augment_params = {
        'negative_value': -9999,
        'noise_strength': args.noise_strength,
        'scale_range': (args.scale_min, args.scale_max),
        'max_shift': args.max_shift,
        'max_removed_points': args.max_removed_points,
        'swap_prob': args.swap_prob,
        'mask_prob': args.mask_prob,
        'augment_probabilities': {
            'add_noise': 1.0,
            'scale_data': 1.0,
            'translate_y': args.translate_y_prob,
            'translate_x': args.translate_x_prob,
            'swap_adjacent_range_gates': args.swap_adj_prob,
            'mask_data': args.mask_data_prob
        },
        'verbose': False
    }

    train_dataset = SuperDARNDataset(
        args.train_h5_file_path,
        negative_value=-9999,
        apply_augmentations=True,
        augment_params=augment_params
    )

    val_dataset = SuperDARNDataset(
        args.val_h5_file_path,
        negative_value=-9999,
        apply_augmentations=True,
        augment_params=augment_params
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn,
        num_workers=4,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=contrastive_collate_fn,
        num_workers=4,
        drop_last=True
    )

    base_encoder = BaseEncoder(input_channels=1)
    model = SimCLR(
        base_encoder=base_encoder,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        device=device
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_accuracy_top1, val_accuracy_top5, val_accuracy_top10 = validate(model, val_loader, device, epoch)

        scheduler.step()

        logging.info(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_accuracy_top1:.2f},{val_accuracy_top5:.2f},{val_accuracy_top10:.2f}")

        # check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, filename="best_model.pth")
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == args.num_epochs:
            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir)

        if epochs_without_improvement >= args.patience:
            break


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_accuracy = 0
    count = 0

    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", leave=False) as pbar:
        for batch_data, _, _, _ in data_loader:
            x_i, x_j = torch.chunk(batch_data, 2, dim=0)
            x_i = x_i.unsqueeze(1).to(device)
            x_j = x_j.unsqueeze(1).to(device)

            z_i = model(x_i)  # forward
            z_j = model(x_j)

            loss = model.compute_loss(z_i, z_j)
            accuracy = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature, top_k=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy
            count += 1

            pbar.update(1)

    average_loss = total_loss / count
    average_accuracy = total_accuracy / count
    return average_loss, average_accuracy


def validate(model, data_loader, device, epoch):
    model.eval()
    total_loss = 0
    total_acc_top1 = 0
    total_acc_top5 = 0
    total_acc_top10 = 0
    count = 0

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", leave=False) as pbar:
            for batch_data, _, _, _ in data_loader:
                x_i, x_j = torch.chunk(batch_data, 2, dim=0)
                x_i = x_i.unsqueeze(1).to(device)
                x_j = x_j.unsqueeze(1).to(device)

                z_i = model(x_i)
                z_j = model(x_j)

                loss = model.compute_loss(z_i, z_j)
                acc_top1 = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature, top_k=1)
                acc_top5 = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature, top_k=min(5, 2*data_loader.batch_size))
                acc_top10 = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature, top_k=min(10, 2*data_loader.batch_size))

                total_loss += loss.item()
                total_acc_top1 += acc_top1
                total_acc_top5 += acc_top5
                total_acc_top10 += acc_top10
                count += 1

                pbar.update(1)

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    average_loss = total_loss / count
    average_acc_top1 = total_acc_top1 / count
    average_acc_top5 = total_acc_top5 / count
    average_acc_top10 = total_acc_top10 / count

    return average_loss, average_acc_top1, average_acc_top5, average_acc_top10


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, filename=None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if filename is None:
        filename = f"simclr_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
