import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse

from charles.self_supervised_approach.data_loader import SuperDARNDataset, contrastive_collate_fn
from charles.self_supervised_approach.models import BaseEncoder, SimCLR, BYOL, SimCLR_topk_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SimCLR/BYOL model with configurable parameters.")

    # Model Type
    parser.add_argument('--model_type', type=str, default='SimCLR', choices=['SimCLR', 'BYOL'],
                        help='Which model to train: SimCLR or BYOL.')

    # data paths
    parser.add_argument('--train_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5",
                        help='Path to training H5 file.')
    parser.add_argument('--val_h5_file_path', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\val.h5",
                        help='Path to validation H5 file.')

    # model save path and logging path
    parser.add_argument('--weights_dir', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details",
                        help="Base directory to save model checkpoints.")
    parser.add_argument('--log_dir', type=str,
                        default=r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\model_details",
                        help="Base directory to save logs.")


    # training params
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for NT-Xent loss (SimCLR).')
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')

    # BYOL- params
    parser.add_argument('--moving_avg_decay', type=float, default=0.99,
                        help='Exponential moving average decay for BYOL target network.')

    # augmentation params
    parser.add_argument('--noise_strength', type=float, default=0.02, help='Strength of noise augmentation.')
    parser.add_argument('--scale_min', type=float, default=0.95, help='Minimum scale factor for scaling augmentation.')
    parser.add_argument('--scale_max', type=float, default=1.05, help='Maximum scale factor for scaling augmentation.')

    # time cropping augmentation limits and probs
    parser.add_argument('--time_crop_min', type=float, default=0.7,
                        help='Minimum fraction of the time range to keep during cropping.')
    parser.add_argument('--time_crop_max', type=float, default=1.0,
                        help='Maximum fraction of the time range to keep during cropping.')
    parser.add_argument('--min_valid_ratio', type=float, default=0.3,
                        help='Minimum proportion of valid (non-missing) data required after cropping.')

    # swapping and masking params and probs
    parser.add_argument('--swap_intensity', type=float, default=0.1,
                        help='Proportion of valid data points that are swapped.')
    parser.add_argument('--mask_intensity', type=float, default=0.1,
                        help='Proportion of valid data points that are masked.')

    # Probabilities of Applying Augmentations
    parser.add_argument('--time_crop_prob', type=float, default=1.0,
                        help='Probability of applying time cropping.')
    parser.add_argument('--add_noise_prob', type=float, default=1.0,
                        help='Probability of applying noise augmentation.')
    parser.add_argument('--scale_data_prob', type=float, default=1.0,
                        help='Probability of applying scaling augmentation.')
    parser.add_argument('--swap_adj_prob', type=float, default=0.5,
                        help='Probability of swapping adjacent gates.')
    parser.add_argument('--mask_data_prob', type=float, default=0.0,
                        help='Probability of applying data masking.')

    return parser.parse_args()



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use seperate logs in model dets for logging weights and training info
    model_dir = os.path.join(args.weights_dir, args.model_type)
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(args.log_dir, args.model_type, "logs")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    checkpoint_file = os.path.join(weights_dir, "model.pth")
    log_file = os.path.join(logs_dir, "training.log")

    # note: overites each time model is trained, currently that how its built
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info("epoch,train_loss,val_loss,val_top1,val_top5,val_top10")

    print(f"Training {args.model_type} model...")
    print(f"Saving model to: {checkpoint_file}")
    print(f"Logging training to: {log_file}")

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # aug params
    augment_params = {
        'negative_value': -9999,
        'noise_strength': args.noise_strength,
        'scale_range': (args.scale_min, args.scale_max),
        'swap_prob': args.swap_intensity,
        'mask_prob': args.mask_intensity,

        # time cropping specific
        'time_crop_frac_range': (args.time_crop_min, args.time_crop_max),
        'min_valid_ratio': args.min_valid_ratio,

        'augment_probabilities': {
            'time_crop_resize': args.time_crop_prob,
            'add_noise': args.add_noise_prob,
            'scale_data': args.scale_data_prob,
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

    base_encoder = BaseEncoder(input_channels=1).to(device)

    if args.model_type.upper() == "SIMCLR":
        model = SimCLR(
            base_encoder=base_encoder,
            projection_dim=args.projection_dim,
            temperature=args.temperature,
            device=device
        )
    elif args.model_type.upper() == "BYOL":
        model = BYOL(
            base_encoder=base_encoder,
            projection_dim=args.projection_dim,
            moving_avg_decay=args.moving_avg_decay,
            device=device
        )
    else:
        raise ValueError("Unknown model_type. Choose from ['SimCLR', 'BYOL'].")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # ----------------------------------
    # training loop here
    # ----------------------------------
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, epoch, args)

        val_loss, val_top1, val_top5, val_top10 = validate(model, val_loader, device, epoch, args)

        logging.info(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_top1:.2f},{val_top5:.2f},{val_top10:.2f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, weights_dir, filename="best_model.pth")
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == args.num_epochs:
            save_checkpoint(model, optimizer, epoch, weights_dir, filename=f"epoch_{epoch}.pth")

        # early stopping
        if epochs_without_improvement >= args.patience:
            logging.info(f"Early stopping at epoch {epoch}.")
            break



def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    """
    If model is SimCLR:
       - do the standard z_i, z_j forward + NT-Xent loss
    If model is BYOL:
       - do the p1, t1, p2, t2 forward + BYOL loss
       - Then call model.update_target_network()
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    count = 0

    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", leave=False) as pbar:
        for batch_data, _, _, _ in data_loader:
            # batch_data has 2*N examples in it due to contrastive_collate_fn
            x_i, x_j = torch.chunk(batch_data, 2, dim=0)

            # + channel dim
            x_i = x_i.unsqueeze(1).to(device)
            x_j = x_j.unsqueeze(1).to(device)

            optimizer.zero_grad()

            # -------------------------------------------------
            #  SIMCLR acc
            # -------------------------------------------------
            if isinstance(model, SimCLR):
                z_i = model(x_i)
                z_j = model(x_j)
                loss = model.compute_loss(z_i, z_j)
                accuracy = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature, top_k=1)

            # -------------------------------------------------
            #  BYOL acc
            # -------------------------------------------------
            elif isinstance(model, BYOL):
                loss = model.compute_loss(x_i, x_j)
                # For BYOL, top-k accuracy is less standard.
                with torch.no_grad():
                    p1, t1, p2, t2 = model.forward(x_i, x_j)
                    # p1, p2 equivelant to zi, zj for simclr
                    accuracy = SimCLR_topk_accuracy(p1, p2, temperature=0.5, top_k=1)

            loss.backward()
            optimizer.step()

            # for BYOL, target network updated with moment values each epoch
            if isinstance(model, BYOL):
                model.update_target_network()

            total_loss += loss.item()
            total_accuracy += accuracy
            count += 1
            pbar.update(1)

    average_loss = total_loss / count
    average_accuracy = total_accuracy / count
    return average_loss, average_accuracy


def validate(model, data_loader, device, epoch, args):
    model.eval()
    total_loss = 0.0
    total_acc_top1 = 0.0
    total_acc_top5 = 0.0
    total_acc_top10 = 0.0
    count = 0

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", leave=False) as pbar:
            for batch_data, _, _, _ in data_loader:
                x_i, x_j = torch.chunk(batch_data, 2, dim=0)
                x_i = x_i.unsqueeze(1).to(device)
                x_j = x_j.unsqueeze(1).to(device)

                # -------------------------------------------
                # Validation for SimCLR
                # -------------------------------------------
                if isinstance(model, SimCLR):
                    z_i = model(x_i)
                    z_j = model(x_j)
                    loss = model.compute_loss(z_i, z_j)

                    acc_top1 = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature, top_k=1)
                    acc_top5 = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature,
                                                    top_k=min(5, 2 * data_loader.batch_size))
                    acc_top10 = SimCLR_topk_accuracy(z_i, z_j, temperature=model.temperature,
                                                     top_k=min(10, 2 * data_loader.batch_size))

                # -------------------------------------------
                # Validation for BYOL
                # -------------------------------------------
                elif isinstance(model, BYOL):
                    loss = model.compute_loss(x_i, x_j)
                    p1, t1, p2, t2 = model.forward(x_i, x_j)

                    acc_top1 = SimCLR_topk_accuracy(p1, p2, temperature=0.5,
                                                    top_k=1)
                    acc_top5 = SimCLR_topk_accuracy(p1, p2, temperature=0.5,
                                                    top_k=min(5, 2 * data_loader.batch_size))
                    acc_top10 = SimCLR_topk_accuracy(p1, p2, temperature=0.5,
                                                     top_k=min(10, 2 * data_loader.batch_size))

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


def save_checkpoint(model, optimizer, epoch, weights_dir, filename=None):
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if filename is None:
        filename = f"simclr_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(weights_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
