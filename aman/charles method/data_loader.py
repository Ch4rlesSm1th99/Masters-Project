import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from augmentation_strategies import augment_power

class SuperDARNDataset(Dataset):
    # global mean and standard deviation for power (from data_inspection.py output)
    global_power_mean = 15.023935
    global_power_std = 9.644889

    def __init__(self, h5_file_path, negative_value=-9999, apply_augmentations=True, augment_params=None):
        """
        Dataset for SuperDARN radar data suitable for contrastive learning.

        Parameters:
            h5_file_path (str): Path to the HDF5 file containing the data.
            negative_value (float): The padding value indicating missing data.
            apply_augmentations (bool): Whether to apply augmentations.
            augment_params (dict): Parameters for the augmentations.
        """
        self.h5_file_path = h5_file_path
        self.negative_value = negative_value
        self.apply_augmentations = apply_augmentations

        with h5py.File(self.h5_file_path, 'r') as hf:
            self.segments = list(hf.keys())
            self.segments.sort(key=lambda x: int(x.split('_')[1]))

        self.augment_params = augment_params if augment_params is not None else {}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            segment_name = self.segments[idx]
            grp = hf[segment_name]
            data = grp['data'][:]  # shape: (time_steps, range_gates, features)
            # extract power data
            power_data = data[:, :, 0]

        if self.apply_augmentations:
            augmented_power_data_1 = augment_power(power_data, **self.augment_params)
            augmented_power_data_2 = augment_power(power_data, **self.augment_params)

            # normalise mask for padding value
            valid_mask_1 = augmented_power_data_1 != self.negative_value
            valid_mask_2 = augmented_power_data_2 != self.negative_value

            '''print(
                f"Before normalisation: Aug1 min={augmented_power_data_1[valid_mask_1].min()}, max={augmented_power_data_1[valid_mask_1].max()}")
            print(
                f"Before normalisation: Aug2 min={augmented_power_data_2[valid_mask_2].min()}, max={augmented_power_data_2[valid_mask_2].max()}")'''

            augmented_power_data_1_unscaled = augmented_power_data_1.copy()     # copies for plotting and comparison
            augmented_power_data_2_unscaled = augmented_power_data_2.copy()

            # normalise the data using global mean and std
            augmented_power_data_1[valid_mask_1] = (
                augmented_power_data_1[valid_mask_1] - self.global_power_mean
            ) / self.global_power_std

            augmented_power_data_2[valid_mask_2] = (
                augmented_power_data_2[valid_mask_2] - self.global_power_mean
            ) / self.global_power_std

            mean_aug1 = np.mean(augmented_power_data_1[valid_mask_1])
            std_aug1 = np.std(augmented_power_data_1[valid_mask_1])
            mean_aug2 = np.mean(augmented_power_data_2[valid_mask_2])
            std_aug2 = np.std(augmented_power_data_2[valid_mask_2])
            '''print(f"After normalisation: Aug1 mean={mean_aug1:.4f}, std={std_aug1:.4f}")
            print(f"After normalisation: Aug2 mean={mean_aug2:.4f}, std={std_aug2:.4f}")'''

            augmented_power_data_1_tensor = torch.from_numpy(augmented_power_data_1).float()
            augmented_power_data_2_tensor = torch.from_numpy(augmented_power_data_2).float()
            return (augmented_power_data_1_tensor, augmented_power_data_2_tensor,
                    augmented_power_data_1_unscaled, augmented_power_data_2_unscaled,
                    segment_name)
        else:
            # for evaluation on raw data --> no aug
            valid_mask = power_data != self.negative_value
            power_data_unscaled = power_data.copy()

            power_data[valid_mask] = (
                power_data[valid_mask] - self.global_power_mean
            ) / self.global_power_std

            mean_power = np.mean(power_data[valid_mask])
            std_power = np.std(power_data[valid_mask])
            '''print(f"After normalisation: mean={mean_power:.4f}, std={std_power:.4f}")'''

            power_data_tensor = torch.from_numpy(power_data).float()
            return power_data_tensor, None, power_data_unscaled, None, segment_name


def plot_augmented_pairs(plot_data_list, negative_value=-9999):
    """Plot pairs of augmented data for multiple segments, comparing normalised and un-normalised data."""
    num_segments = len(plot_data_list)
    fig, axs = plt.subplots(num_segments, 4, figsize=(20, 5 * num_segments))

    for i, data_dict in enumerate(plot_data_list):
        segment_name = data_dict['segment_name']
        # normalised data
        augmented_data_1_norm = data_dict['augmented_1_norm']
        augmented_data_2_norm = data_dict['augmented_2_norm']
        # un-normalised data
        augmented_data_1_unscaled = data_dict['augmented_1_unscaled']
        augmented_data_2_unscaled = data_dict['augmented_2_unscaled']

        # plot aug 1 normalised
        power_masked_norm_1 = np.ma.masked_where(augmented_data_1_norm == negative_value, augmented_data_1_norm)
        vmin_norm = power_masked_norm_1.min()
        vmax_norm = power_masked_norm_1.max()
        im1 = axs[i, 0].imshow(power_masked_norm_1.T, aspect='auto', origin='lower', cmap='viridis',
                               vmin=vmin_norm, vmax=vmax_norm)
        axs[i, 0].set_title(f'Augmented 1 Normalised - {segment_name}')
        axs[i, 0].set_xlabel('Time Steps')
        axs[i, 0].set_ylabel('Range Gates')
        fig.colorbar(im1, ax=axs[i, 0], orientation='vertical', fraction=0.046, pad=0.04)

        # plot aug 2 normalised
        power_masked_norm_2 = np.ma.masked_where(augmented_data_2_norm == negative_value, augmented_data_2_norm)
        im2 = axs[i, 1].imshow(power_masked_norm_2.T, aspect='auto', origin='lower', cmap='viridis',
                               vmin=vmin_norm, vmax=vmax_norm)
        axs[i, 1].set_title(f'Augmented 2 Normalised - {segment_name}')
        axs[i, 1].set_xlabel('Time Steps')
        axs[i, 1].set_ylabel('Range Gates')
        fig.colorbar(im2, ax=axs[i, 1], orientation='vertical', fraction=0.046, pad=0.04)

        # plot aug 1 un normalised
        power_masked_unscaled_1 = np.ma.masked_where(augmented_data_1_unscaled == negative_value, augmented_data_1_unscaled)
        vmin_unscaled = power_masked_unscaled_1.min()
        vmax_unscaled = power_masked_unscaled_1.max()
        im3 = axs[i, 2].imshow(power_masked_unscaled_1.T, aspect='auto', origin='lower', cmap='viridis',
                               vmin=vmin_unscaled, vmax=vmax_unscaled)
        axs[i, 2].set_title(f'Augmented 1 Un-normalised - {segment_name}')
        axs[i, 2].set_xlabel('Time Steps')
        axs[i, 2].set_ylabel('Range Gates')
        fig.colorbar(im3, ax=axs[i, 2], orientation='vertical', fraction=0.046, pad=0.04)

        # plot aug 2 un normalised
        power_masked_unscaled_2 = np.ma.masked_where(augmented_data_2_unscaled == negative_value, augmented_data_2_unscaled)
        im4 = axs[i, 3].imshow(power_masked_unscaled_2.T, aspect='auto', origin='lower', cmap='viridis',
                               vmin=vmin_unscaled, vmax=vmax_unscaled)
        axs[i, 3].set_title(f'Augmented 2 Un-normalised - {segment_name}')
        axs[i, 3].set_xlabel('Time Steps')
        axs[i, 3].set_ylabel('Range Gates')
        fig.colorbar(im4, ax=axs[i, 3], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def contrastive_collate_fn(batch):
    """
    Custom collate function to structure the batch correctly for contrastive learning.

    Parameters:
        batch (list): A list of tuples, each containing augmented tensors and segment_name.

    Returns:
        batch_data: Tensor of shape [2 * batch_size, ...] if augmentations are applied,
                    or [batch_size, ...] if not.
        segment_names: List of segment names for both x_i and x_j, or just one list if no augmentations.
        unnormalised_data: List of unnormalised data for plotting.
    """
    if batch[0][1] is not None:  # check for applied augmentations
        data_1 = [item[0] for item in batch]
        data_2 = [item[1] for item in batch]
        data_1_unscaled = [item[2] for item in batch]
        data_2_unscaled = [item[3] for item in batch]
        segment_names = [item[4] for item in batch]

        data_1 = torch.stack(data_1, dim=0)
        data_2 = torch.stack(data_2, dim=0)

        # concat down batch dim
        batch_data = torch.cat([data_1, data_2], dim=0)  # shape: [2 * batch_size, ...] with [x_i, x_j]
        return batch_data, segment_names, data_1_unscaled, data_2_unscaled
    else:
        # No augmentations --> return only the original data
        data = [item[0] for item in batch]
        data_unscaled = [item[2] for item in batch]
        segment_names = [item[4] for item in batch]

        # Stack data
        batch_data = torch.stack(data, dim=0)  # shape: [batch_size, ...] so just [x_i] + [x_j]
        return batch_data, segment_names, data_unscaled, None


# checking composition of batch and normalisation processes
def main():
    h5_file_path = r"C:\Users\aman\Desktop\MPhys Data\Data\train.h5"

    # augmentation chances
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
        'verbose': False  # true for checking applied augmentations
    }

    dataset = SuperDARNDataset(
        h5_file_path,
        negative_value=-9999,
        apply_augmentations=True,  # set to False to test no augmentations
        augment_params=augment_params
    )

    batch_size = 4  # adjust batch size here; in this script we are only inspecting so 4 is fine
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn
    )

    batch = next(iter(data_loader))
    batch_data, segment_names, data_1_unscaled_list, data_2_unscaled_list = batch
    print(f"Batch shape: {batch_data.shape}")  # shape depends on augmentations
    print(f"Segment names: {segment_names}")

    if dataset.apply_augmentations:
        # split the batch into x_i and x_j and check segment names
        x_i, x_j = torch.chunk(batch_data, 2, dim=0)

        print(f"x_i shape: {x_i.shape}")
        print(f"x_j shape: {x_j.shape}")

        segment_names_x_i = segment_names
        segment_names_x_j = segment_names
        print(f"x_i segment names: {segment_names_x_i}")
        print(f"x_j segment names: {segment_names_x_j}")


        plot_data_list = []
        for idx_to_plot in range(min(3, len(segment_names))):
            segment_name = segment_names[idx_to_plot]
            augmented_1_norm = x_i[idx_to_plot].numpy()
            augmented_2_norm = x_j[idx_to_plot].numpy()

            # collect unormalised data for checks
            augmented_1_unscaled = data_1_unscaled_list[idx_to_plot]
            augmented_2_unscaled = data_2_unscaled_list[idx_to_plot]

            plot_data_list.append({
                'segment_name': segment_name,
                'augmented_1_norm': augmented_1_norm,
                'augmented_2_norm': augmented_2_norm,
                'augmented_1_unscaled': augmented_1_unscaled,
                'augmented_2_unscaled': augmented_2_unscaled
            })

        plot_augmented_pairs(plot_data_list, negative_value=-9999)
    else:
        #  no augmentations --> print shape of original
        print(f"Original data shape: {batch_data.shape}")
        print(f"Segment names (no augmentations): {segment_names}")


if __name__ == "__main__":
    main()
