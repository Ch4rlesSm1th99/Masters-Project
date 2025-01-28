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
        dataset for superdarn radar data suitable for contrastive learning.

        parameters:
            h5_file_path (str): path to the hdf5 file containing the data.
            negative_value (float): the padding value indicating missing data.
            apply_augmentations (bool): whether to apply augmentations.
            augment_params (dict): parameters for the augmentations.
        """
        self.h5_file_path = h5_file_path
        self.negative_value = negative_value
        self.apply_augmentations = apply_augmentations

        # open hdf5 file just to gather segment names
        with h5py.File(self.h5_file_path, 'r') as hf:
            self.segments = list(hf.keys())
            self.segments.sort(key=lambda x: int(x.split('_')[1]))

        # store augmentation params
        # if none given, empty dict -> default in augment_power
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
            # apply the augmentations twice for contrastive pairs
            augmented_power_data_1 = augment_power(power_data, **self.augment_params)
            augmented_power_data_2 = augment_power(power_data, **self.augment_params)

            # figure out which points are valid (not padded)
            valid_mask_1 = augmented_power_data_1 != self.negative_value
            valid_mask_2 = augmented_power_data_2 != self.negative_value

            # keep an unscaled copy for plotting
            augmented_power_data_1_unscaled = augmented_power_data_1.copy()
            augmented_power_data_2_unscaled = augmented_power_data_2.copy()

            # normalise the data using global mean and std
            augmented_power_data_1[valid_mask_1] = (
                augmented_power_data_1[valid_mask_1] - self.global_power_mean
            ) / self.global_power_std

            augmented_power_data_2[valid_mask_2] = (
                augmented_power_data_2[valid_mask_2] - self.global_power_mean
            ) / self.global_power_std

            # convert to torch tensors
            augmented_power_data_1_tensor = torch.from_numpy(augmented_power_data_1).float()
            augmented_power_data_2_tensor = torch.from_numpy(augmented_power_data_2).float()

            return (
                augmented_power_data_1_tensor,
                augmented_power_data_2_tensor,
                augmented_power_data_1_unscaled,
                augmented_power_data_2_unscaled,
                segment_name
            )
        else:
            # for evaluation on raw data --> no aug
            valid_mask = power_data != self.negative_value
            power_data_unscaled = power_data.copy()

            # normalise
            power_data[valid_mask] = (
                power_data[valid_mask] - self.global_power_mean
            ) / self.global_power_std

            power_data_tensor = torch.from_numpy(power_data).float()

            return (
                power_data_tensor,
                None,
                power_data_unscaled,
                None,
                segment_name
            )


def contrastive_collate_fn(batch):
    """
    custom collate function to structure the batch correctly for contrastive learning.

    parameters:
        batch (list): a list of tuples, each containing augmented tensors and segment_name.

    returns:
        batch_data: tensor of shape [2 * batch_size, ...] if augmentations are applied,
                    or [batch_size, ...] if not.
        segment_names: list of segment names for both x_i and x_j, or just one list if no augmentations.
        unnormalised_data: list of unnormalised data for plotting.
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
        batch_data = torch.cat([data_1, data_2], dim=0)  # shape: [2*batch_size, ...]
        return batch_data, segment_names, data_1_unscaled, data_2_unscaled
    else:
        # no augmentations --> return only the original data
        data = [item[0] for item in batch]
        data_unscaled = [item[2] for item in batch]
        segment_names = [item[4] for item in batch]

        batch_data = torch.stack(data, dim=0)  # shape: [batch_size, ...]
        return batch_data, segment_names, data_unscaled, None


def plot_triplet_subplots(plot_data_dict, negative_value=-9999):
    """
    show 3 columns:
     - aug 1 unnormalised
     - aug 2 unnormalised
     - aug 1 normalised

    only for one segment in this simplified version.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    segment_name = plot_data_dict['segment_name']
    # get unnormalized arrays
    aug1_unscaled = plot_data_dict['augmented_1_unscaled']
    aug2_unscaled = plot_data_dict['augmented_2_unscaled']
    # get normalized arrays
    aug1_norm = plot_data_dict['augmented_1_norm']

    # column 0: aug1 unscaled
    masked_aug1_unsc = np.ma.masked_where(aug1_unscaled == negative_value, aug1_unscaled)
    vmin_1u = masked_aug1_unsc.min()
    vmax_1u = masked_aug1_unsc.max()

    im0 = axs[0].imshow(masked_aug1_unsc.T, aspect='auto', origin='lower', cmap='viridis',
                        vmin=vmin_1u, vmax=vmax_1u)
    axs[0].set_title(f"aug1 unscaled - {segment_name}")
    axs[0].set_xlabel("time steps")
    axs[0].set_ylabel("range gates")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # column 1: aug2 unscaled
    masked_aug2_unsc = np.ma.masked_where(aug2_unscaled == negative_value, aug2_unscaled)
    vmin_2u = masked_aug2_unsc.min()
    vmax_2u = masked_aug2_unsc.max()

    im1 = axs[1].imshow(masked_aug2_unsc.T, aspect='auto', origin='lower', cmap='viridis',
                        vmin=vmin_2u, vmax=vmax_2u)
    axs[1].set_title("aug2 unscaled")
    axs[1].set_xlabel("time steps")
    axs[1].set_ylabel("range gates")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # column 2: aug1 normalized
    masked_aug1_norm = np.ma.masked_where(aug1_norm == negative_value, aug1_norm)
    vmin_1n = masked_aug1_norm.min()
    vmax_1n = masked_aug1_norm.max()

    im2 = axs[2].imshow(masked_aug1_norm.T, aspect='auto', origin='lower', cmap='viridis',
                        vmin=vmin_1n, vmax=vmax_1n)
    axs[2].set_title("aug1 normalised")
    axs[2].set_xlabel("time steps")
    axs[2].set_ylabel("range gates")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main():
    h5_file_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5"

    # example augmentation parameters
    augment_params = {
        'negative_value': -9999,
        'noise_strength': 0.05,
        'scale_range': (0.8, 1.2),
        'swap_prob': 0.1,
        'mask_prob': 0.1,
        'augment_probabilities': {
            'time_crop_resize': 1.0,
            'add_noise': 1.0,
            'scale_data': 1.0,
            'swap_adjacent_range_gates': 0.5,
            'mask_data': 0.5
        },
        # set verbose=true to see which augments are applied
        'verbose': False
    }

    dataset = SuperDARNDataset(
        h5_file_path,
        negative_value=-9999,
        apply_augmentations=True,
        augment_params=augment_params
    )

    # small batch to visualize
    batch_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn
    )

    # get one batch
    batch_data, segment_names, data_1_unscaled_list, data_2_unscaled_list = next(iter(loader))
    print(f"batch shape: {batch_data.shape}")
    print(f"segment names: {segment_names}")

    # if we have augmentations, x_i and x_j are stacked
    x_i, x_j = torch.chunk(batch_data, 2, dim=0)
    print(f"x_i shape: {x_i.shape}, x_j shape: {x_j.shape}")

    # we'll just plot the first sample in the batch
    idx = 0
    seg_name = segment_names[idx]
    aug1_norm_array = x_i[idx].numpy()
    aug2_norm_array = x_j[idx].numpy()

    aug1_unscaled = data_1_unscaled_list[idx]
    aug2_unscaled = data_2_unscaled_list[idx]

    # build dictionary for single-segment plotting
    single_plot_data = {
        'segment_name': seg_name,
        'augmented_1_unscaled': aug1_unscaled,
        'augmented_2_unscaled': aug2_unscaled,
        'augmented_1_norm': aug1_norm_array
    }

    # now plot just that single segment in a 1x3 layout
    plot_triplet_subplots(single_plot_data, negative_value=-9999)


if __name__ == "__main__":
    main()
