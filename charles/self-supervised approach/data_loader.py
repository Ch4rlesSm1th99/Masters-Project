import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


from augmentation_strategies import augment_power

class SuperDARNDataset(Dataset):
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

        # loads segmentsw
        with h5py.File(self.h5_file_path, 'r') as hf:
            self.segments = list(hf.keys())
            self.segments.sort(key=lambda x: int(x.split('_')[1]))

        # storage for augmentation params
        self.augment_params = augment_params if augment_params is not None else {}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            segment_name = self.segments[idx]
            grp = hf[segment_name]
            data = grp['data'][:]  # shape: (time_steps, range_gates, features)
            # extract power
            power_data = data[:, :, 0]

        if self.apply_augmentations:
            augmented_power_data_1 = augment_power(power_data, **self.augment_params)
            augmented_power_data_2 = augment_power(power_data, **self.augment_params)
            # arrays --> tensors
            augmented_power_data_1 = torch.from_numpy(augmented_power_data_1).float()
            augmented_power_data_2 = torch.from_numpy(augmented_power_data_2).float()
            return augmented_power_data_1, augmented_power_data_2, segment_name
        else:
            # for evaluation on raw data
            power_data_tensor = torch.from_numpy(power_data).float()
            return power_data_tensor, None, segment_name


def plot_augmented_pairs(plot_data_list, negative_value=-9999):
    """Plot pairs of augmented data for multiple segments."""
    num_segments = len(plot_data_list)
    fig, axs = plt.subplots(num_segments, 2, figsize=(10, 5 * num_segments))

    for i, data_dict in enumerate(plot_data_list):
        segment_name = data_dict['segment_name']
        augmented_data_1 = data_dict['augmented_1']
        augmented_data_2 = data_dict['augmented_2']

        power_masked = np.ma.masked_where(augmented_data_1 == negative_value, augmented_data_1)
        vmin = power_masked.min()
        vmax = power_masked.max()
        im = axs[i, 0].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f'Augmented 1 - {segment_name}')
        axs[i, 0].set_xlabel('Time Steps')
        axs[i, 0].set_ylabel('Range Gates')
        fig.colorbar(im, ax=axs[i, 0], orientation='vertical', fraction=0.046, pad=0.04)

        power_masked = np.ma.masked_where(augmented_data_2 == negative_value, augmented_data_2)
        im = axs[i, 1].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f'Augmented 2 - {segment_name}')
        axs[i, 1].set_xlabel('Time Steps')
        axs[i, 1].set_ylabel('Range Gates')
        fig.colorbar(im, ax=axs[i, 1], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def contrastive_collate_fn(batch):
    """
    Custom collate function to structure the batch correctly for contrastive learning.

    Parameters:
        batch (list): A list of tuples, each containing two augmented tensors and segment_name.

    Returns:
        batch_data: Tensor of shape [2 * batch_size, ...] if augmentations are applied,
                    or [batch_size, ...] if not.
        segment_names: List of segment names for both x_i and x_j, or just one list if no augmentations.
    """
    if batch[0][1] is not None:  # check for augs
        data_1 = [item[0] for item in batch]
        data_2 = [item[1] for item in batch]
        segment_names = [item[2] for item in batch]

        # stack data
        data_1 = torch.stack(data_1, dim=0)
        data_2 = torch.stack(data_2, dim=0)

        #concat down batch dimension
        batch_data = torch.cat([data_1, data_2], dim=0)  # shape: [2 * batch_size, ...] with [x_i,x_j]
        return batch_data, segment_names
    else:
        # no augmentations --> return only the original data
        data = [item[0] for item in batch]
        segment_names = [item[2] for item in batch]

        # stack data
        batch_data = torch.stack(data, dim=0)  # shape: [batch_size, ...] so just [x_i] or [x_j]
        return batch_data, segment_names


# test main for checking indexing and composition of batches
def main():
    h5_file_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\beam_0_selected_data.h5"

    # initial augmentation parameters
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

    batch_size = 4  # adjust batchsize here in this script we are only inspecting so 4 is fine
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn
    )

    batch_data, segment_names = next(iter(data_loader))
    print(f"Batch shape: {batch_data.shape}")  # shape dependent augmentations a
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

        # opt to plot if augmentations
        plot_data_list = []
        for idx_to_plot in range(3):
            segment_name = segment_names[idx_to_plot]
            augmented_1 = x_i[idx_to_plot].numpy()
            augmented_2 = x_j[idx_to_plot].numpy()

            plot_data_list.append({
                'segment_name': segment_name,
                'augmented_1': augmented_1,
                'augmented_2': augmented_2
            })

        plot_augmented_pairs(plot_data_list, negative_value=-9999)
    else:
        # for no augmentations --> print the shape of the original data
        print(f"Original data shape: {batch_data.shape}")
        print(f"Segment names (no augmentations): {segment_names}")



if __name__ == "__main__":
    main()
