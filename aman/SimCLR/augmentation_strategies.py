#%%
import numpy as np

def augment_power(
    data,
    negative_value=-9999,
    noise_strength=0.2,
    scale_range=(0.9, 1.1),
    max_shift=2,
    max_removed_points=25,
    swap_prob=0.1,
    mask_prob=0.1,
    saturation_factor=1.5,
    augment_probabilities=None,
    verbose=False
):
    """
    Apply augmentations to the power data.

    Parameters:
        data (np.ndarray): The power data array of shape (time_steps, range_gates).
        negative_value (float): The padding value indicating missing data.
        noise_strength (float): Standard deviation of Gaussian noise to add.
        scale_range (tuple): Range (min, max) for scaling factor.
        max_shift (int): Maximum shift for translation augmentations.
        max_removed_points (int): Maximum allowed removed points for translations.
        swap_prob (float): Probability of swapping adjacent range gates.
        mask_prob (float): Probability of masking data points.
        augment_probabilities (dict): Probabilities for each augmentation. Keys are:
            'add_noise', 'scale_data', 'translate_y', 'translate_x',
            'swap_adjacent_range_gates', 'mask_data'.
            Values are floats between 0 and 1.
        verbose (bool): If True, prints messages during augmentation.
        saturation_factor (float): Factor for adjusting saturation.

    Returns:
        np.ndarray: The augmented data array.
    """

    data_power = data.copy()

    # Default probabilities if not specfied when called
    if augment_probabilities is None:
        augment_probabilities = {
            'add_noise': 0.5,
            'scale_data': 0.5,
            'translate_y': 0.5,
            'translate_x': 0.5,
            'swap_adjacent_range_gates': 0.5,
            'mask_data': 0.5,
            'adjust_saturation': 0.5
        }

    def add_noise(data):
        valid_mask = data != negative_value
        if np.any(valid_mask):  # Check if there are valid data points
            data_range = data[valid_mask].max() - data[valid_mask].min()
            dynamic_noise_strength = noise_strength * data_range
            noise = np.random.normal(0, dynamic_noise_strength, size=data.shape)
            data[valid_mask] += noise[valid_mask]
        return data

    def scale_data_func(data):
        valid_mask = data != negative_value
        scale_factor = np.random.uniform(*scale_range)
        data[valid_mask] *= scale_factor
        return data

    def translate_y(data):
        attempt = 0
        while attempt < 3:  # limit the number of attempts
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift == 0:
                return data  # no shift

            num_time_steps, num_range_gates = data.shape
            shifted_data = np.full_like(data, negative_value)

            if shift > 0:
                # shift data up (towards higher range gates)
                shifted_data[:, shift:] = data[:, :-shift]
            else:
                # shift data down (towards lower range gates)
                shifted_data[:, :num_range_gates + shift] = data[:, -shift:]

            removed_points = np.sum((data != negative_value) & (shifted_data == negative_value))
            if removed_points <= max_removed_points:
                if verbose:
                    print(f"Applied translate_y with shift {shift}")
                return shifted_data  # successful translation
            attempt += 1

        if verbose:
            print(f"Translation in Y aborted after {attempt} attempts: {removed_points} points would be removed.")
        return data  # original data

    def translate_x(data):
        attempt = 0
        while attempt < 3:  # limit the number of attempts without removing too much data
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift == 0:
                return data  # no shift

            num_time_steps, num_range_gates = data.shape
            shifted_data = np.full_like(data, negative_value)

            if shift > 0:
                # shift data forward in time
                shifted_data[shift:, :] = data[:-shift, :]
            else:
                # shift data backward in time
                shifted_data[:num_time_steps + shift, :] = data[-shift:, :]

            removed_points = np.sum((data != negative_value) & (shifted_data == negative_value))
            if removed_points <= max_removed_points:
                if verbose:
                    print(f"Applied translate_x with shift {shift}")
                return shifted_data  # Successful translation
            attempt += 1

        if verbose:
            print(f"Translation in X aborted after {attempt} attempts: {removed_points} points would be removed.")
        return data  # original data

    def swap_adjacent_range_gates(data):
        num_time_steps, num_range_gates = data.shape
        for t in range(num_time_steps):
            for r in range(num_range_gates - 1):
                if np.random.rand() < swap_prob:
                    if (data[t, r] != negative_value) and (data[t, r + 1] != negative_value):
                        data[t, r], data[t, r + 1] = data[t, r + 1], data[t, r]
        if verbose:
            print(f"Applied swap_adjacent_range_gates with swap_prob {swap_prob}")
        return data

    def mask_data(data):
        num_time_steps, num_range_gates = data.shape
        mask = (np.random.rand(num_time_steps, num_range_gates) < mask_prob) & (data != negative_value)
        data[mask] = negative_value
        if verbose:
            print(f"Applied mask_data with mask_prob {mask_prob}")
        return data
    
    
    def adjust_saturation(data):
        valid_mask = data != negative_value
        mean_value = np.mean(data[valid_mask])
        data[valid_mask] = mean_value + saturation_factor * (data[valid_mask] - mean_value)
        if verbose:
            print(f"Applied adjust_saturation with saturation_factor {saturation_factor}")
        return data

    # map augs to funct
    augmentations = {
        'add_noise': add_noise,
        'scale_data': scale_data_func,
        'translate_y': translate_y,
        'translate_x': translate_x,
        'swap_adjacent_range_gates': swap_adjacent_range_gates,
        'mask_data': mask_data,
        'adjust_saturation': adjust_saturation
    }

    # apply aug
    for aug_name, aug_func in augmentations.items():
        prob = augment_probabilities.get(aug_name, 0)
        if np.random.rand() < prob:
            data_power = aug_func(data_power)
            if verbose and aug_name not in ['translate_y', 'translate_x']:
                print(f"Applied augmentation: {aug_name}")
        else:
            if verbose:
                print(f"Skipped augmentation: {aug_name}")

    return data_power


import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random


h5_file_path = r"C:\Users\aman\Desktop\MPhys Data\Data\all_beams_selected_data.h5"

# negative padding value used in your data
negative_value = -9999


def main():
    with h5py.File(h5_file_path, 'r') as hf:
        beams = list(hf.keys())
        #segments.sort(key=lambda x: int(x.split('_')[1]))

        num_segments_to_process = 3
        selected_beams = random.sample(beams, min(num_segments_to_process, len(beams)))
        print(f"Selected beams: {selected_beams}")

        plot_data_list = []

        for beam_name in selected_beams:
            beam_grp = hf[beam_name]
            segment_names = list(beam_grp.keys())  
            segment_name = random.choice(segment_names)
            segment_grp = beam_grp[segment_name]
            data = segment_grp["data"][:]  # shape: (time_steps, range_gates, features)
            power_data = data[:, :, 0] 

            # augmentation parameters
            augment_params = {
                'negative_value': negative_value,
                'noise_strength': 0.02,
                'scale_range': (0.95, 1.05),
                'max_shift': 2,
                'max_removed_points': 25,
                'swap_prob': 0.1,
                'mask_prob': 0.1,
                'saturation_factor': 0.5,
                'augment_probabilities': {
                    'add_noise': 0,  # 1.0 = always apply, 0 = no
                    'scale_data': 0,  # 1.0 = always apply, 0 = no
                    'translate_y': 0,  # 50% chance
                    'translate_x': 0,  # 50% chance
                    'swap_adjacent_range_gates': 0,  # 50% chance
                    'mask_data': 0, 
                    'adjust_saturation': 1 # 50% chance
                },
                'verbose': True  # print messages during augmentation for adjusting
            }

            augmented_power_data_1 = augment_power(power_data, **augment_params)
            augmented_power_data_2 = augment_power(power_data, **augment_params)

            plot_data_list.append({
                'segment_name': segment_name,
                'original': power_data,
                'augmented_1': augmented_power_data_1,
                'augmented_2': augmented_power_data_2
            })

        plot_augmented_data_grid(plot_data_list, negative_value)


import numpy as np
import matplotlib.pyplot as plt

def plot_augmented_data_grid(plot_data_list, negative_value=-9999):
    """Plot original and augmented data for multiple segments in a grid."""
    num_segments = len(plot_data_list)
    fig, axs = plt.subplots(num_segments, 3, figsize=(15, 5 * num_segments))

    for i, data_dict in enumerate(plot_data_list):
        segment_name = data_dict['segment_name']
        original_data = data_dict['original']
        augmented_data_1 = data_dict['augmented_1']
        augmented_data_2 = data_dict['augmented_2']

        power_masked = np.ma.masked_where(original_data == negative_value, original_data)
        vmin = power_masked.min()
        vmax = power_masked.max()
        im = axs[i, 0].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f'Original - {segment_name}')
        axs[i, 0].set_xlabel('Time Steps')
        axs[i, 0].set_ylabel('Range Gates')
        fig.colorbar(im, ax=axs[i, 0], orientation='vertical', fraction=0.046, pad=0.04)

        # augmented data 1
        power_masked = np.ma.masked_where(augmented_data_1 == negative_value, augmented_data_1)
        im = axs[i, 1].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title('Augmented 1')
        axs[i, 1].set_xlabel('Time Steps')
        axs[i, 1].set_ylabel('Range Gates')
        fig.colorbar(im, ax=axs[i, 1], orientation='vertical', fraction=0.046, pad=0.04)

        # augmented data 2
        power_masked = np.ma.masked_where(augmented_data_2 == negative_value, augmented_data_2)
        im = axs[i, 2].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 2].set_title('Augmented 2')
        axs[i, 2].set_xlabel('Time Steps')
        axs[i, 2].set_ylabel('Range Gates')
        fig.colorbar(im, ax=axs[i, 2], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()#only runs main function if this script is run directly, not if it is imported as a module
