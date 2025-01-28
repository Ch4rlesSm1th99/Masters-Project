import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def random_time_crop_resize(
    data,
    negative_value=-9999,
    time_crop_frac_range=(0.8, 1.0),
    min_valid_ratio=0.0,
    max_tries=20,
    verbose=False
):
    """
    randomly crop the time dimension from either the left side or the right side (50/50 chance).
    then resize back to the original number of time steps (leaving the range dimension
    unchanged). uses 1d interpolation along axis=0 (nearest neighbor by default).

    parameters
    ----------
    data : np.ndarray
        input array of shape (time_steps, range_gates).
    negative_value : float
        value indicating missing data (e.g., -9999).
    time_crop_frac_range : tuple
        (min_frac, max_frac) fraction of the original time dimension to keep.
        e.g., (0.8, 1.0) means we'll randomly keep 80% to 100% of the time steps.
    min_valid_ratio : float
        min fraction of non-missing data required in the chosen subrange.
    max_tries : int
        number of attempts to find a valid subregion.
    verbose : bool
        if true, prints debug messages.

    returns
    -------
    np.ndarray
        cropped and then time-resized array, same shape as original.
    """
    import numpy as np
    from scipy.ndimage import zoom

    data_out = data.copy()
    time_dim, range_dim = data_out.shape
    valid_mask = (data_out != negative_value)

    for attempt_i in range(max_tries):
        # pick how many time steps we'll keep
        crop_t_size = int(time_dim * np.random.uniform(*time_crop_frac_range))
        crop_t_size = max(1, min(crop_t_size, time_dim))  # clamp so we don't exceed bounds

        # 50% chance to crop from the left, 50% from the right
        if np.random.rand() < 0.5:
            # crop from left
            t_start = 0
            t_end = crop_t_size
            side = "LEFT"
        else:
            # crop from right
            t_start = time_dim - crop_t_size
            t_end = time_dim
            side = "RIGHT"

        subregion = data_out[t_start:t_end, :]
        subregion_mask = valid_mask[t_start:t_end, :]

        # check how many valid points we have in this subregion
        valid_ratio = np.sum(subregion_mask) / float(crop_t_size * range_dim)

        if valid_ratio < min_valid_ratio:
            # not enough valid data --> try again
            continue

        # resize subregion back to the original time_dim
        zoom_factor_t = time_dim / float(crop_t_size)
        # nearest-neighbor by default (order=0). switch to order=1 for bilinear if you prefer
        resized = zoom(subregion, (zoom_factor_t, 1), order=0)      # bilinear breaks loader at the moment

        if verbose:
            # note: t_end-1 is the last index included if you think of it as inclusive range
            print(f"[random_time_crop_resize] success ({side}): t_crop={crop_t_size}, valid_ratio={valid_ratio:.2f},"
                  f" subrange=({t_start} -> {t_end-1})")
        return resized

    # if we never find a subregion that meets the valid ratio, return original
    if verbose:
        print("[random_time_crop_resize] failed to find valid subregion, returning original.")
    return data_out




def augment_power(
    data,
    negative_value=-9999,
    noise_strength=0.01,
    scale_range=(0.9, 1.1),
    swap_prob=0.5,
    mask_prob=0.1,
    time_crop_frac_range=(0.8, 1.0),
    min_valid_ratio=0.0,
    max_tries=20,
    augment_probabilities=None,
    verbose=False
):
    """
    apply the following augmentations to power data (time_steps x range_gates):
      1) random time-only crop+resize
      2) add noise
      3) scale data
      4) swap adjacent range gates
      5) mask data

    parameters
    ----------
    data : np.ndarray
        shape (time_steps, range_gates).
    negative_value : float
        missing data value (e.g., -9999).
    noise_strength : float
        std dev of gaussian noise.
    scale_range : tuple
        range for random scaling factor.
    swap_prob : float
        probability of swapping adjacent gates in a row.
    mask_prob : float
        probability of masking valid points with negative_value.
    time_crop_frac_range : tuple
        (min_frac, max_frac) for time-only cropping.
    min_valid_ratio : float
        minimum fraction of valid data in the cropped subregion.
    max_tries : int
        attempts to find a valid subregion for cropping.
    augment_probabilities : dict
        e.g. {
          'time_crop_resize': 1.0,
          'add_noise': 1.0,
          'scale_data': 1.0,
          'swap_adjacent_range_gates': 0.5,
          'mask_data': 0.5
        }
    verbose : bool
        print debug info if true.

    returns
    -------
    np.ndarray
        augmented data, same shape as input.
    """

    data_aug = data.copy()

    # default vals
    if augment_probabilities is None:
        augment_probabilities = {
            'time_crop_resize': 1.0,
            'add_noise': 1.0,
            'scale_data': 1.0,
            'swap_adjacent_range_gates': 0.5,
            'mask_data': 0.5
        }

    # helper func for each aug
    # all helper function leave padding vals at their exact values (ie -9999 or whatever is chosen)
    def time_crop_resize_func(data_in):
        return random_time_crop_resize(
            data_in,
            negative_value=negative_value,
            time_crop_frac_range=time_crop_frac_range,
            min_valid_ratio=min_valid_ratio,
            max_tries=max_tries,
            verbose=verbose
        )

    def add_noise_func(data_in):
        valid_mask = (data_in != negative_value)
        noise = np.random.normal(loc=0, scale=noise_strength, size=data_in.shape)
        data_in[valid_mask] += noise[valid_mask]
        if verbose:
            print("[add_noise_func] Applied noise.")
        return data_in

    def scale_data_func(data_in):
        valid_mask = (data_in != negative_value)
        scale_factor = np.random.uniform(*scale_range)
        data_in[valid_mask] *= scale_factor
        if verbose:
            print(f"[scale_data_func] scale_factor={scale_factor:.2f}")
        return data_in

    def swap_adjacent_range_gates_func(data_in):
        num_time_steps, num_range_gates = data_in.shape
        for t in range(num_time_steps):     # loop over each time step and try swapping adjacent gates
            for r in range(num_range_gates - 1):
                if np.random.rand() < swap_prob:           # only swap if both cells are valid (ie not padded)
                    if (data_in[t, r] != negative_value) and (data_in[t, r + 1] != negative_value):
                        data_in[t, r], data_in[t, r + 1] = data_in[t, r + 1], data_in[t, r]
        if verbose:
            print(f"[swap_adjacent_range_gates_func] swap_prob={swap_prob}")
        return data_in

    def mask_data_func(data_in):
        num_time_steps, num_range_gates = data_in.shape
        mask = (np.random.rand(num_time_steps, num_range_gates) < mask_prob) & (data_in != negative_value)
        data_in[mask] = negative_value
        if verbose:
            print(f"[mask_data_func] mask_prob={mask_prob}")
        return data_in

    # define the order of augmentations
    augmentations = [
        ('time_crop_resize', time_crop_resize_func),
        ('add_noise', add_noise_func),
        ('scale_data', scale_data_func),
        ('swap_adjacent_range_gates', swap_adjacent_range_gates_func),
        ('mask_data', mask_data_func)
    ]

    # go through each augmentation in order and apply it with some probability
    for aug_name, aug_func in augmentations:
        prob = augment_probabilities.get(aug_name, 0)
        if np.random.rand() < prob:
            data_aug = aug_func(data_aug)
            if verbose and aug_name not in ['swap_adjacent_range_gates', 'mask_data']:
                print(f"Applied augmentation: {aug_name}")
        else:
            if verbose:
                print(f"Skipped augmentation: {aug_name}")

    return data_aug


if __name__ == "__main__":
    """
    test: load a random segment from train.h5, apply time-only cropping (+ other augmentations)
          twice to get two augmented views, then plot them alongside the original.
    """
    import os

    h5_file_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\train.h5"
    NEG_VALUE = -9999

    # probability config for time-only augmentation
    aug_probs = {
        'time_crop_resize': 1.0,  # always do time-only crop
        'add_noise': 1.0,
        'scale_data': 1.0,
        'swap_adjacent_range_gates': 0.5,
        'mask_data': 0.0
    }

    with h5py.File(h5_file_path, 'r') as hf:
        segments = list(hf.keys())
        seg_name = random.choice(segments)
        print(f"Segment chosen: {seg_name}")
        data = hf[seg_name]['data'][:]  # shape: (time_steps, range_gates, features)
        power_data = data[:, :, 0]      # feature 0 is power, others are doppler velocity and spectral width

    # gen two augmented views
    aug1 = augment_power(
        power_data,
        negative_value=NEG_VALUE,
        noise_strength=0.1,
        scale_range=(0.8, 1.2),
        swap_prob=0.4,
        mask_prob=0.2,
        time_crop_frac_range=(0.7, 1.0),  # bigger range, e.g. 70% to 100%
        min_valid_ratio=0.0,             # currently set to no threshold
        max_tries=5,
        augment_probabilities=aug_probs,
        verbose=True
    )

    print("\n---- SECOND AUGMENTATION ----\n")

    aug2 = augment_power(
        power_data,
        negative_value=NEG_VALUE,
        noise_strength=0.1,
        scale_range=(0.8, 1.2),
        swap_prob=0.4,
        mask_prob=0.2,
        time_crop_frac_range=(0.7, 1.0),
        min_valid_ratio=0.0,
        max_tries=5,
        augment_probabilities=aug_probs,
        verbose=True
    )

    # original vs. two augmented
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # og
    masked_orig = np.ma.masked_where(power_data == NEG_VALUE, power_data)
    vmin, vmax = masked_orig.min(), masked_orig.max()
    im0 = axs[0].imshow(masked_orig.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Original - {seg_name}")
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Range Gates")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # aug1
    masked_aug1 = np.ma.masked_where(aug1 == NEG_VALUE, aug1)
    im1 = axs[1].imshow(masked_aug1.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title("Augmented #1")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Range Gates")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # aug2
    masked_aug2 = np.ma.masked_where(aug2 == NEG_VALUE, aug2)
    im2 = axs[2].imshow(masked_aug2.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set_title("Augmented #2")
    axs[2].set_xlabel("Time Steps")
    axs[2].set_ylabel("Range Gates")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
