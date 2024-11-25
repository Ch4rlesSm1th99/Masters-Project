import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Set the path to your HDF5 file
h5_file_path = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\beam_0_padded_data.h5"

# Set the negative padding value used in your data
negative_value = -9999

# Open the HDF5 file
with h5py.File(h5_file_path, 'r') as hf:
    # List all segments
    segments = list(hf.keys())
    # Sort the segment names numerically
    segments.sort(key=lambda x: int(x.split('_')[1]))
    num_segments = len(segments)
    print(f"Number of segments: {num_segments}\n")

    # Initialize lists to collect number of time intervals per segment
    time_intervals_list = []

    # Open a log file to write the output
    log_file_path = 'inspection_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Number of segments: {num_segments}\n\n")

        # Iterate over segments and collect statistics
        for segment_name in segments:
            grp = hf[segment_name]
            data = grp['data'][:]
            start_time = grp.attrs['start_time']
            end_time = grp.attrs['end_time']

            num_time_intervals = data.shape[0]
            time_intervals_list.append(num_time_intervals)

            print(f"Segment: {segment_name}")
            print(f"Start Time: {start_time}")
            print(f"End Time: {end_time}")
            print(f"Data shape: {data.shape}")  # Shape: (time_steps, range_gates, features)
            print(f"Number of time intervals: {num_time_intervals}")

            # Write to log file
            log_file.write(f"Segment: {segment_name}\n")
            log_file.write(f"Start Time: {start_time}\n")
            log_file.write(f"End Time: {end_time}\n")
            log_file.write(f"Data shape: {data.shape}\n")
            log_file.write(f"Number of time intervals: {num_time_intervals}\n")

            # Extract features
            power = data[:, :, 0]
            velocity = data[:, :, 1]
            spectral_width = data[:, :, 2]

            # Create a mask for valid data (exclude padding values)
            valid_mask = power != negative_value

            # Compute summary statistics without including padding values
            if np.any(valid_mask):
                power_valid = power[valid_mask]
                power_min = np.min(power_valid)
                power_max = np.max(power_valid)
                power_mean = np.mean(power_valid)
            else:
                power_min = power_max = power_mean = np.nan

            # Repeat for velocity and spectral width
            valid_mask_v = velocity != negative_value
            if np.any(valid_mask_v):
                velocity_valid = velocity[valid_mask_v]
                velocity_min = np.min(velocity_valid)
                velocity_max = np.max(velocity_valid)
                velocity_mean = np.mean(velocity_valid)
            else:
                velocity_min = velocity_max = velocity_mean = np.nan

            valid_mask_w = spectral_width != negative_value
            if np.any(valid_mask_w):
                spectral_width_valid = spectral_width[valid_mask_w]
                spectral_width_min = np.min(spectral_width_valid)
                spectral_width_max = np.max(spectral_width_valid)
                spectral_width_mean = np.mean(spectral_width_valid)
            else:
                spectral_width_min = spectral_width_max = spectral_width_mean = np.nan

            # Print the statistics
            print(f"Power - min: {power_min}, max: {power_max}, mean: {power_mean}")
            print(f"Velocity - min: {velocity_min}, max: {velocity_max}, mean: {velocity_mean}")
            print(f"Spectral Width - min: {spectral_width_min}, max: {spectral_width_max}, mean: {spectral_width_mean}\n")

            # Write statistics to log file
            log_file.write(f"Power - min: {power_min}, max: {power_max}, mean: {power_mean}\n")
            log_file.write(f"Velocity - min: {velocity_min}, max: {velocity_max}, mean: {velocity_mean}\n")
            log_file.write(f"Spectral Width - min: {spectral_width_min}, max: {spectral_width_max}, mean: {spectral_width_mean}\n\n")

        # After iterating over all segments, compute total number of time steps
        total_time_intervals = sum(time_intervals_list)

        # Print and log overall statistics
        overall_stats = (
            f"Overall statistics:\n"
            f"Total number of segments: {num_segments}\n"
            f"Total number of time steps across all segments: {total_time_intervals}\n"
            f"Average number of time intervals per segment: {np.mean(time_intervals_list)}\n"
            f"Minimum number of time intervals in a segment: {np.min(time_intervals_list)}\n"
            f"Maximum number of time intervals in a segment: {np.max(time_intervals_list)}\n"
        )
        print(overall_stats)
        log_file.write(overall_stats)

    # Randomly select 9 segments to plot
    random_segments = random.sample(segments, min(9, len(segments)))
    print(f"Randomly selected segments for plotting: {random_segments}")

    # Plot the power for the selected segments
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    for idx, segment_name in enumerate(random_segments):
        grp = hf[segment_name]
        data = grp['data'][:]
        power = data[:, :, 0]
        # Mask the negative padding values
        power_masked = np.ma.masked_where(power == negative_value, power)
        im = axs[idx].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis')
        axs[idx].set_title(f'{segment_name}')
        axs[idx].set_xlabel('Time Steps')
        axs[idx].set_ylabel('Range Gates')
        fig.colorbar(im, ax=axs[idx], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Define a function to fetch metadata for a specific segment
def get_segment_metadata(h5_file_path, segment_name):
    with h5py.File(h5_file_path, 'r') as hf:
        if segment_name in hf:
            grp = hf[segment_name]
            start_time = grp.attrs['start_time']
            end_time = grp.attrs['end_time']
            data_shape = grp['data'].shape
            metadata = {
                'segment_name': segment_name,
                'start_time': start_time,
                'end_time': end_time,
                'data_shape': data_shape
            }
            return metadata
        else:
            print(f"Segment {segment_name} not found in the file.")
            return None

# Example usage of the function
segment_metadata = get_segment_metadata(h5_file_path, 'segment_0')
print("Metadata for segment_0:")
print(segment_metadata)
