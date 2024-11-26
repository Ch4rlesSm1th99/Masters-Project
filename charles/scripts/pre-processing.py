import os
import bz2
import pydarn
import datetime as dt
import numpy as np
import pandas as pd
import warnings
import h5py
import matplotlib.pyplot as plt
import random

warnings.filterwarnings("ignore", category=UserWarning)

# user options
debugging_mode = False  # true for debugging data directory
plotting_enabled = True  # enable or disable plotting
extract_data = False  # extract data into segments or just plot
logging_enabled = True  # enable or disable logging to a file
beam_number = 0  # beam number to process
negative_value = -9999  # negative padding value used in data

# data directories
if debugging_mode:
    data_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\debugging data"
else:
    data_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\1995"

save_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data"

# path to the preprocessed HDF5 file
h5_file_path_preprocessed = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\beam_0_selected_data.h5"

def load_and_preprocess_data(data_directory, beam_number, negative_value, extract_data):
    if extract_data:
        # records to store data from files
        all_records = []

        files = [f for f in os.listdir(data_directory) if f.endswith('.bz2')]

        if not files:
            print("No files found in the directory.")
            return None
        else:
            print(f"Processing {len(files)} files...")

            for fitacf_file_name in files:
                fitacf_file = os.path.join(data_directory, fitacf_file_name)

                try:
                    with bz2.open(fitacf_file, 'rb') as fp:
                        fitacf_stream = fp.read()

                    sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
                    fitacf_data = sdarn_read.read_fitacf()

                    if not fitacf_data:
                        print(f"No data found in the file {fitacf_file_name}.")
                        continue

                    for record in fitacf_data:
                        if record.get('bmnum') != beam_number:
                            continue

                        # check for 'slist' and not empty
                        if 'slist' not in record or len(record['slist']) == 0:
                            continue

                        # extract metadata --> timestamp
                        try:
                            record_time = dt.datetime(
                                record['time.yr'],
                                record['time.mo'],
                                record['time.dy'],
                                record['time.hr'],
                                record['time.mt'],
                                record['time.sc'],
                                int(record['time.us'] / 1000)  # microseconds to milliseconds
                            )
                        except ValueError as e:
                            print(f"Invalid date in record: {e}. Skipping.")
                            continue

                        common_data = {
                            'time': record_time,
                            'bmnum': record['bmnum'],
                            'channel': record.get('channel', np.nan),
                            'cp': record.get('cp', np.nan),
                            'nrang': record['nrang'],
                            'frang': record['frang'],
                            'rsep': record['rsep'],
                            'stid': record['stid'],
                        }

                        # for each range gate in slist, extract the features
                        slist = record['slist']
                        for idx, gate in enumerate(slist):
                            gate_data = common_data.copy()
                            gate_data.update({
                                'range_gate': gate,
                                'p_l': record['p_l'][idx],
                                'v': record['v'][idx],
                                'w_l': record['w_l'][idx],
                                'gflg': record['gflg'][idx] if 'gflg' in record else np.nan,
                            })
                            all_records.append(gate_data)

                except Exception as e:
                    print(f"An error occurred while processing {fitacf_file_name}: {e}")

        if not all_records:
            print("No data was collected for the specified beam.")
            return None

        df = pd.DataFrame(all_records)

        # convert data types
        df['bmnum'] = df['bmnum'].astype(int)
        df['stid'] = df['stid'].astype(int)
        df['range_gate'] = df['range_gate'].astype(int)

        # set a MultiIndex with 'time' and 'range_gate'
        df.set_index(['time', 'range_gate'], inplace=True)

        # sort the DataFrame
        df.sort_index(inplace=True)

        # check for duplicate time stamps
        duplicates = df.index[df.index.duplicated()]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} duplicate time-range_gate pairs. Removing duplicates.")
            # remove duplicates
            df = df[~df.index.duplicated(keep='first')]
        else:
            print("No duplicate time-range_gate pairs found.")

        # df properties
        print(f"\nInitial DataFrame shape: {df.shape}")
        print(f"Number of unique times: {df.index.get_level_values('time').nunique()}")
        print(f"Number of unique range gates: {df.index.get_level_values('range_gate').nunique()}")

        # handle missing data (fill NaNs with a negative value)
        df['p_l'] = df['p_l'].fillna(negative_value)
        df['v'] = df['v'].fillna(negative_value)
        df['w_l'] = df['w_l'].fillna(negative_value)

        # count valid data points
        num_valid_power = np.sum(df['p_l'].values != negative_value)
        num_valid_velocity = np.sum(df['v'].values != negative_value)
        num_valid_spectral_width = np.sum(df['w_l'].values != negative_value)

        print(f"\nNumber of valid power measurements: {num_valid_power}")
        print(f"Number of valid velocity measurements: {num_valid_velocity}")
        print(f"Number of valid spectral width measurements: {num_valid_spectral_width}")

        # get unique times and range gates from the data
        times = df.index.get_level_values('time').unique()
        range_gates = np.arange(df.index.get_level_values('range_gate').min(),
                                df.index.get_level_values('range_gate').max() + 1)

        power_pivot = df['p_l'].unstack(level='range_gate').reindex(index=times, columns=range_gates)
        velocity_pivot = df['v'].unstack(level='range_gate').reindex(index=times, columns=range_gates)
        spectral_width_pivot = df['w_l'].unstack(level='range_gate').reindex(index=times, columns=range_gates)

        # fill NaNs
        power_pivot = power_pivot.fillna(negative_value)
        velocity_pivot = velocity_pivot.fillna(negative_value)
        spectral_width_pivot = spectral_width_pivot.fillna(negative_value)

        timestamps = pd.to_datetime(power_pivot.index.to_numpy())

        def segment_and_select_closest_features(power_array, velocity_array, spectral_width_array, timestamps,
                                                segment_length='1H', num_time_steps=30, min_valid_points=100):
            segments = []
            segment_times = []
            delta = pd.Timedelta(segment_length)
            current_time = timestamps[0]
            end_time = timestamps[-1]

            while current_time < end_time:
                next_time = current_time + delta
                # find indices corresponding to the segment
                mask = (timestamps >= current_time) & (timestamps < next_time)
                segment_times_in_window = timestamps[mask]

                if np.sum(mask) > 0:
                    # collect data for the segment
                    segment_power = power_array[mask]
                    segment_velocity = velocity_array[mask]
                    segment_spectral_width = spectral_width_array[mask]
                    # check if segment has enough valid data
                    valid_points = np.sum(segment_power != negative_value)
                    if valid_points >= min_valid_points:
                        print(f"Processing segment starting at {current_time} with {valid_points} valid points.")
                        # create 30 evenly spaced desired timestamps within the segment
                        desired_times = pd.date_range(start=current_time, end=next_time - pd.Timedelta('1ns'), periods=num_time_steps)
                        # for each desired timestamp, find closest actual timestamp in segment
                        actual_times = segment_times_in_window
                        selected_indices = []
                        for desired_time in desired_times:
                            time_diffs = np.abs(actual_times - desired_time)
                            min_diff_idx = np.argmin(time_diffs)
                            selected_indices.append(min_diff_idx)
                        selected_indices = np.array(selected_indices)
                        # extract data at the selected indices
                        selected_power = segment_power[selected_indices]
                        selected_velocity = segment_velocity[selected_indices]
                        selected_spectral_width = segment_spectral_width[selected_indices]
                        # stack
                        combined_data = np.stack([selected_power, selected_velocity, selected_spectral_width], axis=-1)
                        segments.append(combined_data)
                        segment_times.append(current_time)
                    else:
                        print(f"Segment starting at {current_time} skipped due to insufficient valid data.")
                else:
                    print(f"No data in segment starting at {current_time}.")
                current_time = next_time
            return segments, segment_times

        segments, segment_times = segment_and_select_closest_features(
            power_pivot.values,
            velocity_pivot.values,
            spectral_width_pivot.values,
            timestamps,
            segment_length='1H',
            num_time_steps=30,
            min_valid_points=100
        )

        output_file = os.path.join(save_directory, f'beam_{beam_number}_selected_data.h5')

        with h5py.File(output_file, 'w') as hf:
            for idx, (data, time) in enumerate(zip(segments, segment_times)):
                group_name = f'segment_{idx}'
                grp = hf.create_group(group_name)
                grp.create_dataset('data', data=data)

                # convert numpy.datetime64 to datetime before calling isoformat()
                start_time_dt = pd.to_datetime(time).to_pydatetime()
                end_time_dt = pd.to_datetime(time + pd.Timedelta('1H')).to_pydatetime()

                grp.attrs['start_time'] = start_time_dt.isoformat()
                grp.attrs['end_time'] = end_time_dt.isoformat()

        print(f"\nPreprocessed data saved to {output_file}")
        h5_file_path = output_file
    else:
        # if not extracting data, use the predefined HDF5 file path
        h5_file_path = h5_file_path_preprocessed
        print(f"Using preprocessed data from {h5_file_path}")

    # plots the number of valid range gate spots over time
    if plotting_enabled and extract_data:
        valid_power_per_time = np.sum(power_pivot.values != negative_value, axis=1)
        plt.figure(figsize=(12, 6))
        plt.plot(power_pivot.index, valid_power_per_time)
        plt.title('Number of Valid Power Measurements Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Valid Measurements')
        plt.show()

    return h5_file_path

def inspect_data(h5_file_path, negative_value, logging_enabled, plotting_enabled):
    if h5_file_path is None or not os.path.exists(h5_file_path):
        print("No HDF5 file path provided for inspection or file does not exist.")
        return

    with h5py.File(h5_file_path, 'r') as hf:
        segments = list(hf.keys())
        segments.sort(key=lambda x: int(x.split('_')[1]))
        num_segments = len(segments)
        print(f"Number of segments: {num_segments}\n")

        time_intervals_list = []

        if logging_enabled:
            log_file_path = 'inspection_log.txt'
            log_file = open(log_file_path, 'w')
            log_file.write(f"Number of segments: {num_segments}\n\n")
        else:
            log_file = None

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
            print(f"Data shape: {data.shape}")  # shape = (time_steps, range_gates, features)
            print(f"Number of time intervals: {num_time_intervals}")

            # log segment details
            if logging_enabled:
                log_file.write(f"Segment: {segment_name}\n")
                log_file.write(f"Start Time: {start_time}\n")
                log_file.write(f"End Time: {end_time}\n")
                log_file.write(f"Data shape: {data.shape}\n")
                log_file.write(f"Number of time intervals: {num_time_intervals}\n")

            # isolate features
            power = data[:, :, 0]
            velocity = data[:, :, 1]
            spectral_width = data[:, :, 2]

            # mask for padded data
            valid_mask = power != negative_value

            # summaries without the padding
            if np.any(valid_mask):
                power_valid = power[valid_mask]
                power_min = np.min(power_valid)
                power_max = np.max(power_valid)
                power_mean = np.mean(power_valid)
            else:
                power_min = power_max = power_mean = np.nan

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

            print(f"Power - min: {power_min}, max: {power_max}, mean: {power_mean}")
            print(f"Velocity - min: {velocity_min}, max: {velocity_max}, mean: {velocity_mean}")
            print(f"Spectral Width - min: {spectral_width_min}, max: {spectral_width_max}, mean: {spectral_width_mean}\n")

            if logging_enabled:
                log_file.write(f"Power - min: {power_min}, max: {power_max}, mean: {power_mean}\n")
                log_file.write(f"Velocity - min: {velocity_min}, max: {velocity_max}, mean: {velocity_mean}\n")
                log_file.write(f"Spectral Width - min: {spectral_width_min}, max: {spectral_width_max}, mean: {spectral_width_mean}\n\n")

        total_time_intervals = sum(time_intervals_list)

        overall_stats = (
            f"Overall statistics:\n"
            f"Total number of segments: {num_segments}\n"
            f"Total number of time steps across all segments: {total_time_intervals}\n"
            f"Average number of time intervals per segment: {np.mean(time_intervals_list)}\n"
            f"Minimum number of time intervals in a segment: {np.min(time_intervals_list)}\n"
            f"Maximum number of time intervals in a segment: {np.max(time_intervals_list)}\n"
        )
        print(overall_stats)
        if logging_enabled:
            log_file.write(overall_stats)

        if logging_enabled:
            log_file.close()

        # select 9 random segments to plot
        if plotting_enabled:
            random_segments = random.sample(segments, min(9, len(segments)))
            print(f"Randomly selected segments for plotting: {random_segments}")

            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            axs = axs.flatten()
            for idx, segment_name in enumerate(random_segments):
                grp = hf[segment_name]
                data = grp['data'][:]
                power = data[:, :, 0]
                # mask the negative padding values
                power_masked = np.ma.masked_where(power == negative_value, power)
                im = axs[idx].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis')
                axs[idx].set_title(f'{segment_name}')
                axs[idx].set_xlabel('Time Steps')
                axs[idx].set_ylabel('Range Gates')
                fig.colorbar(im, ax=axs[idx], orientation='vertical', fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()

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

    segment_metadata = get_segment_metadata(h5_file_path, 'segment_0')
    print("Metadata for segment_0:")
    print(segment_metadata)

def main():
    h5_file_path = load_and_preprocess_data(data_directory, beam_number, negative_value, extract_data)
    inspect_data(h5_file_path, negative_value, logging_enabled, plotting_enabled)

if __name__ == "__main__":
    main()
