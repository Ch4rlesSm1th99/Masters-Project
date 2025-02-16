import os
import bz2
import pydarn
from pydarn.utils.coordinates import aacgm_coordinates
import datetime as dt
import numpy as np
import pandas as pd
import warnings
import h5py
import matplotlib.pyplot as plt
import random
import gc

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# user options
debugging_mode  = True  # True for a smaller test dataset
plotting_enabled= True   # enable or disable plotting
extract_data    = True   # extract data into segments or just plot
logging_enabled = True   # enable or disable logging to a file for segment features
negative_value  = -9999  # negative padding value used in data

# data directories
if debugging_mode:
    data_directory = r"C:\Users\aman\Desktop\MPhys Data\Data\Debugging"
else:
    data_directory = r"C:\Users\aman\Desktop\MPhys Data\Data\1995"

save_directory = r"C:\Users\aman\Desktop\MPhys Data\Data"

output_h5_file = os.path.join(save_directory, 'all_beams_selected_data.h5')

###############################################################################
def load_and_preprocess_beam(
    data_directory,
    beam_number,
    negative_value,
    min_valid_points=100,
    num_time_steps=30
):
    """
    Reads all .bz2 files in `data_directory`, filters just the records
    for the given `beam_number`, returns a list of (segment_data, start_time)
    for that beam.
    """
    all_records = []

    files = [f for f in os.listdir(data_directory) if f.endswith('.bz2')]
    if not files:
        print("No files found in the directory.")
        return []

    print(f"\n[Beam {beam_number}] Processing {len(files)} files...")

    # Collect all records for this beam
    for fitacf_file_name in files:
        fitacf_file = os.path.join(data_directory, fitacf_file_name)
        try:
            with bz2.open(fitacf_file, 'rb') as fp:
                fitacf_stream = fp.read()
            sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
            fitacf_data = sdarn_read.read_fitacf()

            if not fitacf_data:
                # e.g. empty or invalid file
                # print(f"No data found in {fitacf_file_name}.")
                continue

            for record in fitacf_data:
                bm = record.get('bmnum')
                if bm != beam_number:
                    continue

                if 'slist' not in record or len(record['slist']) == 0:
                    continue

                # extract timestamp
                try:
                    record_time = dt.datetime(
                        record['time.yr'],
                        record['time.mo'],
                        record['time.dy'],
                        record['time.hr'],
                        record['time.mt'],
                        record['time.sc'],
                        int(record['time.us'] / 1000)  # microseconds --> milliseconds
                    )
                except ValueError as e:
                    # invalid date
                    continue

                common_data = {
                    'time': record_time,
                    'bmnum': bm,
                    'channel': record.get('channel', np.nan),
                    'cp': record.get('cp', np.nan),
                    'nrang': record.get('nrang'),
                    'frang': record.get('frang'),
                    'rsep': record.get('rsep'),
                    'stid': record['stid'],
                }

                slist = record['slist']
                for idx, gate in enumerate(slist):
                    gate_data = common_data.copy()
                    gate_data.update({
                        'range_gate': gate,
                        'p_l': record['p_l'][idx],
                        'v': record['v'][idx],
                        'w_l': record['w_l'][idx],
                        'gflg': record['gflg'][idx] if 'gflg' in record else np.nan
                    })
                    all_records.append(gate_data)

        except Exception as e:
            print(f"[Beam {beam_number}] Error reading {fitacf_file_name}: {e}")

    if not all_records:
        print(f"[Beam {beam_number}] No data collected.")
        return []

    df_beam = pd.DataFrame(all_records)
    # convert data types
    df_beam['stid']       = df_beam['stid'].astype(int)
    df_beam['range_gate'] = df_beam['range_gate'].astype(int)
    # set index: (time, range_gate)
    df_beam.set_index(['time', 'range_gate'], inplace=True)
    df_beam.sort_index(inplace=True)

    # remove duplicates
    duplicates = df_beam.index[df_beam.index.duplicated()]
    if not duplicates.empty:
        df_beam = df_beam[~df_beam.index.duplicated(keep='first')]

    # fill missing
    df_beam['p_l'] = df_beam['p_l'].fillna(negative_value)
    df_beam['v']   = df_beam['v'].fillna(negative_value)
    df_beam['w_l'] = df_beam['w_l'].fillna(negative_value)

    # pivot by range_gate
    times = df_beam.index.get_level_values('time').unique()
    rgate_min = df_beam.index.get_level_values('range_gate').min()
    rgate_max = df_beam.index.get_level_values('range_gate').max()
    range_gates = np.arange(rgate_min, rgate_max+1)

    #extracting parameters for aacgm conversion
    #gets total number of beams
    unique_beams = {record.get('bmnum') for record in fitacf_data}
    num_beams = len(unique_beams)
    #gets the station id
    stationid = int(df_beam['stid'].unique()[0])

    #uses first time step as conversion time
    conversion_time = pd.to_datetime(times[0])

    mag_lat_corners, mag_lon_corners = aacgm_coordinates(
        stid=stationid,
        beams=num_beams,         
        gates=(rgate_min, rgate_max),
        date=conversion_time         
        )

    #calculates the mlats from the center of the corners
    mag_lat_centers = (mag_lat_corners[:, :-1] + mag_lat_corners[:, 1:]) / 2


    mag_lat = mag_lat_centers.mean(axis=1)  


    df_beam = df_beam.reset_index()

    #assign mag_lat based on range_gate index position
    df_beam['mag_lat'] = mag_lat[df_beam['range_gate'].values - rgate_min]

    df_beam.set_index(['time', 'mag_lat'], inplace=True)

    power_pivot = df_beam['p_l'].unstack(level='mag_lat').reindex(index=times, columns=mag_lat)
    velocity_pivot = df_beam['v'].unstack(level='mag_lat').reindex(index=times, columns=mag_lat)
    spectral_width_pivot = df_beam['w_l'].unstack(level='mag_lat').reindex(index=times, columns=mag_lat)

    power_pivot.fillna(negative_value, inplace=True)
    velocity_pivot.fillna(negative_value, inplace=True)
    spectral_width_pivot.fillna(negative_value, inplace=True)

    timestamps = pd.to_datetime(power_pivot.index)
    power_array    = power_pivot.values
    velocity_array = velocity_pivot.values
    width_array    = spectral_width_pivot.values
    print(power_array)

    # segment
    segments, segment_times = segment_and_select_closest_features(
        power_array,
        velocity_array,
        width_array,
        timestamps,
        segment_length='1H',
        num_time_steps=num_time_steps,
        min_valid_points=min_valid_points,
        negative_value=negative_value
    )

    # return the segments plus relevant metadata
    # a list of (numpy_array, start_time, stid_list)
    stid_list = df_beam['stid'].unique()  # in case you want station info
    results = []
    for seg_data, seg_start in zip(segments, segment_times):
        results.append((seg_data, seg_start, stid_list, mag_lat.tolist()))

    # free any big memory consumption
    del df_beam, power_pivot, velocity_pivot, spectral_width_pivot
    gc.collect()

    return results

###############################################################################
def load_and_preprocess_data_all_beams(
    data_directory,
    negative_value,
    extract_data,
    beams_to_process=range(16)
):
    """
    Master function to process each beam in `beams_to_process` individually
    and store them in a single HDF5 file as `beam_XXX/segment_YYY` groups.
    """
    if not extract_data:
        # jump directly to inspection of file
        print(f"Using preprocessed data from {output_h5_file}")
        return output_h5_file

    # create the output HDF5 or overwrite
    with h5py.File(output_h5_file, 'w') as hf_full:
        for beam_number in beams_to_process:
            # 1) parse data for this specific beam
            beam_segments = load_and_preprocess_beam(
                data_directory,
                beam_number,
                negative_value,
                min_valid_points=100,
                num_time_steps=30
            )
            if not beam_segments:
                print(f"No valid segments created or no data for beam {beam_number}.")
                continue

            # 2) store each segment into a group
            beam_group = hf_full.create_group(f'beam_{beam_number}')

            for idx_seg, (seg_data, seg_start_time, stid_list, mag_lat) in enumerate(beam_segments):
                seg_group = beam_group.create_group(f'segment_{idx_seg}')
                seg_group.create_dataset('data', data=seg_data)

                start_time_dt = pd.to_datetime(seg_start_time).to_pydatetime()
                end_time_dt   = start_time_dt + pd.Timedelta('1H')

                seg_group.attrs['start_time'] = start_time_dt.isoformat()
                seg_group.attrs['end_time']   = end_time_dt.isoformat()
                seg_group.attrs['beam_number']= beam_number
                seg_group.attrs['stid_list']  = ','.join(str(x) for x in stid_list)
                seg_group.attrs['mag_lat'] = ','.join(map(str, mag_lat))

        # the file now contains all beams & segments
        # split to train val test if wanted
        all_beam_segments = []
        for beam_name in hf_full.keys():
            # e.g. 'beam_0', 'beam_1', ...
            if not beam_name.startswith('beam_'):
                continue
            group = hf_full[beam_name]
            for seg_name in group.keys():
                all_beam_segments.append((beam_name, seg_name))

        total_segments = len(all_beam_segments)
        print(f"\nTotal segments across all beams: {total_segments}")

        if total_segments > 0:
            indices = list(range(total_segments))
            random.shuffle(indices)

            train_ratio = 0.7
            val_ratio   = 0.15
            test_ratio  = 0.15

            train_end = int(train_ratio * total_segments)
            val_end   = train_end + int(val_ratio * total_segments)

            train_indices = indices[:train_end]
            val_indices   = indices[train_end:val_end]
            test_indices  = indices[val_end:]

            print(f"Training segments: {len(train_indices)}")
            print(f"Validation segments: {len(val_indices)}")
            print(f"Test segments: {len(test_indices)}")

            # function to write splits
            def save_split(indices_list, split_name):
                split_file = os.path.join(save_directory, f'{split_name}.h5')
                with h5py.File(split_file, 'w') as hf_split:
                    for i, idx_in_all in enumerate(indices_list):
                        b_name, s_name = all_beam_segments[idx_in_all]
                        src_group = hf_full[b_name][s_name]
                        data_seg  = src_group['data'][:]

                        # create a group in split file
                        grp = hf_split.create_group(f'{b_name}_{s_name}')
                        grp.create_dataset('data', data=data_seg)
                        for attr_key, attr_val in src_group.attrs.items():
                            grp.attrs[attr_key] = attr_val

                print(f"{split_name.capitalize()} set saved to {split_file}")

            save_split(train_indices, 'train')
            save_split(val_indices, 'val')
            save_split(test_indices, 'test')

    return output_h5_file


###############################################################################
def segment_and_select_closest_features(
    power_array,
    velocity_array,
    spectral_width_array,
    timestamps,
    segment_length='1H',
    num_time_steps=30,
    min_valid_points=100,
    negative_value=-9999
):
    """
    Splits data into time segments (e.g. 1-hour windows) and then
    selects `num_time_steps` time points from each segment by
    picking the actual times closest to equally spaced desired timestamps.
    """
    segments = []
    segment_times = []
    delta = pd.Timedelta(segment_length)

    if len(timestamps) == 0:
        return segments, segment_times

    current_time = timestamps[0]
    end_time     = timestamps[-1]

    while current_time < end_time:
        next_time = current_time + delta
        # find indices for the segment
        mask = (timestamps >= current_time) & (timestamps < next_time)
        segment_times_in_window = timestamps[mask]

        if np.any(mask):
            # collect data for the segment
            seg_power = power_array[mask]
            seg_vel   = velocity_array[mask]
            seg_width = spectral_width_array[mask]

            # check if segment has enough valid data
            valid_points = np.sum(seg_power != negative_value)
            if valid_points >= min_valid_points:
                print(f"Processing segment starting at {current_time} with {valid_points} valid points.")

                # create evenly spaced desired timestamps
                desired_times = pd.date_range(
                    start=current_time,
                    end=next_time - pd.Timedelta('1ns'),
                    periods=num_time_steps
                )

                actual_times = segment_times_in_window
                selected_indices = []
                for desired_time in desired_times:
                    time_diffs = np.abs(actual_times - desired_time)
                    min_diff_idx = np.argmin(time_diffs)
                    selected_indices.append(min_diff_idx)

                selected_indices = np.array(selected_indices)

                # extract data at these indices
                selected_power = seg_power[selected_indices]
                selected_vel   = seg_vel[selected_indices]
                selected_width = seg_width[selected_indices]

                # shape => (num_time_steps, num_range_gates, 3)
                combined_data = np.stack(
                    [selected_power, selected_vel, selected_width],
                    axis=-1
                )

                segments.append(combined_data)
                segment_times.append(current_time)
            else:
                # not enough valid points
                pass

        current_time = next_time

    return segments, segment_times


###############################################################################
def inspect_data(h5_file_path, negative_value, logging_enabled, plotting_enabled):
    """
    Inspects the HDF5 dataset, printing summary info and optionally plotting.
    """
    if h5_file_path is None or not os.path.exists(h5_file_path):
        print("No HDF5 file path provided for inspection or file does not exist.")
        return

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    total_segments = 0
    time_intervals_list = []

    log_file = None
    if logging_enabled:
        log_file_path = 'inspection_log.txt'
        log_file = open(log_file_path, 'w')

    with h5py.File(h5_file_path, 'r') as hf:
        beam_groups = [grp for grp in hf.keys() if grp.startswith('beam_')]
        if not beam_groups:
            print("No beam groups found in the HDF5 file.")
            return

        for beam_grp_name in beam_groups:
            beam_grp = hf[beam_grp_name]
            segment_names = list(beam_grp.keys())
            segment_names.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

            print(f"\nBeam group: {beam_grp_name} with {len(segment_names)} segments.")
            if log_file:
                log_file.write(f"\nBeam group: {beam_grp_name} with {len(segment_names)} segments.\n")

            total_segments += len(segment_names)

            for seg_name in segment_names:
                seg_grp = beam_grp[seg_name]
                data = seg_grp['data'][:]
                start_time = seg_grp.attrs['start_time']
                end_time   = seg_grp.attrs['end_time']
                bmnum      = seg_grp.attrs.get('beam_number', 'Unknown')

                num_time_intervals = data.shape[0]
                time_intervals_list.append(num_time_intervals)

                msg_info = (
                    f"Segment: {seg_name}\n"
                    f"  Start Time: {start_time}\n"
                    f"  End Time:   {end_time}\n"
                    f"  Beam Num:   {bmnum}\n"
                    f"  Data shape: {data.shape}\n"
                    f"  Time steps: {num_time_intervals}"
                )
                print(msg_info)
                if log_file:
                    log_file.write(msg_info + "\n")

                # gather quick stats
                power = data[:, :, 0]
                velocity = data[:, :, 1]
                spectral_width = data[:, :, 2]

                valid_mask_p = (power != negative_value)
                if np.any(valid_mask_p):
                    p_min = power[valid_mask_p].min()
                    p_max = power[valid_mask_p].max()
                    p_mean= power[valid_mask_p].mean()
                else:
                    p_min = p_max = p_mean = np.nan

                valid_mask_v = (velocity != negative_value)
                if np.any(valid_mask_v):
                    v_min = velocity[valid_mask_v].min()
                    v_max = velocity[valid_mask_v].max()
                    v_mean= velocity[valid_mask_v].mean()
                else:
                    v_min = v_max = v_mean = np.nan

                valid_mask_w = (spectral_width != negative_value)
                if np.any(valid_mask_w):
                    w_min = spectral_width[valid_mask_w].min()
                    w_max = spectral_width[valid_mask_w].max()
                    w_mean= spectral_width[valid_mask_w].mean()
                else:
                    w_min = w_max = w_mean = np.nan

                msg_stats = (
                    f"  Power - min: {p_min}, max: {p_max}, mean: {p_mean}\n"
                    f"  Velocity - min: {v_min}, max: {v_max}, mean: {v_mean}\n"
                    f"  Spectral Width - min: {w_min}, max: {w_max}, mean: {w_mean}\n"
                )
                print(msg_stats)
                if log_file:
                    log_file.write(msg_stats + "\n")

    if time_intervals_list:
        total_time_intervals = sum(time_intervals_list)
        overall_stats = (
            f"\nOverall statistics:\n"
            f"  Total number of segments: {total_segments}\n"
            f"  Total number of time steps across all segments: {total_time_intervals}\n"
            f"  Average number of time intervals per segment: {np.mean(time_intervals_list)}\n"
            f"  Min number of time intervals in a segment: {np.min(time_intervals_list)}\n"
            f"  Max number of time intervals in a segment: {np.max(time_intervals_list)}\n"
        )
        print(overall_stats)
        if log_file:
            log_file.write(overall_stats)
    else:
        print("No time intervals found in any segments.")
        if log_file:
            log_file.write("No time intervals found in any segments.\n")

    if log_file:
        log_file.close()

    # plotting check for verifying correctly built segments
    if plotting_enabled:
        with h5py.File(h5_file_path, 'r') as hf:
            all_segments = []
            for beam_grp_name in hf.keys():
                if beam_grp_name.startswith('beam_'):
                    segs = list(hf[beam_grp_name].keys())
                    for seg_name in segs:
                        all_segments.append((beam_grp_name, seg_name))

            if not all_segments:
                print("No segments available for plotting.")
                return

            random_segments = random.sample(all_segments, min(9, len(all_segments)))
            print(f"Randomly selected segments for plotting: {random_segments}")

            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            axs = axs.flatten()

            for idx, (beam_grp_name, seg_name) in enumerate(random_segments):
                data = hf[beam_grp_name][seg_name]['data'][:]
                power = data[:, :, 0]
                power_masked = np.ma.masked_where(power == negative_value, power)

                im = axs[idx].imshow(power_masked.T, aspect='auto', origin='lower', cmap='viridis')
                axs[idx].set_title(f"{beam_grp_name}/{seg_name}")
                axs[idx].set_xlabel('Time Steps')

                # set y-ticks to show magnetic latitude
                if 'mag_lat' in hf[beam_grp_name][seg_name].attrs:
                    mag_lat_str = hf[beam_grp_name][seg_name].attrs['mag_lat']
                    mag_lat = [int(float(x)) for x in mag_lat_str.split(',')]
                    num_y_ticks = 10 
                    ytick_positions = np.linspace(0, len(mag_lat) - 1, num_y_ticks, dtype=int)
                    ytick_labels = [mag_lat[pos] for pos in ytick_positions]
        
                    axs[idx].set_yticks(ytick_positions)
                    axs[idx].set_yticklabels(ytick_labels)
                    axs[idx].set_ylabel('Magnetic Latitude')
                else:
                    axs[idx].set_ylabel('Range Gates')
                fig.colorbar(im, ax=axs[idx], orientation='vertical', fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()


###############################################################################
def main():
    if extract_data:
        h5_path = load_and_preprocess_data_all_beams(
            data_directory,
            negative_value,
            extract_data=True,
            beams_to_process=range(16)
        )       # take all 16 beams but this could be changed to whatever is wanted
    else:
        h5_path = output_h5_file

    inspect_data(h5_path, negative_value, logging_enabled, plotting_enabled)


if __name__ == "__main__":
    main()
