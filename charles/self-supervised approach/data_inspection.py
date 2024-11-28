import os
import bz2
import pydarn
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


data_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\1995"


files = [f for f in os.listdir(data_directory) if f.endswith('.bz2')]

if not files:
    print("No files found in the directory.")
else:
    fitacf_file = os.path.join(data_directory, files[0])

    try:
        with bz2.open(fitacf_file, 'rb') as fp:
            fitacf_stream = fp.read()

        sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
        fitacf_data = sdarn_read.read_fitacf()

        if fitacf_data:
            print(f"Data from file: {files[0]}")
            print(f"Number of records: {len(fitacf_data)}")
        else:
            print("No data found in the file.")
            exit()


        records = []
        for record in fitacf_data:
            common_data = {
                'time': dt.datetime(
                    record['time.yr'],
                    record['time.mo'],
                    record['time.dy'],
                    record['time.hr'],
                    record['time.mt'],
                    record['time.sc'],
                    int(record['time.us'] / 1000)
                ),
                'bmnum': record['bmnum'],  # beam number
                'channel': record.get('channel', np.nan),
                'cp': record.get('cp', np.nan),
                'nrang': record['nrang'],  # number of range gates
                'frang': record['frang'],  # distance to the first range gate (km)
                'rsep': record['rsep'],    # range separation (km)
                'stid': record['stid'],    # station ID
            }

            # for each range gate in slist, extract the features
            slist = record['slist']  # list of range gate indices with data
            for idx, gate in enumerate(slist):
                gate_data = common_data.copy()
                gate_data.update({
                    'range_gate': gate,
                    'p_l': record['p_l'][idx],
                    'v': record['v'][idx],
                    'w_l': record['w_l'][idx],
                    'gflg': record['gflg'][idx] if 'gflg' in record else np.nan,  # ground scatter flag
                })
                records.append(gate_data)

        # create a DataFrame
        df = pd.DataFrame(records)

        # convert data types to appropriate formats
        df['bmnum'] = df['bmnum'].astype(int)
        df['stid'] = df['stid'].astype(int)
        df['range_gate'] = df['range_gate'].astype(int)

        # sort the DataFrame by time, beam number, and range gate
        df.sort_values(by=['time', 'bmnum', 'range_gate'], inplace=True)

        print("\nSample of the processed DataFrame:")
        print(df.head())

        # get the unique beams and range gates
        unique_beams = df['bmnum'].unique()
        unique_range_gates = df['range_gate'].unique()

        print(f"\nUnique beam numbers: {unique_beams}")
        print(f"Number of beams: {len(unique_beams)}")
        print(f"Range gates span from {unique_range_gates.min()} to {unique_range_gates.max()}")
        print(f"Total number of range gates: {len(unique_range_gates)}")

        # create a 3D array (time x beams x range gates) for each feature
        beam_of_interest = unique_beams[0]
        df_beam = df[df['bmnum'] == beam_of_interest]

        power_pivot = df_beam.pivot_table(index='time', columns='range_gate', values='p_l')

        print(f"\nPower data for beam {beam_of_interest}:")
        print(power_pivot.head())

        # similarly, you can create pivot tables for 'v' and 'w_l'

        # visualise the power over time for the selected beam
        plt.figure(figsize=(12, 6))
        plt.imshow(power_pivot.T, aspect='auto', extent=[
            power_pivot.index.min(), power_pivot.index.max(),
            unique_range_gates.min(), unique_range_gates.max()
        ], origin='lower', cmap='viridis')
        plt.colorbar(label='Power (p_l)')
        plt.xlabel('Time')
        plt.ylabel('Range Gate')
        plt.title(f'Power over Time for Beam {beam_of_interest}')
        plt.show()

        # inspect the time coverage
        print(f"\nTime coverage from {df['time'].min()} to {df['time'].max()}")
        print(f"Total number of time steps: {df['time'].nunique()}")

        missing_p_l = df['p_l'].isnull().sum()
        missing_v = df['v'].isnull().sum()
        missing_w_l = df['w_l'].isnull().sum()
        print(f"\nMissing data counts:")
        print(f"  p_l: {missing_p_l}")
        print(f"  v: {missing_v}")
        print(f"  w_l: {missing_w_l}")

        print("\nSummary statistics for key features:")
        print(df[['p_l', 'v', 'w_l']].describe())

    except Exception as e:
        print(f"An error occurred: {e}")



