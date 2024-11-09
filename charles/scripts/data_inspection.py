import os
import bz2
import pydarn
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\charles\data\1995"


files = [f for f in os.listdir(data_directory) if f.endswith('.bz2')]

if len(files) == 0:
    print("No files found in the directory.")
else:
    fitacf_file = os.path.join(data_directory, files[0])

    try:
        with bz2.open(fitacf_file, 'rb') as fp:
            fitacf_stream = fp.read()

        # parse files
        sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
        fitacf_data = sdarn_read.read_fitacf()

        if fitacf_data:
            print(f"Data from file: {files[0]}")
            print(f"Number of records: {len(fitacf_data)}")
            print("First 3 records (summary):")
            for i, record in enumerate(fitacf_data[:3]):
                print(f"Record {i + 1}: {{")
                for key, value in list(record.items())[:10]:  # data lim
                    print(f"    {key}: {value}")
                print("    ...")
                print("}")

            # inspect all available keys
            print("\nAll available keys in the first record:")
            for key in fitacf_data[0].keys():
                print(f"  {key}")
        else:
            print("No data found in the file.")

        # data types and dimensions for key features
        print("\nInspecting the three key features (power, velocity, spectral width) in the first record:")
        key_features = ['p_l', 'v', 'w_l']
        if fitacf_data:
            first_record = fitacf_data[0]
            for feature in key_features:
                if feature in first_record:
                    feature_data = first_record[feature]
                    print(f"\nFeature: {feature}")
                    print(f"  Type: {type(feature_data)}")
                    if isinstance(feature_data, (list, tuple)):
                        print(f"  Length: {len(feature_data)}")
                        print(f"  Sample Values: {feature_data[:5]}")
                    else:
                        print(f"  Value: {feature_data}")
                else:
                    print(f"{feature}: Not found in the record")

        # time coverage of the dataset
        times = [record['time'] for record in fitacf_data if 'time' in record]
        if times:
            print("\nTime Coverage:")
            print(f"  Start Time: {min(times)}")
            print(f"  End Time: {max(times)}")
            print(f"  Number of Records: {len(times)}")

        # missing data
        for feature in key_features:
            missing_count = sum(1 for record in fitacf_data if feature in record and record[feature] is None)
            print(f"\nMissing data for {feature}: {missing_count} out of {len(fitacf_data)} records")

        # histograms for spread of features
        plt.figure(figsize=(15, 5))
        for idx, feature in enumerate(key_features):
            feature_values = [record[feature] for record in fitacf_data if feature in record and record[feature] is not None]
            if feature_values:
                plt.subplot(1, 3, idx + 1)
                plt.hist(np.hstack(feature_values), bins=50, alpha=0.7, color='b')
                plt.title(f"Histogram of {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # summary statistics for key features
        for feature in key_features:
            feature_values = [record[feature] for record in fitacf_data if feature in record and record[feature] is not None]
            if feature_values:
                flat_values = np.hstack(feature_values)
                print(f"\nSummary Statistics for {feature}:")
                print(f"  Mean: {np.mean(flat_values):.2f}")
                print(f"  Median: {np.median(flat_values):.2f}")
                print(f"  Standard Deviation: {np.std(flat_values):.2f}")
                print(f"  Min: {np.min(flat_values):.2f}")
                print(f"  Max: {np.max(flat_values):.2f}")

        # visualisation of key parameters pydarn bs
        plt.figure(figsize=(12, 16))
        parameters = [('p_l', "Power (dB)", 'rainbow'), ('v', "Velocity (m/s)", 'gist_rainbow'),
                      ('w_l', "Spectral Width", 'plasma')]
        start_time = dt.datetime(1995, 6, 19, 0, 0)
        end_time = dt.datetime(1995, 6, 19, 1, 0)

        for idx, (parameter, ylabel, cmap) in enumerate(parameters):
            plt.subplot(3, 1, idx + 1)
            pydarn.RTP.plot_coord_time(
                fitacf_data,
                beam_num=10,
                date_fmt='%H:%M',
                parameter=parameter,
                zmax=800 if parameter == 'v' else 30,
                zmin=-800 if parameter == 'v' else 0,
                colorbar_label=ylabel,
                cmap=cmap,
                start_time=start_time,
                end_time=end_time,
                range_estimation=pydarn.RangeEstimation.SLANT_RANGE,
                latlon='lat',
                coords=pydarn.Coords.AACGM,
                plot_equatorward=False
            )
            plt.ylabel(ylabel)
            plt.title(f"Beam 10 - {parameter}")

        plt.tight_layout()
        plt.suptitle(f"SuperDARN Summary Plots for {files[0]}", y=1.02)
        plt.show()

        # save processed data to JSON file
        output_file = os.path.join(data_directory, 'processed_fitacf_data.json')
        with open(output_file, 'w') as f:
            json.dump(fitacf_data[:100], f, indent=4)
        print(f"\nProcessed data saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
