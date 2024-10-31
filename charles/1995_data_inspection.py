import os
import bz2
import pydarn
import matplotlib.pyplot as plt
import datetime as dt


data_directory = r"C:\Users\charl\PycharmProjects\Masters_Project\Masters-Project\data\1995"


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
        else:
            print("No data found in the file.")


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
                coords=pydarn.Coords.AACGM
            )
            plt.ylabel(ylabel)
            plt.title(f"Beam 10 - {parameter}")

        plt.tight_layout()
        plt.suptitle(f"SuperDARN Summary Plots for {files[0]}", y=1.02)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
