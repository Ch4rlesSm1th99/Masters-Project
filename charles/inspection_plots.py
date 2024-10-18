import bz2
import matplotlib.pyplot as plt
import pydarn

fitacf_file = r"C:\Users\charl\OneDrive\Masters Proj\20020208.1600.00.pgr.fitacf.bz2"
with bz2.open(fitacf_file, 'rb') as fp:
    fitacf_stream = fp.read()

sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
fitacf_data = sdarn_read.read_fitacf()

def plot_parameter(data, parameter, title, ylabel, color='blue'):
    if any(parameter in record for record in data):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        pydarn.RTP.plot_time_series(data, parameter=parameter, color=color)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.show()
    else:
        print(f"No data found for parameter '{parameter}'.")

plot_parameter(fitacf_data, 'tfreq', "Transmit Frequency Time Series", "Frequency (kHz)", color='purple')
plot_parameter(fitacf_data, 'v', "Doppler Velocity Time Series", "Velocity (m/s)", color='blue')
plot_parameter(fitacf_data, 'p_l', "Power Time Series", "Power (dB)", color='orange')
plot_parameter(fitacf_data, 'w_l', "Spectral Width Time Series", "Spectral Width", color='green')
