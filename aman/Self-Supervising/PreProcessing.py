#%%
import numpy as np


def pre_process(data):
    beam_numbers = sorted({record['bmnum'] for record in data if 'bmnum' in record})
    beam_to_index = {beam_number: idx for idx, beam_number in enumerate(beam_numbers)}
    num_beams = len(beam_numbers)
    max_range_gate = 76 

    
    time_steps = []
    current_time_step_records = []
    beams_in_current_time_step = set()

    for record in data:
        beam_number = record.get('bmnum')

        if beam_number in beams_in_current_time_step:
            time_steps.append(current_time_step_records)
            current_time_step_records = []
            beams_in_current_time_step = set()

        beams_in_current_time_step.add(beam_number)
        current_time_step_records.append(record)

    
    if current_time_step_records:
        time_steps.append(current_time_step_records)

    total_time_steps = len(time_steps)
    power = np.zeros((num_beams, total_time_steps, max_range_gate))
    velocity = np.zeros((num_beams, total_time_steps, max_range_gate))

   
    min_velocity_per_beam = {beam_idx: None for beam_idx in range(num_beams)}

    
    for time_step_records in time_steps:
        for record in time_step_records:
            beam_number = record.get('bmnum')
            beam_idx = beam_to_index[beam_number]

            if 'v' in record:
                velocity_values = np.array(record['v'])
                min_velocity = velocity_values.min()

                current_min = min_velocity_per_beam[beam_idx]
                if current_min is None or min_velocity < current_min:
                    min_velocity_per_beam[beam_idx] = min_velocity

    for time_step_index, time_step_records in enumerate(time_steps):
        for record in time_step_records:
            beam_number = record.get('bmnum')
            beam_idx = beam_to_index[beam_number]

            if 'slist' in record:
                range_gates = np.array(record['slist'])
            else:
                continue  

            
            if 'p_l' in record:
                power_values = np.array(record['p_l']) + 1  
            else:
                power_values = np.zeros_like(range_gates)

            power[beam_idx, time_step_index, range_gates] = power_values

            
            if 'v' in record:
                velocity_values = np.array(record['v'])
                min_velocity = min_velocity_per_beam[beam_idx] or 0
                
                velocity_values = velocity_values - min_velocity + 1
                velocity[beam_idx, time_step_index, range_gates] = velocity_values
            else:
                velocity_values = np.zeros_like(range_gates)

    return power, velocity
#%%
import seaborn as sns
import matplotlib.pyplot as plt
def visualisation(data):
    data = np.where(data == -1, np.nan, data)  # Replace -1 with NaN
    plt.figure(figsize=(10, 6))  # Adjust the figure size to be more squished vertically
    ax = sns.heatmap(data.T, cmap='rainbow', cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Power (dB)', fontsize=12)  # Add label to the color bar with increased font size
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Range Gate', fontsize=12)
    plt.title('Heatmap', fontsize=14)
    plt.gca().invert_yaxis()  # Flip the y-axis
    plt.xlim([0, 120])  # Assuming each time step is 1 minute, adjust as needed
    plt.xticks(ticks=[0, 30, 60, 90, 120], labels=['18:00', '18:30', '19:00', '19:30', '20:00'], fontsize=10, rotation=0)
    plt.yticks(ticks=[0,10,20,30,40,50,60,70],labels = [0,10,20,30,40,50,60,70],fontsize=10)
    plt.show()

#%%
import bz2
import pydarn
from datetime import datetime
path = r"C:\Users\aman\Desktop\MPhys Data\Data\2017\20020208.1600.00.pgr.fitacf.bz2"
path2 = r"C:\Users\aman\Desktop\MPhys Data\Data\2017\20020208.1800.00.pgr.fitacf.bz2"
with bz2.open(path) as fp:
      fitacf_stream = fp.read()

with bz2.open(path2) as fp:
      fitacf_stream2 = fp.read()

reader1 = pydarn.SuperDARNRead(fitacf_stream, True)
test_data1 = reader1.read_fitacf()
reader2 = pydarn.SuperDARNRead(fitacf_stream2, True)
test_data2 = reader2.read_fitacf()
test_data=test_data1+test_data2
#%%
from datetime import datetime
#start_time = datetime.strptime("17:20", "%H:%M")
#end_time = datetime.strptime("18:20", "%H:%M")
plt.figure(figsize=(10, 3.5))
plt.xlabel('Time', fontsize=14)
plt.title("Beam 10, Prince George Radar", fontsize=14)
plt.ylabel('Range Gates', fontsize=14)
pydarn.RTP.plot_range_time(test_data, beam_num=10, range_estimation=pydarn.RangeEstimation.RANGE_GATE,
                           cmap='rainbow', parameter="p_l",colorbar_label=("Power (dB)"),zmax=27,zmin=0)
plt.show()
#pydarn.RTP.plot_coord_time(test_data, beam_num=10, parameter='p_l',
#                           range_estimation=pydarn.RangeEstimation.SLANT_RANGE,
#                           latlon='lat', coords=pydarn.Coords.AACGM, zmin=0)
#plt.show()
#power, velocity = pre_process(test_data)

#visualisation(power[10])

#19951220.1800.00.han.fitacf.bz2 first few time steps padding problem as 'p_l' key is missing


# %%
