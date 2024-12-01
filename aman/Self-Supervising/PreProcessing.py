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
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.T, cmap='rainbow', cbar=True)
    plt.xlabel('Time')
    plt.ylabel('Range Gate')
    plt.title('Heatmap')
    plt.gca().invert_yaxis()  # Flip the y-axis
    plt.show()

#%%
import bz2
import pydarn
path = r"C:\Users\aman\Desktop\MPhys Data\Data\2017\20020208.1600.00.pgr.fitacf.bz2"

with bz2.open(path) as fp:
      fitacf_stream = fp.read()

reader = pydarn.SuperDARNRead(fitacf_stream, True)
test_data = reader.read_fitacf()
pydarn.RTP.plot_range_time(test_data, beam_num=10,range_estimation=pydarn.RangeEstimation.RANGE_GATE,
                            cmap = 'rainbow',parameter = "p_l")
plt.show()
pydarn.RTP.plot_coord_time(test_data, beam_num=10, parameter ='p_l',
                        range_estimation=pydarn.RangeEstimation.SLANT_RANGE,
                        latlon='lat', coords=pydarn.Coords.AACGM,zmin = 0)
plt.show()
power, velocity  = pre_process(test_data)

visualisation(power[10])

#19951220.1800.00.han.fitacf.bz2 first few time steps padding problem as 'p_l' key is missing



# %%
