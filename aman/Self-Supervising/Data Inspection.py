#%%
import bz2
import pydarn
import glob
import matplotlib.pyplot as plt
import numpy as np
path = r"C:\MPhys Data\Data\Data Section\19951230.1600.00.han.fitacf.bz2"


with bz2.open(path) as fp:
      fitacf_stream = fp.read()

reader = pydarn.SuperDARNRead(fitacf_stream, True)
datai = reader.read_fitacf()
#%%
def inspect_data(data):
    num_records = len(data)
    beam_counts = np.zeros(16, dtype=int)
    time_steps = 0
    current_beam = -1

    for record in data:
        bmnum = record.get('bmnum', None)
        if bmnum is not None and bmnum < 16:
            beam_counts[bmnum] += 1
            if bmnum == 0 and current_beam != 0:
                time_steps += 1
            current_beam = bmnum

    # Ensure the last set of beams is counted as a time step
    if current_beam == 15:
        time_steps += 1

    print("Number of Records:", num_records)
    print("Beam Counts:", beam_counts)
    print("Total Number of Beams:", len(beam_counts))
    print("Number of Time Steps:", time_steps)

    return beam_counts, time_steps
inspect_data(datai)
# %%
def pre_process(data):
    number_of_records = len(data)
    number_of_beams = 16
    total_time_steps = sum(1 for record in data if record['bmnum'] == 0)
    
    power = [[] for _ in range(number_of_beams)]
    range_gate = [[] for _ in range(number_of_beams)]
    velocity = [[] for _ in range(number_of_beams)]

    for time_step in range(number_of_records):

        beam_number = data[time_step].get('bmnum', None)

        if beam_number is not None and beam_number < number_of_beams:
            power[beam_number].append(data[time_step]['p_l'])
            range_gate[beam_number].append(data[time_step]['slist'])
            velocity[beam_number].append(data[time_step]['v'])

    padded_power = [[[0] * 75 for _ in range(total_time_steps)] for _ in range(number_of_beams)]
    padded_velocity = [[[0] * 75 for _ in range(total_time_steps)] for _ in range(number_of_beams)]

    for beam_number in range(number_of_beams):
        for time_step in range(total_time_steps):
            for pixel in range(len(range_gate[beam_number][time_step])):
                index = range_gate[beam_number][time_step][pixel]-1
                padded_power[beam_number][time_step][index] = power[beam_number][time_step][pixel]
                padded_velocity[beam_number][time_step][index] = velocity[beam_number][time_step][pixel]

    padded_power = np.array(padded_power)
    padded_velocity = np.array(padded_velocity)
    return(padded_power, padded_velocity)