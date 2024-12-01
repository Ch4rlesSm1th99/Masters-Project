
import bz2
import pydarn
import glob
import matplotlib.pyplot as plt

path = r"C:\Users\aman\OneDrive - University of Southampton\Desktop\Year 4\MPhys Project\Lo\Masters-Project\aman\1995 Plots\Data\Data Section"

names = glob.glob(f"{path}/*.bz2")
print(names)
def readfile(names):
      data = []
      for name in names:
            with bz2.open(name, 'rb') as fp:
                  fitacf_stream = fp.read()
            reader = pydarn.SuperDARNRead(fitacf_stream, True)
            temp = reader.read_fitacf()
            data.append(temp)
      return(data)

alldata = readfile(names)

beam = 3
y = alldata[-1]

plt.figure(figsize=(7, 5))
pydarn.RTP.plot_range_time(y, beam_num=beam,range_estimation=pydarn.RangeEstimation.RANGE_GATE, cmap = 'rainbow',parameter = "p_l")
plt.title(f"Backscatter Power, Beam: {beam}") 
plt.show()

plt.figure(figsize=(7, 3))
pydarn.RTP.plot_range_time(y, beam_num=beam,range_estimation=pydarn.RangeEstimation.RANGE_GATE, cmap = 'gist_rainbow',parameter = "v")
plt.title(f"Doppler Velocity, Beam: {beam}") 
plt.show()

