import bz2
import matplotlib.pyplot as plt
import pydarn


fitacf_file = r"C:\Users\charl\OneDrive\Masters Proj\20020208.1600.00.pgr.fitacf.bz2"
with bz2.open(fitacf_file, 'rb') as fp:
    fitacf_stream = fp.read()

sdarn_read = pydarn.SuperDARNRead(fitacf_stream, True)
fitacf_data = sdarn_read.read_fitacf()

plt.figure(figsize=(12, 8))
pydarn.RTP.plot_summary(fitacf_data)
plt.suptitle("SuperDARN Summary Plot")
plt.show()
