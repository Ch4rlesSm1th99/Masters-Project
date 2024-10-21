# -*- coding: utf-8 -*-
"""
@author: jaro7
"""
import matplotlib.pyplot as plt
import bz2
import pydarn
import datetime as dt

fitacf_file = 'C:/Users/jaro7/OneDrive/Documents/masters/20020208.1600.00.pgr.fitacf.bz2'
with bz2.open(fitacf_file) as fp:
      fitacf_stream = fp.read()
     
     
fitac_file = 'C:/Users/jaro7/OneDrive/Documents/masters/20020208.1800.00.pgr.fitacf.bz2'
with bz2.open(fitac_file) as fp1:
    fit = fp1.read()

     
reader = pydarn.SuperDARNRead(fitacf_stream, True)
records = reader.read_fitacf()

header = pydarn.SuperDARNRead(fit,True)
re = header.read_fitacf()

data = records + re

time = dt.datetime(2002 , 2 , 8,17 , 20)
tie = dt.datetime(2002,2,8,18,20)


#subplots , axs = 
pydarn.RTP.plot_coord_time(data, beam_num=10, date_fmt='%H%M' , parameter='p_l' , zmax=30,
                           zmin=0, colorbar_label='p',
                           cmap='rainbow' , start_time=time , end_time=tie , range_estimation=pydarn.RangeEstimation.SLANT_RANGE,
                           latlon='lat', coords=pydarn.Coords.AACGM)

#pydarn.RTP.plot_coord_time(data, beam_num=10, date_fmt='%H%M' , parameter='v' , zmax=800,
 #                          zmin=-800, colorbar_label='v',
  #                         cmap='gist_rainbow' , start_time=time , end_time=tie , range_estimation=pydarn.RangeEstimation.SLANT_RANGE,
   #                        latlon='lat', coords=pydarn.Coords.AACGM)

plt.title("Beam 10")


plt.show()