import  numpy as np
import  pandas as pd
import random
#
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from lib.Shared_Functions import *
filename='rng_1.txt'
file = open(filename, "rb")

byte = file.read(1)
cnt=1

channels=[]
while byte:

    # print(bin(byte))
    byte = file.read(1)

    if len(byte) > 0:
        # print(byte[0])
        if len(byte) > 1:
            print('byte of different size', len(byte))
            break

        channels.append(int(byte[0]))

        cnt=cnt+1
        if cnt > 100e6:
            break

# print(channels)
print(cnt)
# channels2=int(channels)
hist,bin_edges = np.histogram(channels,bins=30)
print(hist)
# #% plot history
# plt.figure(1)
# plt.plot(channels, label='stream')
# plt.legend()

# #% plot history
# plt.figure(2)
# plt.bar(bin_edges[:-1], hist)#, width = 0.9, color='#0504aa',alpha=0.7)
# plt.xlabel('Frequency')
# plt.ylabel('Occupation counts')
# plt.legend()
# plt.show()


## fit to sin
print(hist)
fit_function(hist)


#% plot history
plt.figure(4)
plt.plot(hist,'r', label='Band Occupation Pattern')#, width = 0.9, color='#0504aa',alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Occupation counts')
plt.legend()
plt.show()
