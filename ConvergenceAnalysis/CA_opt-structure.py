import matplotlib.pyplot as plt
import matplotlib
import os
import re
import numpy as np
lines = os.popen('grep F= run.log').read().strip().split('\n')

F = []
E0 = []
dE = []
step_num = list(range(len(lines)))
print(step_num)
for line in lines:
    F += re.findall(r".*F=(.*)E0", line)
    E0 += re.findall(r".*E0=(.*)d E", line)
    dE += re.findall(r".*d E =(.*)", line)


# plt.plot(step_num, np.array(F).astype(float))
plt.plot(step_num, np.array(E0).astype(float))
# plt.plot(step_num[10:], np.array(dE).astype(float)[10:])
plt.show()
# LORBIT += list(map(float, LORBIT))
# if LORBIT != []:
#     parameter['LORBIT'] = LORBIT[0]  # print(line)
