import os
import re

path = '/media/ones/My Passport/workstation/SnO2TWIN544/OPT/orig'

os.chdir(path)

f = open("OUTCAR", 'r')
lines = f.readlines()
f.close()

flag = 0
for line in lines:
    if "POSITION" in line:
        for i in range(40):
            print(lines[flag+i])
    flag += 1
