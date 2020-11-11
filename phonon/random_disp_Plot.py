#!/usr/bin/env python
# coding=utf-8
import os
import shutil
import numpy as np


delta = 0.0025
fileNumber = 20
path = '/media/ones/My Passport/2020/ScF3_ICSD/POTCAR(Sc F)/opt(cutoff=520)/x2/OPT/random_disp/random_minus2/random_element/rand_Element'
title = 'random_minus2_rand_element'

def getEnergy(delta, fileNumber):

    import re
    arr = np.zeros((fileNumber, 2))

    for i in range(0, fileNumber):
        flag = (i+1)*delta
        folder = str(flag)+str(0)*(6-len(str(flag)))

        os.chdir(folder)
        f = open("OUTCAR", 'r')
        lines = f.readlines()
        for line in lines:
            if "energy  without entropy=" in line:
                Enthalpy = re.compile(
                    r"(?<=energy  without entropy=)\s*\-\d+\.?\d*").findall(line)
                Enthalpy = list(map(float, Enthalpy))
        f.close()
        os.chdir('..')
        arr[i][0] = np.array(flag)
        arr[i][1] = np.array(Enthalpy)

    np.savez('random_Disp.npz',  distance=arr[:, 0], energy=arr[:, 1])



def plotModulation_Energy(path, title):
    import matplotlib
    import matplotlib.pyplot as plt
    data = np.load(path)
    distance = data['distance']
    energy = data['energy']
    minE = min(energy)
    plt.plot(distance, energy-minE, "o-.")

    for i in range(0, len(energy)):
        if (energy[i] == minE):
            plt.plot(distance[i], 0.0, "ro")
            # print 'min_d: %.1f' %list[0][i]
            plt.annotate(str(minE), xytext=(distance[i], energy[i-1]-energy[i]), xy=(distance[i], (energy[i-1]-energy[i])*0.1), weight='bold', color='k', arrowprops=dict(arrowstyle="simple", connectionstyle='arc3', color='red'))

    plt.title(title,fontsize=18, weight="bold", color='#1f618D')
    plt.xlim(min(distance)*0.9, max(distance)*1.01)
    plt.tick_params(labelsize=16)
    plt.xlabel(r"$\mathregular{\Delta}$", fontsize=18, weight="bold")
    plt.ylabel(r"$\mathregular{\Delta E\ (eV)}$", fontsize=18, weight="bold")
    plt.tight_layout()
    plt.savefig("Modu-E_"+title+".png", dpi=600)
    plt.show()



if __name__ == '__main__':
    os.chdir(path)
    getEnergy(delta, fileNumber)
    plotModulation_Energy('./random_Disp.npz', title)
