#!/usr/bin/env python
# coding=utf-8
import os
import shutil
import numpy as np


def MODULATION(delta, fileNumber):
    for i in range(0, fileNumber):
        displace = float(i * delta)
        filename = 'modulation-' + str(displace)
        conf = 'modulation-' + str(displace) + '.conf'
        f = open(conf, 'w')
        f.write('ATOM_Name = SnO2-' + str(displace) + '\n')
        # 之前用于算声子谱的扩胞系数
        f.write('DIM = 2 1 3\n')
        # 阔胞时分析对称性的允许误差
        f.write('SYMMETRY_TOLERANCE = 1e-3\n')
        # 前三个为软膜找相变，将unitcell所阔的胞，由于在0 0.5 0 方向上有虚频，所以在y轴方向上阔胞
        f.write('MODULATION = 1 2 1,0 0.5 0 1 ' + str(displace))
        f.close()
        os.system('phonopy ' + conf)
        os.system('rm MPOSCAR-001')
        os.system('mv MPOSCAR POSCAR-' + str(displace))

    for i in range(0, fileNumber):
        displace = i * delta
        foldname = 'Modulation-' + str(displace)
        if not os.path.exists(foldname):
            os.system('mkdir ' + foldname)
        for file in ['INCAR', 'POTCAR', 'POSCAR-' + str(displace)]:
            shutil.copy2(file, foldname)
        os.chdir(foldname)
        os.rename('POSCAR-' + str(displace), 'POSCAR')
        os.chdir('..')

    os.system('rm modulation-*')
    os.system('rm POSCAR-*')


def getEnergy(delta, fileNumber):

    import re
    arr = np.zeros((fileNumber, 2))
    for i in range(0, fileNumber + 1):
        displace = i * delta
        foldname = 'Modulation-' + str(displace)

        os.chdir(foldname)
        f = open("OUTCAR", 'r')
        lines = f.readlines()
        for line in lines:
            if "energy  without entropy=" in line:
                Enthalpy = re.compile(
                    r"(?<=energy  without entropy=)\s*\-\d+\.?\d*").findall(line)
                Enthalpy = list(map(float, Enthalpy))
        f.close()
        os.chdir('..')
        arr[i][0] = np.array(displace)
        arr[i][1] = np.array(Enthalpy)

    np.savez('Modulation_Energy.npz',distance=arr[:,0],energy=arr[:,1])



def plotModulation_Energy(path,title):
    import matplotlib
    import matplotlib.pyplot as plt
    data=np.load(path)
    distance=data['distance']
    energy=data['energy']
    minE=min(energy)
    plt.plot(distance, energy-minE, "o-.")

    for i in range(0, len(energy)):
        if (energy[i] == minE):
            plt.plot(distance[i],0.0, "ro")
            # print 'min_d: %.1f' %list[0][i]
            plt.annotate(str(minE),xytext=(distance[i]+0.1, 0.0+0.001),xy=(distance[i]+0.1, 0.0+0.0001),weight='bold',color='k',arrowprops=dict(arrowstyle="simple",connectionstyle='arc3',color='red'))

    plt.title(title,fontsize=18, weight="bold",color='#1f618D')
    plt.xlim(-0.1, 20.1)
    plt.tick_params(labelsize=16)
    plt.xlabel(r"$\mathregular{Displacement}$", fontsize=18, weight="bold")
    plt.ylabel(r"$\mathregular{\Delta E\ (eV)}$", fontsize=18, weight="bold")
    plt.tight_layout()
    plt.savefig("Modu-E_"+title+".png", dpi=600)
    plt.show()



if __name__ == '__main__':
    # MODULATION(delta=1.0, fileNumber=20)
    # getEnergy(delta=1.0, fileNumber=20)
    plotModulation_Energy('./Modulation_Energy.npz',title = 'minus3(0.97)')
