#!/usr/bin/env python
import os
import numpy
import math
import pickle
import matplotlib
matplotlib.use('Agg')


def getSymbol():
    speckp = []
    kpfile = open("KPOINTS", 'r')
    kplines = kpfile.readlines()
    mode = kplines[2].split()[0]

    if mode[0] == "L" or mode[0] == "l":
        kpintervel = int(kplines[1].strip())
        for i in range(4, len(kplines)):
            if kplines[i] != "\n" and len(kplines[i].strip().split()) == 5:
                speckp.append(kplines[i].strip())
    else:
        print("kpoints are Not set as Line-mode!")
        sys.exit()
    kpfile.close()
# print speckp
    kpsym = []

    if len(speckp[0].split()) == 3:
        print("this script is just for KPOINTS with symbol")
        print("for example: 0.000   0.000   0.000   !  G")
        sys.exit()
    elif len(speckp[0].split()) == 4:
        print("keep at least one space between ! and kpoint-symbol in KPOINTS")
        sys.exit()
    elif len(speckp[0].split()) == 5:
        for i in range(0, int(len(speckp) / 2)):
            kpsym.append(speckp[2 * i].split()[-1])
            for j in range(1, kpintervel - 1):
                kpsym.append("")
            kpsym.append(speckp[2 * i + 1].split()[-1])
    return kpsym


def plotBandStructure(filename, Eup, Edown):
    import matplotlib.pyplot as plt

    import numpy as np
    import sys
    from matplotlib.ticker import MultipleLocator
    hkpt = []
    symp = []
    showf = False
    bdf = str(filename) + '-BandStructure'
    if len(sys.argv) == 2 and (str(sys.argv[1]) == 'T' or str(sys.argv[1]) == 't'):
        showf = True
    elif len(sys.argv) >= 2:
        bdf = str(sys.argv[1])

    # get the data
    (Symbols, Numbands, Nkpoints, dis, bande) = pickle.load(
        open('bands.data', 'rb'))
    fig = plt.figure(figsize=(10, 12))
    #plt.scatter(x1, y1, s= z1, color='b', marker='.')
    for kk in range(Numbands):
        plt.plot(dis, bande[kk], 'k-', lw=3)
    #plt.ylim(-10.0, 10.0)

    plt.ylim(Edown, Eup)
    plt.xlim(dis[0], dis[-1])
    plt.axhline(linewidth=2.5, color='r', linestyle='--')
    plt.axes.linewidth = 3.0
    plt.xlabel("Band Structure", fontsize=26)
    plt.ylabel("Energy (eV)", fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    plt.title(str(filename), fontsize=36)

    for kk in range(len(Symbols)):
        if Symbols[kk] != '':
            hkpt.append(kk)
            symp.append(Symbols[kk])
    for ii in range(1, len(hkpt) - 1):
        plt.axvline(dis[hkpt[ii]])
    Symlocs = []
    Symbols = []
    Symlocs.append(dis[0])
    Symbols.append(symp[0])
    for ii in range(len(hkpt)):
        if ii % 2 != 0:
            Symlocs.append(dis[hkpt[ii]])
            Symbols.append(symp[ii])
    plt.xticks(Symlocs, Symbols, fontsize=18)
    if showf:
        plt.show()
    else:
        plt.savefig(bdf + ".png")


def getBandEnergy():
    Numbands = 0
    Nkpoints = 0
    e_fermil = 0
    numbandk = []
    kpostion = []

    # Read the number of fermi level, kpoints and bands.
    numbandk = os.popen("grep 'NBANDS='   OUTCAR").readline().split()
    e_fermil = os.popen("grep 'E-fermi'  OUTCAR").readline().split()[2]

    Nkpoints = int(numbandk[3])
    Numbands = int(numbandk[14])

    # Grep data from OUTCAR.
    getdata1 = "grep 'k-point'                   OUTCAR   > kpt.data"
    getdata2 = "tail -" + str(Nkpoints) + "  kpt.data > kps.data"
    getdata3 = "grep -A" + \
        str(Numbands) + \
        " 'band No.  band energies     occupation' OUTCAR > engery1.data"
    getdata4 = "sed  '/--/d'   engery1.data               > engery.data"
    getdata5 = "rm             kpt.data kps.data engery*.data"
    os.system(getdata1)
    os.system(getdata2)
    os.system(getdata3)
    os.system(getdata4)

    # Get the distance between ajacent kpoints.
    kdata = open('kps.data')
    for ii in range(Nkpoints):
        kpostion.append((kdata.readline().split()[3:]))
    kdata.close()

    ### calculate distance between two neighbor kpoints ###
    dis = []        # increasing distance with kpoints
    distance = 0

    dis.append(0)
    for i in range(1, Nkpoints):
        dx = float(kpostion[i][0]) - float(kpostion[i - 1][0])
        dy = float(kpostion[i][1]) - float(kpostion[i - 1][1])
        dz = float(kpostion[i][2]) - float(kpostion[i - 1][2])
        delta = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2))
        distance += delta
        dis.append(distance)

    # Get the band energy level.
    bande = [[0.0 for kk in range(Nkpoints)] for bb in range(Numbands)]
    maxer = -100.0
    miner = 100
    isdirect = False
    direct_gap = 0.0
    indirect_gap = 0.0
    vbmkp = 0
    cbmkp = 0
    vbmkk = 0
    erbar = 0.0

    edata = open('engery.data')
    for kk in range(Nkpoints):
        edata.readline()
        for bb in range(Numbands):
            tmp = 0.0
            tmp = float(edata.readline().split()[1]) - float(e_fermil)
            bande[bb][kk] = tmp

            if tmp <= 0 and tmp >= maxer:
                maxer = tmp
                vbmkp = kk
                vbmkk = bb
            if tmp >= 0 and tmp <= miner:
                miner = tmp
                cbmkp = kk

    edata.close()
    # get the direct/indirect gap.
    if cbmkp == vbmkp:
        isdirect = True
    direct_gap = bande[vbmkk + 1][vbmkp] - bande[vbmkk][vbmkp]
    indirect_gap = miner - maxer
    print('isdirect, indirect_gap, direct_gap, indirect_gap - direct_gap')
    print(isdirect, indirect_gap, direct_gap, indirect_gap - direct_gap)
    # correct the fermi level.
    erbar = 0.0 - maxer
    for kk in range(Nkpoints):
        for bb in range(Numbands):
            bande[bb][kk] = bande[bb][kk] + erbar
    os.system(getdata5)
    # Get symbols
    Symbols = getSymbol()
    # Save the band structure data.
    pickle.dump((Symbols, Numbands, Nkpoints, dis, bande),
                open('bands.data', 'wb'))
    return maxer


# get the band structure figure.
getBandEnergy()
title = input('title: ')
Eup = float(input('the top of energy of band:'))
Edown = float(input('the bottom of energy of band:'))
plotBandStructure(filename=title, Eup=Eup, Edown=Edown)
