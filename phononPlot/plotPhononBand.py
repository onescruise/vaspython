#!/usr / bin / env python
# coding = utf - 8

import numpy as np
from collections import OrderedDict
import os
import yaml
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.pyplot import subplot
matplotlib.use('Agg')

HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0, 0.0, 0.5]), ('H', [-0.333, 0.667, 0.5]), ('K', [-0.333, 0.667, 0.0]),
                       ('$\Gamma$ ', [0.0, 0.0, 0.0]), ('M', [0.0, 0.5, 0.0]), ('L', [0.0, 0.5, 0.5]), ('H ', [-0.333, 0.667, 0.5])]


class PhononyBand():
    def __init__(self, path=os.getcwd(), HighlySymmetricPath=None):
        self.path = path
        self.HighlySymmetricPath = HighlySymmetricPath

    def readBandYaml(self):
        '''
        Extract the information in band.yaml

        you can get:
        bandParameters;
        kPosition, kDistance, eigenvectors;
        '''

        '''(..)===== Black Box,you can ignored me =====(..)'''
        def _ConvertYaml():
            with open(self.path + '/band.yaml', 'r') as f:
                '''read the diction in band.yaml as bandFile '''
                bandFile = OrderedDict()
                bandFile = yaml.load(f)

                '''reprocess the bandFile as dict and return dict'''
                dict = OrderedDict()
                dict['nqpoint'] = int(bandFile['nqpoint'])
                dict['npath'] = int(bandFile['npath'])
                dict['natom'] = int(bandFile['natom'])
                dict['reciprocal_lattice'] = np.array(
                    bandFile['reciprocal_lattice'])
                dict['lattice'] = np.array(bandFile['lattice'])
                dict['phonon'] = np.array(bandFile['phonon'])

                return dict

        def _bandinfo(bandParameters):
            Natoms = np.array((bandParameters['natom']))
            kPosition = np.zeros((bandParameters['nqpoint'], 3))
            kDistance = np.zeros(bandParameters['nqpoint'])
            eigenvectors = np.zeros(
                (bandParameters['nqpoint'], bandParameters['natom'] * 3))

            PhononBand = bandParameters['phonon']
            for x in range(bandParameters['nqpoint']):
                dict = PhononBand[x]
                kPosition[x, :] = np.array(dict['q-position'])
                kDistance[x] = float(dict['distance'])
                for y in range(bandParameters['natom'] * 3):
                    eigenvectors[x][y] = dict['band'][y]['frequency']

            return Natoms, kPosition, kDistance, eigenvectors

        def _getHighlySymmetricPath():
            list = []
            start = 0
            for value in HighlySymmetricPath.values():
                for i in range(start, len(kPosition)):
                    if (np.array(value) == kPosition[i]).all():
                        list.append(kDistance[i])
                        start = i
                        break
            return list
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        bandParameters = _ConvertYaml()
        Natoms, kPosition, kDistance, eigenvectors = _bandinfo(bandParameters)

        HighlySymmetricPath = OrderedDict(self.HighlySymmetricPath)
        distance_HighlySymmetricPath = _getHighlySymmetricPath()

        return Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath
        '''[^-^]Important things say three times!!![^-^]'''


def SinglePicture():
    inputpath = 'F:/2020/C_Si_Ge/Si111/four_bilayers/PHO/orig'
    # HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0000000000,0.0000000000,0.5000000000]), ('H', [-0.333333333, 0.6666666667, 0.5000000000]), ('K', [-0.333333333,0.6666666667,0.0000000000]),('$\Gamma$', [0.0, 0.0, 0.0]), ('M', [0.0000000000,0.5000000000,0.0000000000]), ('L', [0.0, 0.5, 0.5]), ('H', [-0.333333333,0.6666666667,0.5000000000])]
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()
    '''[^-^]Look here! Look here! Look here![^-^]'''

    plt.figure(figsize=(9, 3))

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        plt.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    plt.xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath:
        plt.axvline(x=xHSP, color='k', linestyle='--',)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xticks(distance_HighlySymmetricPath,
               OrderedDict(HighlySymmetricPath).keys())

    # set y axis
    # title = input('#Title:')
    filename = input('#filename:')
    # plt.title(title, fontsize=28)
    plt.ylim(0, 20)
    plt.ylabel('$\omega(THz)$', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)

    '''[^-^]Important things say three times!!![^-^]'''


def AdjacentPictures():

    # HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0000000000,0.0000000000,0.5000000000]), ('H', [-0.333333333, 0.6666666667, 0.5000000000]), ('K', [-0.333333333,0.6666666667,0.0000000000]),('$\Gamma$', [0.0, 0.0, 0.0]), ('M', [0.0000000000,0.5000000000,0.0000000000]), ('L', [0.0, 0.5, 0.5]), ('H', [-0.333333333,0.6666666667,0.5000000000])]
    plt.figure(figsize=(18, 10), dpi=600)
    plt.subplots_adjust(hspace=0, wspace=0.05)

    ax10 = plt.subplot2grid((3, 6), (0, 0), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/C/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax10.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax10.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax10.axvline(x=xHSP, color='k', linestyle='--')
    ax10.set_xticklabels('')
    ax10.set_ylim(-5, 40)
    ax10.set_yticks(np.arange(0, 41, 10))
    ax10.set_yticklabels(np.arange(0, 41, 10), fontsize='18')
    ax10.tick_params(labelsize='18')
    ax10.text(0.01, -0.5 * 5, '(a)', fontsize='18')
    # ax10.set_ylabel('$\omega(THz)$', fontsize='18')

    ax11 = plt.subplot2grid((3, 6), (0, 2), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/C111/two_bilayers/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax11.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax11.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax11.axvline(x=xHSP, color='k', linestyle='--')
    ax11.set_xticklabels('')
    ax11.set_ylim(-5, 40)
    ax11.set_yticks(np.arange(0, 41, 10))
    ax11.set_yticklabels('')
    ax11.tick_params(labelsize='18')
    ax11.text(0.01, -0.5 * 5, '(d)', fontsize='18')
    # ax11.set_ylabel('$\omega(THz)$', fontsize='18')

    ax12 = plt.subplot2grid((3, 6), (0, 4), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/C111/four_bilayers/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax12.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax12.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax12.axvline(x=xHSP, color='k', linestyle='--')
    ax12.set_xticklabels('')
    ax12.set_ylim(-5, 40)
    ax12.set_yticks(np.arange(0, 41, 10))
    ax12.set_yticklabels('')
    ax12.tick_params(labelsize='18')
    ax12.text(0.01, -0.5 * 5, '(g)', fontsize='18')
    # ax12.set_ylabel('$\omega(THz)$', fontsize='18')

    ax20 = plt.subplot2grid((3, 6), (1, 0), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/Si/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax20.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax20.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax20.axvline(x=xHSP, color='k', linestyle='--')
    ax20.set_xticklabels('')
    ax20.set_ylim(-2.5, 20)
    ax20.set_yticks(np.arange(0, 21, 5))
    ax20.set_yticklabels(np.arange(0, 21, 5), fontsize='18')
    ax20.tick_params(labelsize='18')
    ax20.set_ylabel('$\omega(THz)$', fontsize='18')
    ax20.text(0.01, -0.5 * 2.5, '(b)', fontsize='18')

    ax21 = plt.subplot2grid((3, 6), (1, 2), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/Si111/two_bilayers/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax21.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax21.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax21.axvline(x=xHSP, color='k', linestyle='--')
    ax21.set_xticklabels('')
    ax21.set_ylim(-2.5, 20)
    ax21.set_yticks(np.arange(0, 21, 5))
    ax21.set_yticklabels('')
    ax21.tick_params(labelsize='18')
    ax21.text(0.01, -0.5 * 2.5, '(e)', fontsize='18')
    # ax12.set_ylabel('$\omega(THz)$', fontsize='18')

    ax22 = plt.subplot2grid((3, 6), (1, 4), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/Si111/four_bilayers/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax22.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax22.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax22.axvline(x=xHSP, color='k', linestyle='--')
    ax22.set_xticklabels('')
    ax22.set_ylim(-2.5, 20)
    ax22.set_yticks(np.arange(0, 21, 5))
    ax22.set_yticklabels('')
    ax22.tick_params(labelsize='18')
    ax22.text(0.01, -0.5 * 2.5, '(h)', fontsize='18')

    ax30 = plt.subplot2grid((3, 6), (2, 0), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/Ge/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax30.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax30.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax30.axvline(x=xHSP, color='k', linestyle='--')
    ax30.set_xticks(distance_HighlySymmetricPath)
    ax30.set_xticklabels(OrderedDict(
        HighlySymmetricPath).keys(), fontsize='18')
    ax30.set_ylim(-1.25, 10)
    ax30.set_yticks(np.arange(0, 11, 2.5))
    ax30.set_yticklabels([0, '', 5, '', 10], fontsize='18')
    ax30.tick_params(labelsize='18')
    ax30.text(0.01, -0.5 * 1.25, '(c)', fontsize='18')
    # ax30.set_ylabel('$\omega(THz)$', fontsize='18')

    ax31 = plt.subplot2grid((3, 6), (2, 2), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/Ge111/two_bilayers/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax31.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax31.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax31.axvline(x=xHSP, color='k', linestyle='--')
    ax31.set_xticks(distance_HighlySymmetricPath)
    ax31.set_xticklabels(OrderedDict(
        HighlySymmetricPath).keys(), fontsize='18')
    ax31.set_ylim(-1.25, 10)
    ax31.set_yticks(np.arange(0, 11, 2.5))
    ax31.set_yticklabels('')
    ax31.tick_params(labelsize='18')
    ax31.text(0.01, -0.5 * 1.25, '(f)', fontsize='18')
    # ax31.set_ylabel('$\omega(THz)$', fontsize='18')

    ax32 = plt.subplot2grid((3, 6), (2, 4), colspan=2, rowspan=1)
    inputpath = 'F:/2020/C_Si_Ge/Ge111/four_bilayers/PHO/orig'
    PB = PhononyBand(inputpath, HighlySymmetricPath)
    Natoms, kPosition, kDistance, eigenvectors, distance_HighlySymmetricPath = PB.readBandYaml()

    colors = plt.cm.jet(np.linspace(0, 1, Natoms * 3))
    for y in range(Natoms * 3):
        # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
        ax32.plot(kDistance, eigenvectors[:, y], color='blue')

    # set x axis
    ax32.set_xlim(kDistance[0], kDistance[-1])
    for xHSP in distance_HighlySymmetricPath[1:-1]:
        ax32.axvline(x=xHSP, color='k', linestyle='--')
    ax32.set_xticks(distance_HighlySymmetricPath)
    ax32.set_xticklabels(OrderedDict(
        HighlySymmetricPath).keys(), fontsize='18')
    ax32.set_ylim(-1.25, 10)
    ax32.set_yticks(np.arange(0, 11, 2.5))
    ax32.set_yticklabels('')
    ax32.tick_params(labelsize='18')
    ax32.text(0.01, -0.5 * 1.25, '(i)', fontsize='18')

    # title = input('#Title:')
    # filename = input('#filename:')
    # plt.title(title, fontsize=28)
    # plt.ylim(Yrange[0], Yrange[1])
    # plt.xticks(distance_HighlySymmetricPath,
    #            OrderedDict(HighlySymmetricPath).keys())
    plt.tick_params(labelsize=18)
    # plt.tight_layout()
    plt.savefig('test1', dpi=600)


if __name__ == "__main__":
    # SinglePicture()
    AdjacentPictures()
