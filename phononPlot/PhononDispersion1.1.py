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


class PhononyBand():
    def __init__(self, path=os.getcwd()):
        self.path = path

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

            return kPosition, kDistance, eigenvectors
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        bandParameters = _ConvertYaml()
        kPosition, kDistance, eigenvectors = _bandinfo(bandParameters)
        return bandParameters, kPosition, kDistance, eigenvectors
        '''[^-^]Important things say three times!!![^-^]'''

    def plotBand(self, HighlySymmetricPath, Yrange):
        '''(..)===== Black Box,you can ignored me =====(..)'''

        def _getHighlySymmetricPath():
            list = []
            start = 0
            print(HighlySymmetricPath)
            for value in HighlySymmetricPath.values():
                for i in range(start, len(kPosition)):
                    if (np.array(value) == kPosition[i]).all():
                        list.append(kDistance[i])
                        start = i
                        break
            return list
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        bandParameters, kPosition, kDistance, eigenvectors = self.readBandYaml()
        HighlySymmetricPath = OrderedDict(HighlySymmetricPath)
        distance_HighlySymmetricPath = _getHighlySymmetricPath()

        title = input('#Title:')

        plt.figure(figsize=(8, 8))

        colors = plt.cm.jet(np.linspace(0, 1, bandParameters['natom'] * 3))
        for y in range(bandParameters['natom'] * 3):
            # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
            plt.plot(kDistance, eigenvectors[:, y], color='black')

        # set x axis
        plt.xlim(kDistance[0], kDistance[-1])
        for xHSP in distance_HighlySymmetricPath:
            plt.axvline(x=xHSP, color='k', linestyle='--',)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xticks(distance_HighlySymmetricPath, HighlySymmetricPath.keys())

        # set y axis
        plt.title(title, fontsize=28)
        plt.ylim(Yrange[0], Yrange[1])
        plt.ylabel('$\omega(THz)$', fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig(self.path + "/PhononyBand.png", dpi=600)

        '''[^-^]Important things say three times!!![^-^]'''


# class GruneisenBand():
#     def __init__(self, path=os.getcwd()):
#         self.path = path
#
#     def getGruneisenBand(self, filelist, dim, HighlySymmetricPath):
#
#         os.chdir(self.path)
#
#         hsp = OrderedDict(HighlySymmetricPath)
#         hsplist = ''
#         for key in hsp.keys():
#             line = str(hsp[key][0]) + ' ' + str(hsp[key][1]) + \
#                 ' ' + str(hsp[key][2]) + '\t'
#             hsplist += str(line)
#         print(hsplist)
#         cmd = str(filelist) + ' --tolerance=1e-3' + ' --dim=' + '\"' + str(dim) + \
#             '\" ' + ' --band=' + '\"' + hsplist + '\"' + ' -p -s -c POSCAR'
#         os.system('phonopy-gruneisen ' + cmd)
#
#     def readGruneisenYaml(self):
#         '''
#         Extract the information in gruneisen.yaml
#
#         you can get:
#         bandParameters;
#         kPosition, kDistance, eigenvectors;
#         '''
#
#         '''(..)===== Black Box,you can ignored me =====(..)'''
#         def _ConvertYaml():
#             with open(self.path + '/gruneisen.yaml', 'r') as f:
#                 '''read the diction in gruneisen.yaml as bandFile '''
#                 bandFile = OrderedDict()
#                 bandFile = yaml.load(f)
#
#                 '''reprocess the bandFile as dict and return dict'''
#                 dict = OrderedDict()
#
#                 # dict['npath'] = int(bandFile['npath'])
#
#                 # dict['reciprocal_lattice'] = np.array(bandFile['reciprocal_lattice'])
#                 # dict['lattice'] = np.array(bandFile['lattice'])
#                 dict['phonon'] = np.array(bandFile['path'][0]['phonon'])
#                 dict['nqpoint'] = int(
#                     len(dict['phonon'])) * (len(dict['path']))
#                 dict['natom'] = int(len(dict['phonon'][0]['band']) / 3)
#
#                 return dict
#
#         def _bandinfo(bandParameters):
#             kPosition = np.zeros((bandParameters['nqpoint'], 3))
#             kDistance = np.zeros(bandParameters['nqpoint'])
#             eigenvectors = np.zeros(
#                 (bandParameters['nqpoint'], bandParameters['natom'] * 3))
#             gruneisen = np.zeros(
#                 (bandParameters['nqpoint'], bandParameters['natom'] * 3))
#
#             PhononBand = bandParameters['phonon']
#             for x in range(bandParameters['nqpoint']):
#                 dict = PhononBand[x]
#                 kPosition[x, :] = np.array(dict['q-position'])
#                 kDistance[x] = float(dict['distance'])
#                 for y in range(bandParameters['natom'] * 3):
#                     eigenvectors[x][y] = dict['band'][y]['frequency']
#                     gruneisen[x][y] = dict['band'][y]['gruneisen']
#
#             return kPosition, kDistance, eigenvectors, gruneisen
#         '''(..) ========================= (..)'''
#
#         '''[^-^]Look here! Look here! Look here![^-^]'''
#         bandParameters = _ConvertYaml()
#         kPosition, kDistance, eigenvectors, gruneisen = _bandinfo(
#             bandParameters)
#         return bandParameters, kPosition, kDistance, eigenvectors, gruneisen
#         '''[^-^]Important things say three times!!![^-^]'''

    # def plotGruneisen(self, title, HighlySymmetricPath, Yrange):
    #     '''(..)===== Black Box,you can ignored me =====(..)'''
    #     def _getHighlySymmetricPath():
    #         list = []
    #         start = 0
    #         for value in HighlySymmetricPath.values():
    #             for i in range(start, len(kPosition)):
    #                 print(kPosition[i])
    #                 if (np.array(value) == kPosition[i]).all():
    #                     list.append(kDistance[i])
    #                     start = i + 1
    #                     break
    #         print(list)
    #         return list
    #     '''(..) ========================= (..)'''
    #
    #     '''[^-^]Look here! Look here! Look here![^-^]'''
    #     bandParameters, kPosition, kDistance, eigenvectors, gruneisen = self.readGruneisenYaml()
    #     HighlySymmetricPath = OrderedDict(np.array(HighlySymmetricPath))
    #     distance_HighlySymmetricPath = _getHighlySymmetricPath()
    #
    #     plt.figure(figsize=(8, 8))
    #
    #     colors = plt.cm.jet(np.linspace(0, 1, bandParameters['natom'] * 3))
    #     for y in range(bandParameters['natom'] * 3):
    #         plt.plot(kDistance, gruneisen[:, y], color='Black')
    #
    #     # set x axis
    #     # plt.xlim(kDistance[0], kDistance[-1])
    #     for xHSP in distance_HighlySymmetricPath:
    #         plt.axvline(x=xHSP, color='k', linestyle='--',)
    #     plt.xticks(distance_HighlySymmetricPath, HighlySymmetricPath.keys())
    #
    #     # set y axis
    #     plt.title(title, fontsize=40)
    #     plt.ylim(Yrange[0], Yrange[1])
    #     plt.ylabel('$\omega(THz)$', fontsize=18)
    #     plt.tick_params(labelsize=18)
    #     plt.tight_layout()
    #     plt.savefig("PhononyGruneisen.png", dpi=600)
    #
    #     '''[^-^]Important things say three times!!![^-^]'''


if __name__ == "__main__":
    # 画出了SnO2TWIN的声子谱
    if False:
        for name in ['plus3', 'plus2', 'plus1', 'minus3', 'minus2', 'minus1', 'origin']:
            path = '/media/ones/My Passport/workstation/SnO2TWIN544/PHO/' + name
            PhoB = PhononyBand(path)
            # PhoB.readBandYaml()
            HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('Z', [0.0, 0.0, 0.5]), ('T', [-0.5, 0.0, 0.5]), ('Y', [-0.5, 0.0, 0.0]),
                                   ('S', [-0.5, 0.5, 0.0]), ('X', [0.0, 0.5, 0.0]), ('U', [0.0, 0.5, 0.5]), ('R', [-0.5, 0.5, 0.5])]
            # HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0000000000,0.0000000000,0.5000000000]), ('H', [-0.333333333, 0.6666666667, 0.5000000000]), ('K', [-0.333333333,0.6666666667,0.0000000000]),('$\Gamma$', [0.0, 0.0, 0.0]), ('M', [0.0000000000,0.5000000000,0.0000000000]), ('L', [0.0, 0.5, 0.5]), ('H', [-0.333333333,0.6666666667,0.5000000000])]
            PhoB.plotBand(HighlySymmetricPath)
        '''================================================='''

    # 画出了SnO2TWIN的声子谱
    if True:
        path = 'F:/workstation/SnO2TWIN544/PRI/pho'
        PhoB = PhononyBand(path)
        # PhoB.readBandYaml()
        HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('Z', [0.0, 0.0, 0.5]), ('T', [-0.5, 0.0, 0.5]), ('Y', [-0.5, 0.0, 0.0]),
                               ('S', [-0.5, 0.5, 0.0]), ('X', [0.0, 0.5, 0.0]), ('U', [0.0, 0.5, 0.5]), ('R', [-0.5, 0.5, 0.5])]
        # HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0000000000,0.0000000000,0.5000000000]), ('H', [-0.333333333, 0.6666666667, 0.5000000000]), ('K', [-0.333333333,0.6666666667,0.0000000000]),('$\Gamma$', [0.0, 0.0, 0.0]), ('M', [0.0000000000,0.5000000000,0.0000000000]), ('L', [0.0, 0.5, 0.5]), ('H', [-0.333333333,0.6666666667,0.5000000000])]
        PhoB.plotBand(HighlySymmetricPath=HighlySymmetricPath,
                      Yrange=[-2.5, 25])
    '''================================================='''

# 画金刚石的TWIN
    if False:
        path = 'F:/2020/Si111/two_bilayers/pho'
        PhoB = PhononyBand(path)
        os.chdir(path)
        # os.system('phonopy -f disp-001/vasprun.xml')
        os.system('phonopy --tolerance=1e-3 -p -s band.conf')
        HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0, 0.0, 0.5]), ('H', [-0.333, 0.667, 0.5]), ('K', [-0.333, 0.667, 0.0]),
                               ('$\Gamma$ ', [0.0, 0.0, 0.0]), ('M', [0.0, 0.5, 0.0]), ('L', [0.0, 0.5, 0.5]), ('H ', [-0.333, 0.667, 0.5])]
        PhoB.plotBand(
            title='Si orig twin16', HighlySymmetricPath=HighlySymmetricPath, Yrange=[-2.5, 40])

    if False:
        path = '/media/ones/My Passport/2020'
        GruB = GruneisenBand(path)

        orig = 'one_layers_pho_pbe_1.00'
        plus = 'one_layers_pho_pbe_1.01'
        minus = 'one_layers_pho_pbe_0.99'
        filelist = orig + ' ' + plus + ' ' + minus

        dim = '1 3 4'

        HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0, 0.0, 0.5]), ('H', [-0.333, 0.667, 0.5]), ('K', [-0.333, 0.667, 0.0]),
                               ('$\Gamma$ ', [0.0, 0.0, 0.0]), ('M', [0.0, 0.5, 0.0]), ('L', [0.0, 0.5, 0.5]), ('H ', [-0.333, 0.667, 0.5])]

        GruB.getGruneisenBand(filelist, dim, HighlySymmetricPath)
        # GruB.plotGruneisen(HighlySymmetricPath)

# ScF3
    if False:
        path = '''/media/ones/My Passport/2020/ScF3_ICSD/POTCAR(Sc F)/opt(cutoff=520)/x2/OPT/random_disp/random_minus2/random_disp_minus2_0.01/test1'''
        PhoB = PhononyBand(path)
        # HighlySymmetricPath=[('X', [0.5, 0.0, 0.0]), ('R', [0.5, 0.5, 0.5]), ('M', [0.5, 0.5, 0.0]), ('G', [0.0, 0.0, 0.0]), ('R ', [0.5, 0.5, 0.5])]
        HighlySymmetricPath = [('G', [0.0, 0.0, 0.0]), ('X', [0.5, 0.0, 0.0]),
                               ('M', [0.5, 0.5, 0.0]), ('G ', [0.0, 0.0, 0.0]), ('R ', [0.5, 0.5, 0.5]), ('X ', [0.5, 0.0, 0.0])]
        os.chdir(path)
        os.system('phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml')
        os.system('phonopy --tolerance=1e-3 -p -s band.conf')
        PhoB.plotBand(title='ICSD_Sc_cutoff=400K',
                      HighlySymmetricPath=HighlySymmetricPath, Yrange=[-5, 25])

# Si
    if False:
        path = '''/media/ones/My Passport/2020/C/pho'''
        PhoB = PhononyBand(path)
        HighlySymmetricPath = [('G', [0.0, 0.0, 0.0]), ('X', [0.5, 0.0, 0.5]), ('L', [
            0.5, 0.5, 0.5]), ('W', [0.5, 0.25, 0.75])]
        os.chdir(path)
        os.system('phonopy -f disp-001/vasprun.xml')
        os.system('phonopy --tolerance=1e-3 -p -s band.conf')
        PhoB.plotBand(title='ICSD_Si_pho_sf=0.97',
                      HighlySymmetricPath=HighlySymmetricPath, Yrange=[-5, 25])

"""
import re
with open('band.yaml', 'r') as bandFile:
    line = bandFile.readline()
    while(line):
        if 'nqpoint' in line:
            p = re.compile(r"(?<=nqpoint:)\s*\d+\.?\d*")
            nqpoint = list(map(int, p.findall(line)))
            if nqpoint != []:
                self.bandParameters['nqpoints'] = nqpoint[0]
"""
