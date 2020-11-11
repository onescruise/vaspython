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

    def readGruneisenYaml(self):
        '''
        Extract the information in gruneisen.yaml

        you can get:
        bandParameters;
        kPosition, kDistance, eigenvectors;
        '''

        '''(..)===== Black Box,you can ignored me =====(..)'''
        def _ConvertYaml():
            with open(self.path + '/gruneisen.yaml', 'r') as f:
                '''read the diction in gruneisen.yaml as bandFile '''
                bandFile = OrderedDict()
                bandFile = yaml.load(f)

                '''reprocess the bandFile as dict and return dict'''
                dict = OrderedDict()
                dict['nqpoint'] = int(bandFile['path'][0]['nqpoint'])
                # dict['npath'] = int(bandFile['npath'])

                # dict['reciprocal_lattice'] = np.array(bandFile['reciprocal_lattice'])
                # dict['lattice'] = np.array(bandFile['lattice'])
                dict['phonon'] = np.array(bandFile['path'][0]['phonon'])
                dict['natom'] = int(len(dict['phonon'][0]['band']) / 3)

                return dict

        def _bandinfo(bandParameters):
            kPosition = np.zeros((bandParameters['nqpoint'], 3))
            kDistance = np.zeros(bandParameters['nqpoint'])
            eigenvectors = np.zeros((bandParameters['nqpoint'], bandParameters['natom'] * 3))
            gruneisen = np.zeros((bandParameters['nqpoint'], bandParameters['natom'] * 3))

            PhononBand = bandParameters['phonon']
            for x in range(bandParameters['nqpoint']):
                dict = PhononBand[x]
                kPosition[x, :] = np.array(dict['q-position'])
                kDistance[x] = float(dict['distance'])
                for y in range(bandParameters['natom'] * 3):
                    eigenvectors[x][y] = dict['band'][y]['frequency']
                    gruneisen[x][y] = dict['band'][y]['gruneisen']

            return kPosition, kDistance, eigenvectors, gruneisen
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        bandParameters = _ConvertYaml()
        kPosition, kDistance, eigenvectors, gruneisen = _bandinfo(bandParameters)
        return bandParameters, kPosition, kDistance, eigenvectors, gruneisen
        '''[^-^]Important things say three times!!![^-^]'''

    def plotGruneisen(self, HighlySymmetricPath):
        '''(..)===== Black Box,you can ignored me =====(..)'''
        def _getHighlySymmetricPath():
            list = []
            start = 0
            for value in HighlySymmetricPath.values():
                for i in range(start, len(kPosition)):
                    if (np.array(value) == kPosition[i]).all():
                        list.append(kDistance[i])
                        start = i + 1
                        break
            return list
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        bandParameters, kPosition, kDistance, eigenvectors, gruneisen = self.readGruneisenYaml()
        HighlySymmetricPath = OrderedDict(np.array(HighlySymmetricPath))
        distance_HighlySymmetricPath = _getHighlySymmetricPath()

        plt.figure(figsize=(8, 8))

        colors = plt.cm.jet(np.linspace(0, 1, bandParameters['natom'] * 3))
        for y in range(bandParameters['natom'] * 3):
            plt.plot(kDistance, gruneisen[:, y], color='Black', label='circ')

        # set x axis
        # plt.xlim(kDistance[0], kDistance[-1])
        for xHSP in distance_HighlySymmetricPath:
            plt.axvline(x=xHSP, color='k', linestyle='--',)
        plt.xticks(distance_HighlySymmetricPath, HighlySymmetricPath.keys())

        # set y axis
        plt.ylim(-200, 25)
        plt.ylabel('$\omega(THz)$', fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig("PhononyGruneisen.png", dpi=600)

        '''[^-^]Important things say three times!!![^-^]'''


class PhononyPDOS():
    def __init__(self, path=os.getcwd()):
        self.path = path

    def readMeshYaml(self):
        '''
        Extract the information in mesh.yaml

        you can get:
        MeshParameters;
        kPosition, kDistance, eigenvectors;
        '''

        '''(..)===== Black Box,you can ignored me =====(..)'''
        def _ConvertYaml():
            with open(self.path + '/mesh.yaml', 'r') as f:
                '''read the diction in Mesh.yaml as MeshFile '''
                MeshFile = OrderedDict()
                MeshFile = yaml.load(f)
                print(MeshFile.keys())

                '''reprocess the MeshFile as dict and return dict'''
                dict = OrderedDict()
                dict['nqpoint'] = int(MeshFile['nqpoint'])
                dict['npath'] = int(MeshFile['npath'])
                dict['natom'] = int(MeshFile['natom'])
                dict['points'] = MeshFile['points']
                dict['reciprocal_lattice'] = np.array(
                    MeshFile['reciprocal_lattice'])
                dict['lattice'] = np.array(MeshFile['lattice'])
                dict['phonon'] = np.array(MeshFile['phonon'])
                dict['mesh'] = np.array(MeshFile['mesh'])
                return dict

        def _Meshinfo(MeshParameters):
            kPosition = np.zeros((bandParameters['nqpoint'], 3))
            kDistance = np.zeros(bandParameters['nqpoint'])
            frequencies = np.zeros((MeshParameters['nqpoint'], MeshParameters['natom'] * 3))
            weights = np.zeros(MeshParameters['nqpoint'])
            eigenvectors = np.zeros(
                (MeshParameters['nqpoint'], MeshParameters['natom'] * 3))

            PhononPdos = MeshParameters['phonon']
            for x in range(MeshParameters['nqpoint']):
                dict = PhononPdos[x]
                kPosition[x, :] = np.array(dict['q-position'])
                kDistance[x] = float(dict['distance_from_gamma'])
                weights[x] = int(dict['weight'])
                for y in range(MeshParameters['natom'] * 3):
                    frequencies[x][y] = dict['band'][y]['frequency']

            return kPosition, kDistance, eigenvectors
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        MeshParameters = _ConvertYaml()
        kPosition, kDistance, eigenvectors = _Meshinfo(MeshParameters)
        # return MeshParameters, kPosition, kDistance, eigenvectors
        '''[^-^]Important things say three times!!![^-^]'''


if __name__ == "__main__":

    # 画出了SnO2TWIN的声子谱

    path = './'
    PhoB = PhononyBand(path)
    # PhoB.readGruneisenYaml()
    HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('Z', [0.0, 0.0, 0.5]), ('T', [-0.5, 0.0, 0.5]), ('Y', [-0.5, 0.0, 0.0]),
                           ('S', [-0.5, 0.5, 0.0]), ('X', [0.0, 0.5, 0.0]), ('U', [0.0, 0.5, 0.5]), ('R', [-0.5, 0.5, 0.5])]
    HighlySymmetricPath = [('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0, 0.0, 0.5]), ('H', [-0.333, 0.667, 0.5]), ('K', [-0.333, 0.667, 0.0]),
                           ('$\Gamma$', [0.0, 0.0, 0.0]), ('M', [0.0, 0.5, 0.0]), ('L', [0.0, 0.5, 0.5]), ('H ', [-0.333, 0.667, 0.5])]
    PhoB.plotGruneisen(HighlySymmetricPath)
    '''================================================='''
    # PhoP= PhononyPDOS('''./mesh.yaml''')
    # PhoP.readMeshYaml()


# 正则表达寻找文本关键字(与本脚本无关，只是Mark一下)
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
