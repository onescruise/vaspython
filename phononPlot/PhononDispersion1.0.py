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
            with open(self.path+'/band.yaml', 'r') as f:
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

    def plotBand(self, HighlySymmetricPath):
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

        plt.figure(figsize=(8, 8))

        colors = plt.cm.jet(np.linspace(0, 1, bandParameters['natom'] * 3))
        for y in range(bandParameters['natom'] * 3):
            # plt.plot(kDistance, eigenvectors[:, y], color=colors[y])
            plt.plot(kDistance, eigenvectors[:, y], color='black')

        # set x axis
        plt.xlim(kDistance[0], kDistance[-1])
        for xHSP in distance_HighlySymmetricPath:
            plt.axvline(x=xHSP, color='k', linestyle='--',)
        plt.axhline(y=0,color='k', linestyle='-')
        plt.xticks(distance_HighlySymmetricPath, HighlySymmetricPath.keys())

        # set y axis
        plt.ylim(-2.5, 40)
        plt.ylabel('$\omega(THz)$', fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig(self.path+"/PhononyBand.png", dpi=600)

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
            with open(self.path+'/mesh.yaml', 'r') as f:
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
                dict['reciprocal_lattice']=np.array(
                    MeshFile['reciprocal_lattice'])
                dict['lattice']=np.array(MeshFile['lattice'])
                dict['phonon']=np.array(MeshFile['phonon'])
                dict['mesh']=np.array(MeshFile['mesh'])
                return dict

        def _Meshinfo(MeshParameters):
            kPosition = np.zeros((bandParameters['nqpoint'], 3))
            kDistance = np.zeros(bandParameters['nqpoint'])
            frequencies=np.zeros((MeshParameters['nqpoint'],MeshParameters['natom'] * 3))
            weights=np.zeros(MeshParameters['nqpoint'])
            eigenvectors=np.zeros(
                (MeshParameters['nqpoint'], MeshParameters['natom'] * 3))

            PhononPdos=MeshParameters['phonon']
            for x in range(MeshParameters['nqpoint']):
                dict=PhononPdos[x]
                kPosition[x, :]=np.array(dict['q-position'])
                kDistance[x]=float(dict['distance_from_gamma'])
                weights[x]=int(dict['weight'])
                for y in range(MeshParameters['natom'] * 3):
                    frequencies[x][y]=dict['band'][y]['frequency']

            return kPosition, kDistance, eigenvectors
        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        MeshParameters=_ConvertYaml()
        kPosition, kDistance, eigenvectors=_Meshinfo(MeshParameters)
        # return MeshParameters, kPosition, kDistance, eigenvectors
        '''[^-^]Important things say three times!!![^-^]'''


if __name__ == "__main__":
# 画出了SnO2TWIN的声子谱
    if False:
        for name in ['plus3','plus2','plus1','minus3','minus2','minus1','origin']:
            path='/media/ones/My Passport/workstation/SnO2TWIN544/PHO/'+ name
            PhoB=PhononyBand(path)
            # PhoB.readBandYaml()
            HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('Z', [0.0, 0.0, 0.5]), ('T', [-0.5, 0.0, 0.5]), ('Y', [-0.5, 0.0, 0.0]),
                                   ('S', [-0.5, 0.5, 0.0]), ('X', [0.0, 0.5, 0.0]), ('U', [0.0, 0.5, 0.5]), ('R', [-0.5, 0.5, 0.5])]
            # HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0000000000,0.0000000000,0.5000000000]), ('H', [-0.333333333, 0.6666666667, 0.5000000000]), ('K', [-0.333333333,0.6666666667,0.0000000000]),('$\Gamma$', [0.0, 0.0, 0.0]), ('M', [0.0000000000,0.5000000000,0.0000000000]), ('L', [0.0, 0.5, 0.5]), ('H', [-0.333333333,0.6666666667,0.5000000000])]
            PhoB.plotBand(HighlySymmetricPath)
        '''================================================='''

# 画金刚石的TWIN
    if True:
        path='/media/ones/My Passport/2020/one_layers_pho_pbe_1.01'
        PhoB=PhononyBand(path)
        HighlySymmetricPath=[('$\Gamma$', [0.0, 0.0, 0.0]), ('A', [0.0, 0.0, 0.5]), ('H', [-0.333, 0.667, 0.5]), ('K', [-0.333, 0.667, 0.0]), ('$\Gamma$ ', [0.0, 0.0, 0.0]), ('M', [0.0, 0.5, 0.0]), ('L', [0.0, 0.5, 0.5]), ('H ', [-0.333, 0.667, 0.5])]
        PhoB.plotBand(HighlySymmetricPath)
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
