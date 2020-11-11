import os
import matplotlib.pyplot as plt
import matplotlib
import collections
import numpy as np
import pandas as pd

#!/usr/bin/env python
# coding=utf-8
'''
Created on 14:35, Sep. 17, 2019

@author: Yilin Zhang

These files are needed: DOSCAR, POSCAR, OUTCAR
'''


matplotlib.use('Agg')


class PlotDOS():
    def __init__(self):
        '''
        '''
        self.parameter = {}
        self.Orbit = {}

    def _getParameterInDOSCAR(self, path=os.getcwd()):
        '''
        read the parameter in DOSCAR, such as E-max,E-min,NEDOS,E-fermi,the number of DOS.

        '''
        f = open(path + "/DOSCAR", 'r')
        lines = f.readlines()
        DOSCAR_line5 = str(lines[5]).strip().split()
        self.parameter['E-max'] = float(DOSCAR_line5[0])  # print(line)
        self.parameter['E-min'] = float(DOSCAR_line5[1])
        self.parameter['NEDOS'] = int(DOSCAR_line5[2])
        self.parameter['E-fermi'] = float(DOSCAR_line5[3])
        self.parameter['DOS_number'] = (
            len(lines) - 5) // (self.parameter['NEDOS'] + 1)
        f.close()

    def _getParameterInPOSCAR(self, path=os.getcwd()):
        '''
        read the information about the lattice in POSCAR, such as the CellVolume and elements.
        '''
        f = open(path + '/POSCAR', 'r')
        lines = f.readlines()

        a = np.array(str(lines[2]).strip().split()).astype(np.float)
        b = np.array(str(lines[3]).strip().split()).astype(np.float)
        c = np.array(str(lines[4]).strip().split()).astype(np.float)
        self.parameter['CellVolume'] = np.dot(
            np.cross(a, b), c)  # get the CellVolume

        self.elements = collections.OrderedDict()
        elements_key = str(lines[5]).strip().split()
        elements_value = str(lines[6]).strip().split()
        for i in range(len(elements_key)):
            self.elements[elements_key[i]] = int(elements_value[i])
        f.close()

    def _getParameterInOUTCAR(self, path=os.getcwd()):
        '''
        read the necessary parameter in OUTACAR, such as LORBIT, ISPIN and LNONCOLLINEAR.
        '''
        f = open(path + "/OUTCAR", 'r')
        lines = f.readlines()
        for line in lines:
            if "LORBIT" in line:
                self.parameter['LORBIT'] = int(line.strip().split()[2])
            if "ISPIN" in line:
                self.parameter['ISPIN'] = int(line.strip().split()[2])
            if "LNONCOLLINEAR" in line:
                self.parameter['LNONCOLLINEAR'] = line.strip().split()[2]
        f.close()

    def _preHandleDOSCAR(self, path=os.getcwd()):
        # read DOSCAR and pre-handle the DOSCAR
        self.everyDOS = {}
        dataDOS = pd.read_csv(path + "/DOSCAR")
        # print(dataDOS) #test
        for i in range(self.parameter['DOS_number']):
            add = i * (1 + self.parameter['NEDOS'])
            datasets = dataDOS.iloc[5 + add:5
                                    + add + self.parameter['NEDOS'], :]

            datasets.columns = ["DOS" + str(i)]
            datasets = np.array(datasets["DOS" + str(i)].str.strip())
            infoDOS = [datasets[line].split()
                       for line in range(datasets.shape[0])]
            for line in range(self.parameter['NEDOS']):
                for row in range(len(infoDOS[0])):
                    infoDOS[line][row] = np.float(infoDOS[line][row])
            infoDOS = np.array(infoDOS)
            infoDOS[:, 0] -= self.parameter['E-fermi']
            self.everyDOS[i] = infoDOS

    def _spinSeparation(self):

        self.UpDOS = {}
        self.DownDOS = {}

        for i in range(self.parameter['DOS_number']):
            self.UpDOS[i] = np.c_[self.everyDOS[i][:, 0],
                                  self.everyDOS[i][::, 1::2]]
            self.DownDOS[i] = self.everyDOS[i][::, ::2]

    def _getAtomObrit(self, everyDOS):
        # get the s,p,d obrits in each Atom
        AtomDOS = {}
        count = 1
        for key in self.elements.keys():
            for i in range(self.elements[key]):
                flag = AtomDOS[key + '_' + str(i + 1)] = everyDOS[count]
                count += 1
                if self.parameter['LORBIT'] == 10 | 11 | 12:
                    if self.parameter['LNONCOLLINEAR'] == 'F':
                        self.Orbit[key + '_' + str(i + 1) + '_s'] = flag[:, 1]
                        self.Orbit[key + '_'
                                   + str(i + 1) + '_p'] = flag[:, 2] + flag[:, 3] + flag[:, 4]
                        self.Orbit[key + '_' + str(i + 1) + '_d'] = flag[:, 5] + \
                            flag[:, 6] + flag[:, 7] + flag[:, 8] + flag[:, 9]
                    else:
                        self.Orbit[key + '_' + str(i + 1) + '_s'] = flag[:, 1]
                        self.Orbit[key + '_'
                                   + str(i + 1) + '_p'] = flag[:, 4] + flag[:, 8] + flag[:, 12]
                        self.Orbit[key + '_' + str(i + 1) + '_d'] = flag[:, 16] + \
                            flag[:, 20] + flag[:, 24] + \
                            flag[:, 28] + flag[:, 32]
                else:
                    if self.parameter['LNONCOLLINEAR'] == 'F':
                        self.Orbit[key + '_' + str(i + 1) + '_s'] = flag[:, 1]
                        self.Orbit[key + '_' + str(i + 1) + '_p'] = flag[:, 2]
                        self.Orbit[key + '_' + str(i + 1) + '_d'] = flag[:, 3]
                    else:
                        self.Orbit[key + '_' + str(i + 1) + '_s'] = flag[:, 1]
                        self.Orbit[key + '_' + str(i + 1) + '_p'] = flag[:, 4]
                        self.Orbit[key + '_' + str(i + 1) + '_d'] = flag[:, 7]

                self.Orbit[key + '_' + str(i + 1) + '_0'] = self.Orbit[key + '_' + str(i + 1) + '_s'] + \
                    self.Orbit[key + '_' + str(i + 1) + '_p'] + \
                    self.Orbit[key + '_' + str(i + 1) + '_d']

    def _getElementObrit(self, everyDOS):

        elementDOS = {}
        count = 1
        for key in self.elements.keys():
            elementDOS[key] = 0
            for i in range(self.elements[key]):
                elementDOS[key] += everyDOS[count]
                count += 1
            number = i + 1
            flag = elementDOS[key] / number

            if self.parameter['LORBIT'] == 10 | 11 | 12:
                if self.parameter['LNONCOLLINEAR'] == 'F':
                    self.Orbit[key + '_' + str(-1) + '_s'] = flag[:, 1]
                    self.Orbit[key + '_'
                               + str(-1) + '_p'] = flag[:, 2] + flag[:, 3] + flag[:, 4]
                    self.Orbit[key + '_' + str(-1) + '_d'] = flag[:, 5] + \
                        flag[:, 6] + flag[:, 7] + flag[:, 8] + flag[:, 9]
                else:
                    self.Orbit[key + '_' + str(-1) + '_s'] = flag[:, 1]
                    self.Orbit[key + '_'
                               + str(-1) + '_p'] = flag[:, 4] + flag[:, 8] + flag[:, 12]
                    self.Orbit[key + '_' + str(-1) + '_d'] = flag[:, 16] + \
                        flag[:, 20] + flag[:, 24] + \
                        flag[:, 28] + flag[:, 32]
            else:
                if self.parameter['LNONCOLLINEAR'] == 'F':
                    self.Orbit[key + '_' + str(-1) + '_s'] = flag[:, 1]
                    self.Orbit[key + '_' + str(-1) + '_p'] = flag[:, 2]
                    self.Orbit[key + '_' + str(-1) + '_d'] = flag[:, 3]
                else:
                    self.Orbit[key + '_' + str(-1) + '_s'] = flag[:, 1]
                    self.Orbit[key + '_' + str(-1) + '_p'] = flag[:, 4]
                    self.Orbit[key + '_' + str(-1) + '_d'] = flag[:, 7]

            self.Orbit[key + '_' + str(-1) + '_0'] = self.Orbit[key + '_' + str(-1) + '_s'] + \
                self.Orbit[key + '_' + str(-1) + '_p'] + \
                self.Orbit[key + '_' + str(-1) + '_d']

            self.Orbit[key + '_' + str(0) + '_s'] = self.Orbit[key +
                                                               '_' + str(-1) + '_s'] * number
            self.Orbit[key + '_' + str(0) + '_p'] = self.Orbit[key +
                                                               '_' + str(-1) + '_p'] * number
            self.Orbit[key + '_' + str(0) + '_d'] = self.Orbit[key +
                                                               '_' + str(-1) + '_d'] * number

            self.Orbit[key + '_' + str(0) + '_0'] = self.Orbit[key + '_' + str(0) + '_s'] + \
                self.Orbit[key + '_' + str(0) + '_p'] + \
                self.Orbit[key + '_' + str(0) + '_d']

    def _getTotalObrit(self, everyDOS):

        for orbit in ['s', 'p', 'd']:
            self.Orbit['Total_-1_' + orbit] = 0
            self.Orbit['Total_0_' + orbit] = 0
            for key in self.elements.keys():
                self.Orbit['Total_-1_'
                           + orbit] += self.Orbit[key + '_-1_' + orbit]
                self.Orbit['Total_0_'
                           + orbit] += self.Orbit[key + '_0_' + orbit]

        self.Orbit['Total_-1_0'] = everyDOS[0][:, 1] / \
            (self.parameter['DOS_number'] - 1)
        self.Orbit['Total_0_0'] = everyDOS[0][:, 1]

    def _getDefault(self, flag):
        if flag == 'path':
            return os.getcwd()

        if flag == 'erange':
            Emin = self.parameter['E-min'] - self.parameter['E-fermi']
            Emax = self.parameter['E-max'] - self.parameter['E-fermi']
            return [Emin, Emax]

    def _getLinesInfoDefault(self, linesInfo):
        # ('Total', [0, 0,'TotalDOS','b',18,'-'])
        # colors = plt.cm.jet(np.linspace(0, 1, 5))
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

        for i in range(len(linesInfo)):
            if len(linesInfo[i][1]) == 5:
                linesInfo[i][1].append(1)
            if len(linesInfo[i][1]) == 4:
                linesInfo[i][1].append('-')
                linesInfo[i][1].append(1)
            if len(linesInfo[i][1]) == 3:
                if linesInfo[i][0] == 'Total':
                    linesInfo[i][1].append('black')
                else:
                    linesInfo[i][1].append(colors[i % len(colors)])
                linesInfo[i][1].append('-')
                linesInfo[i][1].append(1)
        return linesInfo

    def _getLinesInfo(self, linesInfo):
        '''
        linesInformation
        '''

        # ('Total', [0, 0,'TotalDOS','b','-',1])
        linesPlot = collections.OrderedDict()
        i = 0
        for key in linesInfo.keys():
            linesPlot[i] = []
            linesPlot[i].append(key + '_'
                                + str(linesInfo[key][0]) + '_' + str(linesInfo[key][1]))
            linesPlot[i].append(linesInfo[key][2])
            linesPlot[i].append(linesInfo[key][3])
            linesPlot[i].append(linesInfo[key][4])
            linesPlot[i].append(linesInfo[key][5])
            i += 1

        return linesPlot

    def withoutSpin(self,  linesInfo, erange='default',  path='default'):
        '''
        plot the DOS without Spin

        Arguments:
            path: the path of DOSCAR, POSCAR, and OUTCAR
                [default]:[os.getcwd(),os.getcwd(),os.getcwd()]

            erange: the energy range of DOS
                [default]:[Emin-Efermin,Emax-Efermin,]

            linesInfo: the colors of the lines you want to plot.
                e.g.
                    [('Total', [0, 0,'TotalDOS','b','-',1]), (elememt1,
                      [1, 's','In_s','r']), (elememt2, [2, 'p','Se_p','g']),...]
                list[:,0]:
                    The object of this DOS drawing
                    'Total': the Total DOS of the whole system or a particular energy band
                    elememt1：the DOS of elememt1
                list[:,1]:
                    index 0:
                        -1 :is the mean of DOS, DOS/number of atoms or element atoms
                         0 : is the sum of DOS
                        1-N: the number of elememt
                    index 1:
                         0 : is the sum of orbits
                        s-d: the orbit
                    index 2:the label of lines
                    index 3:the color of lines
                        [default]:['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
                    index 4:the style of lines
                        [default]:'-'
                    index 5:the width of lines
                        [default]:1

        Return:
            None
        '''
        if path == 'default':
            path = self._getDefault('path')
        self._getParameterInDOSCAR(path)
        self._getParameterInPOSCAR(path)
        self._getParameterInOUTCAR(path)
        self._preHandleDOSCAR(path)
        # print(self.parameter)
        if self.parameter['ISPIN'] == 2:
            print('ISPIN==2, please try withSpin()\n')
            exit()

        everyDOS = self.everyDOS
        self._getAtomObrit(everyDOS)
        self._getElementObrit(everyDOS)
        self._getTotalObrit(everyDOS)

        if erange == 'default':
            erange = self._getDefault('erange')

        linesInfo = collections.OrderedDict(
            self._getLinesInfoDefault(linesInfo))
        # print(linesInfo)

        linesPlot = self._getLinesInfo(linesInfo)
        # print(linesOrbit)
        # print(linesColor)

        plt.figure(figsize=(8, 8))
        plt.xlim(erange[0], erange[-1])
        plt.tick_params(labelsize=18)
        plt.xlabel('$E-E_f(eV)$', fontsize=18)
        plt.ylabel('$DOS(states/eV)$', fontsize=18).set_fontweight('bold')
        Xenergy = self.everyDOS[0][:, 0]

        # print(self.Orbit[key])
        for key in linesPlot.keys():
            # ('Total', [0, 0,'TotalDOS','b','-',1])
            plt.plot(
                Xenergy, self.Orbit[linesPlot[key][0]], label=linesPlot[key][1], color=linesPlot[key][2], linestyle=linesPlot[key][3], linewidth=linesPlot[key][4])
        plt.legend()
        plt.tight_layout()
        plt.savefig('DOS.png', dpi=600)
        plt.clf()

    def withSpin(self, linesInfoUp, linesInfoDown, erange='default',  path='default'):
        '''
        plot the DOS with Spin

        Arguments:
            path: the path of DOSCAR, POSCAR, and OUTCAR
                [default]:[os.getcwd(),os.getcwd(),os.getcwd()]

            erange: the energy range of DOS
                [default]:[Emin-Efermin,Emax-Efermin,]

            linesInfo: the colors of the lines you want to plot.
                e.g.
                    [('Total', [0, 0,'TotalDOS(up/down)','b','-',1]), (elememt1,
                      [1, 's','In_s','r']), (elememt2, [2, 'p','Se_p','g']),...]
                list[:,0]:
                    The object of this DOS drawing
                    'Total': the Total DOS of the whole system or a particular energy band
                    elememt1：the DOS of elememt1
                list[:,1]:
                    index 0:
                        -1 :is the mean of DOS, DOS/number of atoms or element atoms
                         0 : is the sum of DOS
                        1-N: the number of elememt
                    index 1:
                         0 : is the sum of orbits
                        s-d: the orbit
                    index 2:the label of lines
                    index 3:the color of lines
                        [default]:['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
                    index 4:the style of lines
                        [default]:'-'
                    index 5:the width of lines
                        [default]:1

        Return:
            None
        '''

        if path == 'default':
            path = self._getDefault('path')

        self._getParameterInDOSCAR(path)
        self._getParameterInPOSCAR(path)
        self._getParameterInOUTCAR(path)
        self._preHandleDOSCAR(path)
        # print(self.parameter)
        if self.parameter['ISPIN'] == 1:
            print('ISPIN==1, please try withoutSpin()\n')
            exit()

        # separation the spin(up/down) DOS
        self._spinSeparation()

        # spin(UP)
        everyDOS = self.UpDOS
        self._getAtomObrit(everyDOS)
        self._getElementObrit(everyDOS)
        self._getTotalObrit(everyDOS)
        self.OrbitUp = self.Orbit

        linesInfoUp = collections.OrderedDict(
            self._getLinesInfoDefault(linesInfoUp))

        linesPlotUp = self._getLinesInfo(linesInfoUp)

        # spin(Down)
        everyDOS = self.DownDOS
        self._getAtomObrit(everyDOS)
        self._getElementObrit(everyDOS)
        self._getTotalObrit(everyDOS)
        self.OrbitDown = self.Orbit

        linesInfoDown = collections.OrderedDict(
            self._getLinesInfoDefault(linesInfoDown))

        linesPlotDown = self._getLinesInfo(linesInfoDown)

        plt.figure(figsize=(8, 8))
        if erange == 'default':
            erange = self._getDefault('erange')
        plt.xlim(erange[0], erange[-1])
        plt.tick_params(labelsize=18)
        plt.xlabel('$E-E_f(eV)$', fontsize=18)
        plt.ylabel('$DOS(states/eV)$', fontsize=18).set_fontweight('bold')
        Xenergy = self.everyDOS[0][:, 0]

        # print(self.Orbit[key])
        for key in linesPlotUp.keys():
            # ('Total', [0, 0,'TotalDOS','b','-',1])
            plt.plot(
                Xenergy, self.OrbitUp[linesPlotUp[key][0]], label=linesPlotUp[key][1] + '(up)', color=linesPlotUp[key][2], linestyle=linesPlotUp[key][3], linewidth=linesPlotUp[key][4])
        for key in linesPlotDown.keys():
            # ('Total', [0, 0,'TotalDOS','b','-',1])
            plt.plot(
                Xenergy, -self.OrbitDown[linesPlotDown[key][0]], label=linesPlotDown[key][1] + '(down)', color=linesPlotDown[key][2], linestyle=linesPlotDown[key][3], linewidth=linesPlotDown[key][4])
        plt.legend()
        plt.tight_layout()
        plt.savefig('DOS_Spin.png', dpi=600)
        plt.clf()


if __name__ == "__main__":

    p = PlotDOS()

    # withoutSpin()
    linesInfo = [('Total', [0, 0, 'TotalDOS', 'y']),
                 ('In', [1, 'd', 'In-d']), ('Se', [0, 's', 'Se-s', 'b', '-', 3])]
    erange = [-5, 5]
    p.withoutSpin(linesInfo, path='DOSxie')

    # withSpin()
    linesInfoUp = [('Total', [0, 0, 'TotalDOS', 'b']),
                   ('Na', [1, 'd', 'Na-d', 'g', ]), ('Cl', [0, 's', 'Cl-s', 'r', '-'])]
    linesInfoDown = [('Total', [0, 0, 'TotalDOS', 'y']),
                     ('Na', [1, 'd', 'Na-d', 'c', ]), ('Cl', [0, 's', 'Cl-s', 'b', '-'])]
    p.withSpin(linesInfoUp, linesInfoDown, path='DOSfu')

    # p.IntegratedDOS()
    # p.Atom()
    # p.Element()
