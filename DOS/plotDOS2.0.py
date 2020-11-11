"""These files are need: DOSCAR, OUTCAR,POSCAR"""

import pandas as pd
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
import os
import re

matplotlib.use('Agg')


class PlotDOS():
    def __init__(self):
        # import os
        self.path = os.getcwd()
        self.parameter = {}

        # 从DOSACAR读入必要参数，诸如费米能等
        f = open(self.path + "/DOSCAR", 'r')
        lines = f.readlines()
        DOSCAR_line5 = str(lines[5]).strip().split()
        self.parameter['E-max'] = float(DOSCAR_line5[0])  # print(line)
        self.parameter['E-min'] = float(DOSCAR_line5[1])
        self.parameter['NEDOS'] = int(DOSCAR_line5[2])
        self.parameter['E-fermi'] = float(DOSCAR_line5[3])
        self.parameter['Atoms_number'] = (len(lines) -
                                          5) // (self.parameter['NEDOS'] + 1)
        # parameter.update(self.getDOSCAR())
        # print(self.parameter)  # test
        f.close()

        # 从OUTACAR读入必要参数
        f = open(self.path + "/OUTCAR", 'r')
        lines = f.readlines()
        for line in lines:
            if "LORBIT" in line:
                LORBIT = re.compile(
                    r"(?<=LORBIT =     )\d+\.?\d*").findall(line)
                LORBIT = list(map(int, LORBIT))
                self.parameter['LORBIT'] = LORBIT[0]  # print(line)
            if "ISPIN" in line:
                ISPIN = re.compile(
                    r"(?<=ISPIN  =      )\d+\.?\d*").findall(line)
                ISPIN = list(map(int, ISPIN))
                self.parameter['ISPIN'] = ISPIN[0]  # print(line
        f.close()
        # print(self.parameter['ISPIN'])
        # test
        print(self.parameter)
    # def getElements

    def _getUnit_CellVolume(self):
        """
        Get unit cell volume
        """
        import numpy as np
        f = open('POSCAR', 'r')
        lines = f.readlines()
        a = np.array(str(lines[2]).strip().split()).astype(np.float)
        b = np.array(str(lines[3]).strip().split()).astype(np.float)
        c = np.array(str(lines[4]).strip().split()).astype(np.float)
        unitCellVolume = np.dot(np.cross(a, b), c)
        return unitCellVolume

    def _getDOS(self):
        everyDOS = {}
        dataDOS = pd.read_csv(self.path + "/DOSCAR")
        # print(dataDOS) #test
        for i in range(self.parameter['Atoms_number']):
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
            everyDOS[i] = infoDOS
            # print(everyDOS['infoDOS0']) # test
        return everyDOS

    def _getElementsInfo(self):
        elements = collections.OrderedDict()
        f = open(self.path + "/POSCAR")
        lines = f.readlines()
        elements_key = str(lines[5]).strip().split()
        elements_value = str(lines[6]).strip().split()
        for i in range(len(elements_key)):
            elements[elements_key[i]] = int(elements_value[i])
        f.close()
        return elements

    def Total(self):
        everyDOS = self._getDOS()
        elementsInfo = self._getElementsInfo()
        totalDOS = {}
        totalDOS['DOS'] = everyDOS[0]

        unitCellVolume = self._getUnit_CellVolume()
        data = totalDOS['DOS']
        data[:, 1] *= unitCellVolume

        plt.tick_params(labelsize=18)
        plt.xlabel('$E-E_f(eV)$', fontsize=18)
        plt.ylabel('$DOS(states/eV)$', fontsize=18)

        if self.parameter['ISPIN'] == 1:
            plt.plot(data[:, 0] - self.parameter['E-fermi'],
                     data[:, 1], label='Total DOS')
        else:
            plt.plot(data[:, 0] - self.parameter['E-fermi'],
                     data[:, 1], label='Total DOS(up)')
            plt.plot(data[:, 0] - self.parameter['E-fermi'],
                     data[:, 2], label='Total DOS(down)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('TotalDOS.png', dpi=600)
        plt.clf()

        pd_data = pd.DataFrame(data[:, 0:2], columns=[
                               '$E-E_f(eV)$', 'Total DOS'])
        pd_data.to_csv('Total_DOS.csv')

    def Atom(self):
        everyDOS = self._getDOS()
        elementsInfo = self._getElementsInfo()

        AtomDOS = {}
        flag = 1
        for key in elementsInfo.keys():
            for i in range(elementsInfo[key]):
                AtomDOS[key + '_' + str(i + 1)] = everyDOS[flag]
                flag += 1

        for key in AtomDOS.keys():
            plt.tick_params(labelsize=18)
            plt.xlabel('$E-E_f(eV)$', fontsize=18)
            plt.ylabel('$' + key + '-DOS(states/eV)$', fontsize=18)
            xdata = AtomDOS[key][:, 0] - self.parameter['E-fermi']
            # lm-decomposed LORBIT == 11 | 12
            if (self.parameter['LORBIT'] == 11 or 12):
                Sorbit = np.sum(AtomDOS[key][:, 1:2], 1)
                # print(Sorbit)
                Porbit = np.sum(AtomDOS[key][:, 2:5], 1)
                # print(Porbit)
                Dorbit = np.sum(AtomDOS[key][:, 5:], 1)
                plt.plot(xdata, Sorbit, label=key + '-s')
                plt.plot(xdata, Porbit, label=key + '-p')
                plt.plot(xdata, Dorbit, label=key + '-d')
            plt.legend()
            plt.tight_layout()
            plt.savefig(key + ' DOS.png', dpi=600)
            plt.clf()

    def Element(self):

        everyDOS = self._getDOS()
        elementsInfo = self._getElementsInfo()
        elementDOS = {}
        flag = 1
        for key in elementsInfo.keys():
            elementDOS[key] = 0
            for i in range(elementsInfo[key]):
                elementDOS[key] += everyDOS[flag]
                flag += 1
            # elementDOS[key][:, 0] /= (i + 1)
            elementDOS[key] /= (i + 1)

        for key in elementDOS.keys():
            plt.tick_params(labelsize=18)
            plt.xlabel('$E-E_f(eV)$', fontsize=18)
            plt.ylabel('$' + key + '-DOS(states/eV)$', fontsize=18)
            xdata = elementDOS[key][:, 0] - self.parameter['E-fermi']
            # lm-decomposed LORBIT == 11 | 12
            if (self.parameter['LORBIT'] == 11 or 12):
                Sorbit = np.sum(elementDOS[key][:, 1:2], 1)
                # print(Sorbit)
                Porbit = np.sum(elementDOS[key][:, 2:5], 1)
                # print(Porbit)
                Dorbit = np.sum(elementDOS[key][:, 5:], 1)
                plt.plot(xdata, Sorbit, label=key + '-s')
                plt.plot(xdata, Porbit, label=key + '-p')
                plt.plot(xdata, Dorbit, label=key + '-d')
            plt.legend()
            plt.tight_layout()
            plt.savefig(key + ' DOS.png', dpi=600)
            plt.clf()


if __name__ == "__main__":
    p = PlotDOS()
    p.Total()
    p.Atom()
    p.Element()
