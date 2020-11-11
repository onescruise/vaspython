"""
These files are need: DOSCAR, OUTCAR,POSCAR
@author: ylzhang
Last updated 11/12/2019
"""

import pandas as pd
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
import os
import re

matplotlib.use('Agg')


class PlotDOS():
    def __init__(self,path=os.getcwd()):
        # import os
        self.path = path
        self.parameter = {}

        # 从DOSACAR读入必要参数，诸如费米能等
        f = open(self.path + "/DOSCAR", 'r')
        lines = f.readlines()
        DOSCAR_line5 = str(lines[5]).strip().split()
        self.parameter['E-max'] = float(DOSCAR_line5[0])  # print(line)
        self.parameter['E-min'] = float(DOSCAR_line5[1])
        self.parameter['NEDOS'] = int(DOSCAR_line5[2])
        self.parameter['E-fermi'] = float(DOSCAR_line5[3])
        self.parameter['Atoms_number'] = (len(lines)
                                          - 5) // (self.parameter['NEDOS'] + 1)
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

    def _getDOS(self):
        everyDOS = {}
        dataDOS = pd.read_csv(self.path + "/DOSCAR")
        # print(dataDOS) #test
        for i in range(self.parameter['Atoms_number']):
            add = i * (1 + self.parameter['NEDOS'])
            datasets = dataDOS.iloc[5 + add:5 +
                                    add + self.parameter['NEDOS'], :]

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

    def _PreDOS(self):
        # Data preprocessing: text2data

        # Data preprocessing: text2data finished

        everyDOS = self._getDOS()
        # print(AtomDOS)
        elementsInfo = self._getElementsInfo()
        # print(elementsInfo)
        totalDOS = {}
        totalDOS['DOS'] = everyDOS[0]
        AtomDOS = {}
        elementDOS = {}
        flag = 1
        for key in elementsInfo.keys():
            elementDOS[key] = 0
            for i in range(elementsInfo[key]):
                AtomDOS[key + '_' + str(i + 1)] = everyDOS[flag]
                elementDOS[key] += everyDOS[flag]
                flag += 1
            # elementDOS[key][:, 0] /= (i + 1)
            elementDOS[key] /= (i + 1)
        # print(AtomDOS)
        # [print(key) for key in AtomDOS]
        # [print(key) for key in totalDOS]
        # print(totalDOS)
        return totalDOS, AtomDOS, elementDOS

    def Total(self):
        totalDOS, AtomDOS, elementDOS = self._PreDOS()
        data = totalDOS['DOS']
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
        plt.savefig(self.path+os.sep+'TotalDOS.png', dpi=600)
        plt.clf()

        pd_data = pd.DataFrame(data[:, 0:2], columns=[
                               '$E-E_f(eV)$', 'Total DOS'])
        pd_data.to_csv('Total_DOS.csv')

    def Atom(self):
        totalDOS, AtomDOS, elementDOS = self._PreDOS()
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
            plt.savefig(self.path+os.sep+key + ' DOS.png', dpi=600)
            plt.clf()

    def Element(self):
        totalDOS, AtomDOS, elementDOS = self._PreDOS()
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
            plt.savefig(self.path+os.sep+key + ' DOS.png', dpi=600)
            plt.clf()

    def ElementPDOS(self,inOnePNG=True):
        totalDOS, AtomDOS, elementDOS = self._PreDOS()
        for key in elementDOS.keys():
            if not inOnePNG:
                plt.tick_params(labelsize=18)
                plt.xlabel('$E-E_f(eV)$', fontsize=18)
                plt.ylabel('$' + key + '-DOS(states/eV)$', fontsize=18)
            else:
                plt.xlabel('$E-E_f(eV)$')
                plt.ylabel('$PDOS(states/eV)$')

            xdata = elementDOS[key][:, 0] - self.parameter['E-fermi']
            # lm-decomposed LORBIT == 11 | 12
            if (self.parameter['LORBIT'] == 11 or 12):
                Sorbit = elementDOS[key][:, 1]
                # print(Sorbit)
                Porbit_py = elementDOS[key][:, 2]
                Porbit_pz = elementDOS[key][:, 3]
                Porbit_px = elementDOS[key][:, 4]
                # print(Porbit)
                Dorbit_dxy = elementDOS[key][:, 5]
                Dorbit_dyz = elementDOS[key][:, 6]
                Dorbit_dz2r2 = elementDOS[key][:, 7]
                Dorbit_dxz = elementDOS[key][:, 8]
                Dorbit_dx2y2 = elementDOS[key][:, 9]
                plt.plot(xdata, Sorbit, label=key + '-s')
                plt.plot(xdata, Porbit_py, label=key + '-$p_y$')
                plt.plot(xdata, Porbit_pz, label=key + '-$p_z$')
                plt.plot(xdata, Porbit_px, label=key + '-$p_x$')
                plt.plot(xdata, Dorbit_dxy, label=key + '-$d_{xy}$')
                plt.plot(xdata, Dorbit_dyz, label=key + '-$d_{yz}$')
                plt.plot(xdata, Dorbit_dz2r2, label=key + '-$d_{z2-r2}$')
                plt.plot(xdata, Dorbit_dxz, label=key + '-$d_{xz}$')
                plt.plot(xdata, Dorbit_dx2y2, label=key + '-$d_{x2-y2}$')
            if not inOnePNG:
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.path+os.sep+key + ' PDOS.png', dpi=600)
                plt.clf()

        if inOnePNG:
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.path+os.sep+'PDOS.png')
            plt.clf()


if __name__ == "__main__":
    p = PlotDOS('/home/one/Downloads')
    p.Total()
    # p.Atom()
    p.Element()
    p.ElementPDOS(inOnePNG=False)
