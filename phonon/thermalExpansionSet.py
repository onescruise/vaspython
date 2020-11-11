import sys
import numpy as np
from collections import OrderedDict
import os
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
# matplotlib.use('Agg')
sys.path.append('/home/one/Nutstore Files/PPT/vaspython/Plot_plot')


class Gruneisen():
    def __init__(self):
        pass

    def readGruneisenYaml(self, path=os.getcwd(), file='gruneisen.yaml', type='phonon'):
        '''
        Extract the information in gruneisen.yaml

        you can get:
        GruneisenParameters;
        kPosition, kDistance, Gruneisen;
        '''
        '''(..)===== Black Box,you can ignored me =====(..)'''
        def _ConvertYaml():

            if type == 'path':
                dict = OrderedDict()
                with open(os.path.join(path, file), 'r') as f:
                    '''read the diction in band.yaml as gruneisenFile '''
                    gruneisenFile = OrderedDict()
                    gruneisenFile = yaml.load(f)
                    gruneisenFile = gruneisenFile['path']
                    '''reprocess the gruneisenFile as dict and return dict'''
                    dict['nqpoint'] = int(gruneisenFile[0]['nqpoint'])
                    dict['natom'] = int(
                        len(gruneisenFile[0]['phonon'][0]['band']) / 3)

                    dict['nPath'] = len(gruneisenFile)
                    dict['gruneisenFile'] = gruneisenFile

                    return dict

            if type == 'phonon':
                dict = OrderedDict()
                with open(os.path.join(path, file), 'r') as f:
                    '''read the diction in band.yaml as gruneisenFile '''
                    gruneisenFile = OrderedDict()
                    gruneisenFile = yaml.load(f)
                    dict['nqpoint'] = int(gruneisenFile['nqpoint'])
                    gruneisenFile = gruneisenFile['phonon']
                    '''reprocess the gruneisenFile as dict and return dict'''
                    dict['natom'] = int(
                        len(gruneisenFile[0]['band']) / 3)
                    dict['gruneisenFile'] = gruneisenFile
                    return dict

        def _gruneiseninfo(gruneisenParameters):
            natom = np.array((gruneisenParameters['natom']))
            nqpoint = gruneisenParameters['nqpoint']

            nPath = 1
            if type == 'path':
                nPath = gruneisenParameters['nPath']

            gruneisens = np.zeros((nqpoint * nPath, natom * 3))
            frequencies = np.zeros((nqpoint * nPath, natom * 3))
            for i in range(nPath):
                flag = gruneisenParameters['gruneisenFile']
                if type == 'path':
                    flag = flag[i]['phonon']
                for j in range(nqpoint):
                    for k in range(natom * 3):
                        if flag[j]['band'][k]['frequency'] < 0.05:
                            gruneisens[i * nqpoint + j][k] = None
                            frequencies[i * nqpoint + j][k] = None
                        else:
                            gruneisens[i * nqpoint + j][k] = flag[j]['band'][k]['gruneisen']
                            frequencies[i * nqpoint + j][k] = flag[j]['band'][k]['frequency']

            return frequencies, gruneisens

        '''(..) ========================= (..)'''

        '''[^-^]Look here! Look here! Look here![^-^]'''
        gruneisenParameters = _ConvertYaml()

        frequencies, gruneisens = _gruneiseninfo(gruneisenParameters)
        return frequencies, gruneisens
        '''[^-^]Important things say three times!!![^-^]'''

    def plot_GCvw(self, frequencies, gruneisens, Temprature=300, outpath=os.getcwd(), save=False, show=False):
        Temprature = 300
        kb = 1.380649 * 1e-23
        hbar = 1.05457266 * 1e-34
        const = hbar * 1e12 / (kb * Temprature)
        # mol = 6.02 * 1e23
        # V = 163.540152412100014

        # 数据处理
        frequencies = np.array(frequencies).flatten()
        gruneisens = np.array(gruneisens).flatten()
        const_f = const * frequencies
        exp_const_f = np.exp(const_f)
        Cvi = kb * np.power(const_f, 2) * exp_const_f / np.power((exp_const_f - 1), 2)

        # Cvi = mol * (0.5 + 1 / (exp_const_f - 1)) / 1e-10 * hbar * frequencies
        # print(np.sum(Cvi))

        # 按频率由大到小排列

        data = np.stack((frequencies, gruneisens, Cvi, gruneisens * Cvi / np.sum(Cvi)), axis=-1)
        data = data[data[:, 0].argsort()]
        name = ['Frequency_i', 'gruneisen_i', 'Cv_i', 'gi*Cvi/sum(Cvi)']

        bin_number = 100
        bin_step = (np.max(data[:, 0]) - np.min(data[:, 0])) / bin_number
        data0_min = np.min(data[:, 0])
        data0_max = np.max(data[:, 0])
        x = []
        y = []
        for i in range(bin_number):
            flag = np.where((data[:, 0] >= data0_min + i * bin_step) & (data[:, 0] <= data0_min + (i + 1) * bin_step))
            x.append(np.mean(data[:, 0][flag]))
            y.append(np.sum(data[:, 3][flag]))
        plt.plot(x, y)
        plt.show()

        # 给出草图
        # from threePicture import A_BCD_Scatter
        # plt = A_BCD_Scatter(name, data, show=False, save=save, outpath=outpath, filename='plot_GCvw.png')

        # 保存数据格式为csv类型
        import pandas as pd
        csv = pd.DataFrame(columns=name, data=data)
        csv.to_csv(os.path.join(outpath, 'plot_GCvw.csv'))


if __name__ == '__main__':
    G = Gruneisen()
    path = '/media/one/My Passport/2020/C_Si_Ge/Si111/two_bilayers/PHO'
    frequencies, gruneisens = G.readGruneisenYaml(path=path, file='gruneisen.yaml', type='phonon')
    G.plot_GCvw(frequencies, gruneisens, outpath=path, save=True)
