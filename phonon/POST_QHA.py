'''
path: the PHO folder
在path中准备对应的mesh.conf文件
'''
import os
import tools
import shutil
import re
import numpy as np


plusNumber = 5
minusNumber = 5

path = '''/media/one/My Passport/2020/C_Si_Ge/各项异性测试/Ge2_111_a/PHO'''
#path = os.getcwd()

tmax = 1000


class POST_qha():
    def __init__(self, path):
        '''
        Set basic parameters
        '''
        self.path = path

        self.number_minusFiles = minusNumber
        self.number_plusFiles = plusNumber

        self.folder = []
        for i in range(self.number_minusFiles):
            self.folder.append('minus' + str(self.number_minusFiles - i))
        self.folder.append('orig')
        for i in range(self.number_plusFiles):
            self.folder.append('plus' + str(i + 1))
        # print(self.folder)

        os.chdir(path)
        if not os.path.exists('QHA'):
            os.mkdir('QHA')

        files = os.listdir()
        if 'mesh.conf' not in files:
            print('''Please vi mesh.conf''')
            exit()

    def getParameterOUTCAR(self, path='.'):
        '''
        get some parameters in OUTCAR
        '''
        parameter = {}
        f = open(path + "/OUTCAR", 'r')
        lines = f.readlines()
        for line in lines:
            if "LORBIT" in line:
                LORBIT = re.compile(
                    r"(?<=LORBIT =)\s*\d+\.?\d*").findall(line)
                LORBIT = list(map(int, LORBIT))
                if LORBIT != []:
                    parameter['LORBIT'] = LORBIT[0]  # print(line)
            if "ISPIN" in line:
                ISPIN = re.compile(
                    r"(?<=ISPIN  =)\s*\d+\.?\d*").findall(line)
                ISPIN = list(map(int, ISPIN))
                if ISPIN != []:
                    parameter['ISPIN'] = ISPIN[0]
            if "energy  without entropy=" in line:
                Enthalpy = re.compile(
                    r"(?<=energy  without entropy=)\s*\-\d+\.?\d*").findall(line)
                Enthalpy = list(map(float, Enthalpy))
                if Enthalpy != []:
                    parameter['Enthalpy'] = Enthalpy[0]
        f.close()
        return parameter

    def getVolumeCONTCAR(self, path='.'):
        """
        Get unit cell volume
        """
        f = open(path + '/CONTCAR', 'r')
        lines = f.readlines()
        sf = np.array(str(lines[1]).strip().split()).astype(np.float)
        a = np.array(str(lines[2]).strip().split()).astype(np.float) * sf
        b = np.array(str(lines[3]).strip().split()).astype(np.float) * sf
        c = np.array(str(lines[4]).strip().split()).astype(np.float) * sf
        Volume = np.dot(np.cross(a, b), c)
        return Volume

    def get_evdat(self):
        '''
        Write e-v.dat
        '''
        os.chdir(self.path)
        os.chdir('../OPT')
        evdat = []
        for folder in self.folder:
            os.chdir(folder)
            Enthalpy = self.getParameterOUTCAR()['Enthalpy']
            Volume = self.getVolumeCONTCAR()
            os.chdir('..')
            evdat.append('%20.10f' % float(Volume) + '%20.10f' %
                         float(Enthalpy) + '\n')

        os.chdir('../PHO/QHA')

        f = open('e-v.dat', 'w')
        for i in range(len(evdat)):
            f.write(evdat[i])
        f.close()
        os.chdir('..')

    def Batch_CSF(self):
        '''
        Prepare necessary files
        '''

        os.chdir(self.path)

        for folder in self.folder:
            os.chdir(folder)
            files = os.listdir()
            number_dispFile = 0
            for filename in files:
                if (filename.find('disp-') != -1):
                    number_dispFile += 1

            list = ' '
            for i in range(1, number_dispFile + 1):
                if(i < 10):
                    str_number_dispFile = 'disp-00' + str(i) + '/vasprun.xml'
                elif(number_dispFile < 100):
                    str_number_dispFile = 'disp-0' + str(i) + '/vasprun.xml'
                else:
                    str_number_dispFile = 'disp-' + str(i) + '/vasprun.xml'
                list += (str_number_dispFile + ' ')
            os.system('phonopy --tolerance=1e-3 -f' + list)
            os.chdir('..')

    def _getThermalFiles(self, filename, aftername):

        os.chdir(filename)
        shutil.copy2('../mesh.conf', './mesh.conf')
        os.system('phonopy --tolerance=1e-3 -t mesh.conf')
        files = os.listdir(os.getcwd())
        flag = False
        for filename in files:
            if (filename.find('thermal_properties.yaml') != -1):
                shutil.copy2(filename, '../QHA/' + filename + aftername)
                shutil.copy2('POSCAR', '../QHA/POSCAR' + aftername)
                flag = True

        if flag:
            tools.Readme(filename + ' -getThermalFile Successfully!\n')
        os.chdir('..')

    def Batch_getThermalFiles(self):
        '''
        Prepare necessary files
        '''

        files = os.listdir(os.getcwd())
        for filename in files:
            if (filename.find('minus') != -1):
                aftername = '--' + filename[5:]
                self._getThermalFiles(filename, aftername)

            if (filename.find('plus') != -1):
                aftername = '-' + filename[4:]
                self._getThermalFiles(filename, aftername)

            if (filename.find('orig') != -1):
                aftername = '-0'
                self._getThermalFiles(filename, aftername)

    def QHA_Submit(self, tmax):
        '''
        Arguments:
            tmax
        '''
        os.chdir(self.path)
        os.chdir('QHA')
        string = ''
        for i in range(minusNumber):
            string += 'thermal_properties.yaml--' + str(minusNumber - i) + ' '
        string += 'thermal_properties.yaml-0 '
        for i in range(1, plusNumber + 1):
            string += 'thermal_properties.yaml-' + str(i) + ' '
        os.system('phonopy-qha -p -s --tmax=' +
                  str(tmax) + ' e-v.dat ' + string)


if __name__ == "__main__":
    Pqha = POST_qha(path)
    Pqha.get_evdat()
    # Pqha.Batch_CSF()
    Pqha.Batch_getThermalFiles()
    Pqha.QHA_Submit(tmax)
