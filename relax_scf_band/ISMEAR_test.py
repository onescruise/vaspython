#!/usr/bin/python
# coding=utf-8
import time
import os
import shutil
from collections import OrderedDict

print('These files need in PATH: INCAR, POTCAR, POSCAR, vasp.pbs\n')
task = input('# Task: ')
note = input('# note: ')
host = str(input('# host： 1,2,3,ib'))
path = input('# PATH: ')

# ismear的相关参数
ismear_modes = dict()
ismear_modes['insu_or_semi'] = {'ISMEAR': 0, 'SIGMA': 0.05}
ismear_modes['metal'] = {'ISMEAR': 1, 'SIGMA': 0.2}
for key in ismear_modes.keys():
    print(key)


class Ismear_test():
    def __init__(self):
        # 预处理所需
        if path == '':
            self.path = os.getcwd()
        else:
            self.path = path
        # 要创建的测试文件夹子
        self.ismear_modes = OrderedDict(ismear_modes)
        self.path_test = self.path + os.sep + 'ismear_test'

    def __writeReadme(self):
        f = open('readme', 'a')
        f.write('[' + time.asctime(time.localtime(time.time())) + ']\n')
        f.write('* ' + note + '\n')
        f.close()

    def __modify_pbs(self):
        '''
        4.用于修改vaspwz.pbs的节点和任务名
        '''
        os.chdir(self.path_test)
        with open('vaspwz.pbs', 'r') as f:
            lines = f.readlines()
            lines[1] = '#PBS -N ' + task + '\n'
            lines[4] = '#PBS -q host' + host + ' #host2 host3 hostib host1\n'

            if host == '1' or host == '2':
                lines[2] = '#PBS -l nodes=1:ppn=20\n'
                lines[19] = 'ND=20 #host3 gaiwei36\n'

            if host == '3' or host == 'ib':
                lines[2] = '#PBS -l nodes=1:ppn=36\n'
                lines[19] = 'ND=36 #host3 gaiwei36\n'

        with open('vaspwz.pbs', 'w') as f:
            for line in lines:
                f.write(line)

    def __modify_INCAR(self, key):
        '''
        5.3 用于修改INCAR
        '''
        with open('INCAR', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if 'ISMEAR' in lines[i]:
                    lines[i] = 'ISMEAR = ' + str(self.ismear_modes[key]['ISMEAR']) + '\n'
                if 'SIGMA' in lines[i]:
                    lines[i] = 'SIGMA = ' + str(self.ismear_modes[key]['SIGMA']) + '\n'
                if 'SYSTEM' in lines[i]:
                    lines[i] = 'SYSTEM = ' + task + '\n'

        with open('INCAR', 'w') as f:
            for line in lines:
                f.write(line)

    def __create_test_folder(self):
        os.chdir(self.path_test)
        # 5.1创建文件夹,，同时修改INCAR
        folder = []
        for key in self.ismear_modes.keys():
            if not os.path.exists(key):
                os.mkdir(key)
            folder.append(key)

        # 5.2 把文件放进文件夹
        for key in folder:
            for file in ['INCAR', 'POSCAR', 'POTCAR', 'vaspwz.pbs']:
                shutil.copy2(file, key)
        # 5.3 修改所需的INCAR
        for key in folder:
            os.chdir(key)
            self.__modify_INCAR(key)
            os.chdir(self.path_test)

    def pretreatment(self):
        # 1.cd 到所在目录
        os.chdir(self.path)

        # 2.创建文件夹,保存其路径
        if not os.path.exists('ismear_test'):
            os.mkdir('ismear_test')

        # 3.把所需的文件复制到ismear_test文件夹
        for file in ['INCAR', 'POSCAR', 'POTCAR', 'vaspwz.pbs']:
            shutil.copy2(file, self.path_test)

        # 4.修改vasp的任务名以及节点
        self.__modify_pbs()

        # 5.创建子test文件夹，并配置好文件
        self.__create_test_folder()

        # 创建readme.
        self.__writeReadme()

    def submit(self):
        os.chdir(self.path_test)
        for key in self.ismear_modes.keys():
            os.chdir(key)
            os.system('qsub vaspwz.pbs')
            os.chdir(self.path_test)


if __name__ == "__main__":
    I = Ismear_test()
    I.pretreatment()
    I.submit()
