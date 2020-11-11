#!/usr/bin/python
# coding=utf-8
import time
import os
import shutil

print('These files need in PATH: INCAR, POTCAR, CONTCAR, vasp.pbs\n')
task = input('# Task: ')
note = input('# note: ')
host = str(input('# host： e.g. 1,2,3,ib: '))
path = input('# PATH: ')


class SCF():
    def __init__(self):
        # 预处理所需
        if path == '':
            self.path = os.getcwd()
        else:
            self.path = path
        # 要创建的scf文件夹
        self.path_scf = self.path + os.sep + 'scf'

    def __writeReadme(self):
        f = open('readme', 'a')
        f.write('[' + time.asctime(time.localtime(time.time())) + ']\n')
        f.write('* ' + note + '\n')
        f.close()

    def __modify_pbs(self):
        '''
        4.用于修改vaspwz.pbs的节点和任务名
        '''
        os.chdir(self.path_scf)
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

    def __modify_INCAR(self):
        '''
        5. 用于修改INCAR
        '''
        with open('INCAR', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if 'SYSTEM' in lines[i]:
                    lines[i] = 'SYSTEM = ' + task + '\n'
                if 'NSW' in lines[i]:
                    lines[i] = 'NSW = 0\n'
                if 'IBRION' in lines[i]:
                    lines[i] = 'IBRION = -1\n'
                if 'LWAVE' in lines[i]:
                    lines[i] = 'LWAVE = .TRUE.\n'
                if 'LCHARG' in lines[i]:
                    lines[i] = 'LCHARG = .TRUE.\n'

        with open('INCAR', 'w') as f:
            for line in lines:
                f.write(line)

    def pretreatment(self):
        # 1.cd 到所在目录
        os.chdir(self.path)

        # 2.创建文件夹,保存其路径
        if not os.path.exists('scf'):
            os.mkdir('scf')

        # 3.把所需的文件复制到scf文件夹
        for file in ['INCAR', 'CONTCAR', 'POTCAR', 'vaspwz.pbs']:
            shutil.copy2(file, self.path_scf)

        # 4.修改vasp的任务名以及节点
        os.chdir(self.path_scf)
        self.__modify_pbs()

        # 5.把CONTACR复制为POSCAR
        shutil.copy2('CONTCAR', 'POSCAR')

        # 5.修改INCAR
        os.chdir(self.path_scf)
        self.__modify_INCAR()

        # 创建readme.
        os.chdir(self.path_scf)
        self.__writeReadme()

    def submit(self):
        os.chdir(self.path_scf)
        os.system('qsub vaspwz.pbs')


if __name__ == "__main__":
    S = SCF()
    S.pretreatment()
    S.submit()
