#!/usr/bin/python
# coding=utf-8
import time
import os
import shutil
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp.inputs import Poscar

print('These files need in PATH: INCAR, POTCAR, CONTCAR, vaspwz.pbs\n')
task = input('# Task: ')
note = input('# note: ')
host = str(input('# host： e.g. 1,2,3,ib: '))
path = input('# PATH: ')


class Band():
    def __init__(self):
        # 预处理所需
        if path == '':
            self.path = os.getcwd()
        else:
            self.path = path
        # 要创建的band文件夹
        self.path_band = self.path + os.sep + 'band'
        print(self.path_band)

    def __writeReadme(self):
        f = open('readme', 'a')
        f.write('[' + time.asctime(time.localtime(time.time())) + ']\n')
        f.write('* ' + note + '\n')
        f.close()

    def __modify_pbs(self):
        '''
        5.用于修改vaspwz.pbs的节点和任务名
        '''
        os.chdir(self.path_band)
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
        7. 用于修改INCAR
        '''
        flag_ISTART = False
        flag_ICHARG = False

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
                if 'KSPACING' in lines[i]:
                    lines[i] = '#' + lines[i]

                if 'ISTART' in lines[i]:
                    lines[i] = 'ISTART = 1\n'
                    flag_ISTART = True
                if 'ICHARG' in lines[i]:
                    lines[i] = 'ICHARG = 11\n'
                    flag_ICHARG = True

        with open('INCAR', 'w') as f:
            for line in lines:
                f.write(line)
                if 'SYSTEM' in line:
                    if not flag_ISTART:
                        f.write('ISTART = 1\n')
                    if not flag_ICHARG:
                        f.write('ICHARG = 11\n')

    def pretreatment(self):
        # 1.cd 到scf的目录
        os.chdir(self.path)

        # 2.创建文件夹,保存其路径
        if not os.path.exists('band'):
            os.mkdir('band')

        # 3.把所需的文件复制到band文件夹
        for file in ['INCAR', 'CONTCAR', 'POTCAR', 'vaspwz.pbs']:
            shutil.copy2(file, self.path_band)

        # 4.建立WAVCAR和CHARGECAR的软链接
        os.chdir(self.path_band)
        os.system('ln -s ../WAVECAR ./')
        os.system('ln -s ../CHGCAR ./')

        # 5.修改vasp的任务名以及节点
        os.chdir(self.path_band)
        self.__modify_pbs()

        # 6.把CONTACR复制为POSCAR
        os.chdir(self.path_band)
        shutil.copy2('CONTCAR', 'POSCAR')

        # 7.修改INCAR
        os.chdir(self.path_band)
        self.__modify_INCAR()

        # 8.产生KPOINTS
        h = HighSymmetryPath(self.path_band)
        h.highSymmetryPath()

        '''test:
        os.chdir(self.path_band)
        flag = input('generate KPOINTS:\n vaspkit[1]|pymatgen[2]: ')
        if int(flag) == 2:
            h = HighSymmetryPath(self.path_band)
            h.highSymmetryPath()
        if int(flag) == 1:
            os.system('vaspkit')
        '''

        # 创建readme.
        os.chdir(self.path_band)
        self.__writeReadme()

    def submit(self):
        os.chdir(self.path_band)
        os.system('qsub vaspwz.pbs')


class HighSymmetryPath(object):
    '''
    Created on Sep 7, 2016
    modifed on Jun 5, 2020 for python3

    @author: fu
    '''

    '''
    get the high symmetry path for a given structure

    Ags:
        path: path of POSCAR excluding self-POSCAR
    '''

    def __init__(self, path=None):
        if path == None:
            self.path = './'
        else:
            self.path = path
            if not(os.path.exists(self.path)):
                print('Warning! path is wrong.')
            elif not(os.path.isdir(self.path)):
                print("Warning! not a directory.")

    def read(self, filename='POSCAR'):
        poscar = Poscar.from_file(self.path + '/' + filename, True, True)
        return poscar.structure

    def highSymmetryPath(self, filename='POSCAR', points=40):
        '''
        output the high symmetry path of POSCAR to KPOINTS file

        Args:
            filename: POSCAR or CONTCAR
            points: number of kpoints among two high symmetry points
        '''

        structure = self.read(filename)
        newStructure = HighSymmKpath(structure)
        kpath = newStructure.kpath['path']
        kpoint = newStructure.kpath['kpoints']

        outfile = self.path + '/KPOINTS'

        if (os.path.exists(outfile)):
            os.system('rm %s' % outfile)
        with open(outfile, 'a') as out:
            out.write('K-Points\n')
            out.write(' %d\n' % points)
            out.write('Line-mode\n')
            out.write('reciprocal\n')

            for i in range(len(kpath)):  # number of path
                # high symmetry points of a path
                for k in range(len(kpath[i])):
                    if (i == 0 and k == 0) or (i == len(kpath) - 1 and k == len(kpath[i]) - 1):
                        out.write(' %10.6f \t%10.6f \t%10.6f # %s\n' % (
                            kpoint[kpath[i][k]][0], kpoint[kpath[i][k]][1], kpoint[kpath[i][k]][2], kpath[i][k]))
                    else:
                        for j in range(0, 2):
                            out.write(' %10.6f \t%10.6f \t%10.6f # %s\n' % (
                                kpoint[kpath[i][k]][0], kpoint[kpath[i][k]][1], kpoint[kpath[i][k]][2], kpath[i][k]))


if __name__ == "__main__":
    B = Band()
    B.pretreatment()
    B.submit()
