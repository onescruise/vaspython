#!/usr / bin / env python
# coding = utf - 8
"""
last update 12/14/2019
@ ylzhang
"""
import os


class stepOptimization():
    def __init__(self, path):
        self.path = path
        os.chdir(self.path)

    def writeINCAR(self, number):
        INCAR = {}
        if number == 1:
            f = open('INCAR', 'w')
            f.write('system = stepOPT1\n')
            f.write('NPAR = 4\n')
            f.write('ISTART = 0\n')
            f.write('PREC = LOW\n')
            f.write('IALGO = 48\n')
            f.write('LREAL = Auto\n')
            f.write('\n')

            f.write('NELM = 200\n')
            f.write('NELMIN = 5\n')
            f.write('EDIFF = 1e-3\n')
            f.write('ISMEAR = 0\n')
            f.write('SIGMA = 0.05\n')
            f.write('\n')

            f.write('NSW = 45\n')
            f.write('IBRION = 2\n')
            f.write('POTIM = 0.2\n')
            f.write('ISIF = 2\n')
            f.write('\n')

            f.write('KSPACING = 0.5\n')
            f.write('\n')

            f.write('LWAVE = .FALSE.\n')
            f.write('LCHARG = .FALSE.\n')
            f.write('LORBIT = 11\n')
            f.write('LMAXMIX = 4\n')
            f.write('\n')
            f.close()

    def getOtherFile(self, otherpath, needFile):
        for file in needFile:
            os.system('cp ' + otherpath + os.sep + file + ' ' + './')

    def renameFile(self, oldname, newname):
        os.system('cp ' + oldname + ' ' + newname)


if __name__ == '__main__':




#软膜找相变的分步优化-第一步
"""
    os.chdir('/media/one/My Passport/workstation/SnO2TWIN544/modulation1')
    if not os.path.exists('step_opt1'):
        os.mkdir('step_opt1')

    step1 = stepOptimization(os.getcwd()+'/step_opt1')
    # 第一次分布优化
    step1.writeINCAR(number=1)
    step1.getOtherFile(otherpath='../', needFile=['POTCAR', '*.pbs'])
    step1.getOtherFile(otherpath='../Modulation-13.0',needFile=['CONTCAR'])
    step1.renameFile(oldname='CONTCAR', newname='POSCAR')


    os.chdir('/media/one/My Passport/workstation/SnO2TWIN544/modulation2')
    if not os.path.exists('step_opt1'):
        os.mkdir('step_opt1')

    step1 = stepOptimization(os.getcwd()+'/step_opt1')
    # 第一次分布优化
    step1.writeINCAR(number=1)
    step1.getOtherFile(otherpath='../', needFile=['POTCAR', '*.pbs'])
    step1.getOtherFile(otherpath='../Modulation-15.0',needFile=['CONTCAR'])
    step1.renameFile(oldname='CONTCAR', newname='POSCAR')


    os.chdir('/media/one/My Passport/workstation/SnO2TWIN544/modulation3')
    if not os.path.exists('step_opt1'):
        os.mkdir('step_opt1')

    step1 = stepOptimization(os.getcwd()+'/step_opt1')
    # 第一次分布优化
    step1.writeINCAR(number=1)
    step1.getOtherFile(otherpath='../', needFile=['POTCAR', '*.pbs'])
    step1.getOtherFile(otherpath='../Modulation-18.0',needFile=['CONTCAR'])
    step1.renameFile(oldname='CONTCAR', newname='POSCAR')
"""


#不同角度twin结构的分步优化
"""
    for flag in ['0','5','10','15','20']:
        path = '/media/one/disk2/2019/December/1212/SnO2TWIN/A_Group/'+flag+'_Degree'
        if not os.path.exists(path + os.sep + 'step1_opt'):
            os.mkdir(path + os.sep + 'step1_opt')
        path=path+'/step1_opt'

        step1 = stepOptimization(path)
        # 第一次分布优化
        step1.writeINCAR(number=1)
        step1.getOtherFile(otherpath='../../', needFile=['POTCAR','vaspwz.pbs'])
        step1.getOtherFile(otherpath='../', needFile=['*.vasp'])
        step1.renameFile(oldname='*.vasp', newname='POSCAR')
"""
