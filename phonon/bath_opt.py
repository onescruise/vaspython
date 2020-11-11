import os
import shutil
'''
创建一个名叫OPT的文件夹，然后把优化好的CONTCAR改成POSCAR放进OPT中

path: OPT所在的路径

注意：最后会出现一个提交脚本submit_bath_pho.py，记得按照里面的提示
'''
#
path = input('# Path: ')

# path = '/media/one/My Passport/2020/C_Si_Ge/各项异性测试/Ge2_111_a/OPT'


class QHA():
    def __init__(self, path=os.getcwd()):
        '''
        Set basic parameters
        '''
        self.path = path
        self.number_minusFiles = 3
        self.number_plusFiles = 3
        self.delta_ScalingFactor = 0.0
        self.dalta_SF_A = 0.01
        self.dalta_SF_B = 0.0
        self.dalta_SF_C = 0.0

    def _lineardelta(self, flag, number, mark):

        folder = flag + str(number)

        if flag == 'minus':
            flag = -1
        else:
            flag = 1
        if mark == 'A':
            lineardelta = 1 + flag * (number * self.dalta_SF_A)
            line_number = 2
        if mark == 'B':
            lineardelta = 1 + flag * (number * self.dalta_SF_B)
            line_number = 3
        if mark == 'C':
            lineardelta = 1 + flag * (number * self.dalta_SF_C)
            line_number = 4

        f = open(folder + '/POSCAR')
        lines = f.readlines()
        list = lines[line_number].strip().split()
        string = '%20.10f' % (float(
            list[0]) * lineardelta) + '%21.10f' % (float(list[1]) * lineardelta) + '%21.10f' % (float(list[2]) * lineardelta)
        return string

    def getScalingFactor(self):
        f = open('POSCAR', 'r')
        lines = f.readlines()
        f.close()
        return float(lines[1].strip())

    def ReplacementLine(self, file, line_number, newstr):
        f = open(file, 'r')
        lines = f.readlines()
        f.close()
        lines[line_number] = str(newstr) + '\n'

        f = open(file, 'w')
        for line in lines:
            f.writelines(line)
        f.close()

    def Mkdir(self):

        os.chdir(self.path)

        self.ScalingFactor = self.getScalingFactor()

        for i in range(self.number_plusFiles):
            folder = 'plus' + str(i + 1)
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.copy2('POSCAR', folder)
            delta_SF = self.ScalingFactor + (1 + i) * self.delta_ScalingFactor
            self.ReplacementLine(folder + '/POSCAR', 1, delta_SF)
            self.ReplacementLine(folder + '/POSCAR', 1, delta_SF)
            self.ReplacementLine(folder + '/POSCAR', 2,
                                 self._lineardelta('plus', i + 1, 'A'))
            self.ReplacementLine(folder + '/POSCAR', 3,
                                 self._lineardelta('plus', i + 1, 'B'))
            self.ReplacementLine(folder + '/POSCAR', 4,
                                 self._lineardelta('plus', i + 1, 'C'))

        for i in range(self.number_minusFiles):
            folder = 'minus' + str(i + 1)
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.copy2('POSCAR', folder)
            delta_SF = self.ScalingFactor - (1 + i) * self.delta_ScalingFactor
            self.ReplacementLine(folder + '/POSCAR', 1, delta_SF)
            self.ReplacementLine(folder + '/POSCAR', 2,
                                 self._lineardelta('minus', i + 1, 'A'))
            self.ReplacementLine(folder + '/POSCAR', 3,
                                 self._lineardelta('minus', i + 1, 'B'))
            self.ReplacementLine(folder + '/POSCAR', 4,
                                 self._lineardelta('minus', i + 1, 'C'))


if __name__ == "__main__":
    shutil.copy2('submit_bath_opt.py', path)
    QHA = QHA(path)
    QHA.Mkdir()
