import os
import numpy as np
from structure import StructureMagic
from structure import StructureBasicInfo


class StructureBondLength():
    '''用于计算结构文件中与键长相关的信息

    Attributes:

    '''

    def __init__(self, file=None, path=None):
        '''初始化结构文件所在路径和结构文件名称'''
        if file is None:
            self.file = 'POSCAR'
        else:
            self.file = file

        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path

    def mean_between_two_elements(self, element0, element1, NN_atom_number=None, min_bond_length=None, max_bond_length=None):
        '''计算两个元素原子间的平均键长，默认计算第1近邻键长

        Args:
            element0:所求的主角元素
            element1:背景元素
            NN_atom_number: [int]所需求的第N邻数1， 默认为第1近邻 优先级低
            min_bond_length：指定的键长下限，默认为第1近邻键长-0.2 优先级高
            max_bond_length：指定的键长上限，默认为第2近邻键长     优先级高
        Returns:
            指定条件下的平均键长

        '''
        # 构建超胞
        SM = StructureMagic(file='POSCAR', path=self.path)
        f = open(self.path + os.sep + 'mean_between_two_elements', 'w')
        dim = [5, 5, 5]
        name_supercell = SM.build_supercell(dim=dim, direct=False)

        # 读取单/超胞基本结构信息
        SBI = StructureBasicInfo(file=name_supercell, path=self.path)
        elements_position_matrix = SBI.get_elements_position_matrix()
        atomA = np.array(elements_position_matrix[element0])
        atomB = np.array(elements_position_matrix[element1])
        lattice_matrix = SBI.get_lattice_matrix() / np.array(dim)

        # 消除边界效应
        atomA = atomA[np.prod(atomA > np.tile(
            np.sum(lattice_matrix, axis=1) * 2, (len(atomA), 1)) - 0.01, axis=1).astype(bool)]
        atomA = atomA[np.prod(atomA < np.tile(
            np.sum(lattice_matrix, axis=1) * (np.array(dim) - 2) - 0.01, (len(atomA), 1)), axis=1).astype(bool)]
        print(element0 + ' number:%d' % len(atomA))
        f.write(element0 + ' number:%d' % len(atomA) + '\n')

        # 创建循环矩阵
        cycle_1_C_PositionMatrix = np.repeat(
            atomA, len(atomB), axis=0)
        cycle_2_C_PositionMatrix = np.tile(atomB, (len(atomA), 1))

        # 计算所有键长
        bond_length = (cycle_1_C_PositionMatrix -
                       cycle_2_C_PositionMatrix)**2
        bond_length = np.sqrt(np.sum(bond_length, axis=1))

        # 获取近邻截断
        atoms_bond_length = np.unique(np.round(bond_length + 0.05, 1))

        # 输出所有键长情况
        print('the NNth bond length:')
        print(atoms_bond_length[:10])
        f.write('the NNth bond length:\n')
        f.write(str(atoms_bond_length[:10]) + '\n')

        # 根据指定参数进行键长筛选
        if min_bond_length is None:
            if NN_atom_number is None:
                min_bond_length = atoms_bond_length[1] - 0.2
            else:
                min_bond_length = atoms_bond_length[NN_atom_number - 1]
                while min_bond_length <= 0:
                    min_bond_length += 0.5

        if max_bond_length is None:
            if NN_atom_number is None:
                max_bond_length = atoms_bond_length[2]
            else:
                max_bond_length = atoms_bond_length[NN_atom_number]

        bond_length = bond_length[bond_length >= min_bond_length]
        bond_length = bond_length[bond_length <= max_bond_length]

        # 汇总平均键长信息
        mean_length = [str(NN_atom_number), min_bond_length,
                       max_bond_length, np.round(np.mean(bond_length), 5)]

        # 打印键长信息
        print('  NN atom number:%s' % mean_length[0] + '\n' + ' min bond length:%f' % mean_length[1] +
              '\n' + ' max bond length:%f' % mean_length[2] + '\n' + 'mean bond length:%f' % mean_length[3])
        f.write('  NN atom number:%s' % mean_length[0] + '\n' + ' min bond length:%f' % mean_length[1] +
                '\n' + ' max bond length:%f' % mean_length[2] + '\n' + 'mean bond length:%f' % mean_length[3])
        f.close()

        return np.mean(bond_length)


if __name__ == "__main__":
    '''test'''
    SBL = StructureBondLength(file='POSCAR', path='/media/one/My Passport/2020/SnO2/SnO2_10136/PHO/orig')
    SBL.mean_between_two_elements(element0='O', element1='O', NN_atom_number=None, min_bond_length=None, max_bond_length=3.4)
