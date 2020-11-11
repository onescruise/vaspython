import os
import numpy as np
from bond import StructureBondLength
from structure import StructureMagic
from structure import StructureBasicInfo


class Point():
    '''所有与点相关的点信息

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

    def vertex_number_in_polyhedron(self, center_element=None, coor_number=None, distance=None, precision=1e-4):
        '''
        统计指定多面体间的顶点信息

        Args:
            center_element: 多面体的中心元素
            coor_number:配位数
            distance: [list]顶点到center_element的距离列表(距离请精确到小数点后4位🙏)
            precision: 对于distance的精度，默认为1e-4
        Returns:
            平均每个多面体对应的vertex_number

        '''
        # 打开可能需要写入信息的文件
        f = open(self.path + os.sep + 'vertex_number_in_polyhedron', 'w')

        # 构建超胞
        SM = StructureMagic(file=self.file, path=self.path)
        dim = [5, 5, 5]
        name_supercell = SM.build_supercell(dim=dim, direct=False)

        # 读取单/超胞基本结构信息
        SBI = StructureBasicInfo(file=name_supercell, path=self.path)
        center_atom_position = SBI.get_elements_position_matrix()[center_element]
        atoms_position = SBI.get_atoms_position_matrix()
        lattice_matrix = SBI.get_lattice_matrix() / np.array(dim)

        # 从atoms_position剔除center_atom_position
        # flag = []
        # for i in range(len(atoms_position)):
        #     for row in center_atom_position:
        #         if (atoms_position[i] == row).all():
        #             flag.append(i)
        # noncenter_atom_position = np.delete(atoms_position, flag, axis=0)

        # 消除边界效应
        atomA = center_atom_position
        atomB = atoms_position

        atomA = atomA[np.prod(atomA > np.tile(
            np.sum(lattice_matrix, axis=1) * 2, (len(atomA), 1)) - 0.01, axis=1).astype(bool)]
        atomA = atomA[np.prod(atomA < np.tile(
            np.sum(lattice_matrix, axis=1) * (np.array(dim) - 2) - 0.01, (len(atomA), 1)), axis=1).astype(bool)]

        # 统计center_atom与noncenter_atom原子数
        atomA_number = len(atomA)
        atomB_number = len(atomB)

        # 创建循环矩阵
        cycle_1_C_PositionMatrix = np.repeat(atomA, len(atomB), axis=0)
        cycle_2_C_PositionMatrix = np.tile(atomB, (len(atomA), 1))

        # 计算所有键长
        bond_length = (cycle_1_C_PositionMatrix -
                       cycle_2_C_PositionMatrix)**2
        bond_length = np.sqrt(np.sum(bond_length, axis=1))

        # 每个center原子的配位原子距离排序
        bond_length = np.sort(bond_length.reshape(atomA_number, atomB_number))[:, 1:coor_number + 1]
        print('vertex_number_in_polyhedron:')
        print(bond_length)
        f.write('vertex_number_in_polyhedron:\n')
        f.write(str(bond_length) + '\n')

        # 根据distance所对应的vertex_number
        distance = np.unique(np.round(np.array(distance), 4))
        bond = bond_length.flatten()
        print('precision: %f' % precision)
        print('all bond length to center element:')
        print(np.sort(np.unique(np.round(bond, 4))))
        f.write('precision: %f\n' % precision)
        f.write('all bond length to center element:\n')
        f.write(str(np.sort(np.unique(np.round(bond, 4)))) + '\n')
        count = 0
        for dis in distance:
            flag_bool = (np.abs(bond - dis) <= precision / 2)
            count += np.sum(flag_bool)
            bond = bond[~flag_bool]
        print('the polyhedron number: %d' % atomA_number)
        print('the sum vertex number: %d' % count)
        print('vertex number/polyhedron: %.4f' % (count / atomA_number))
        f.write('the polyhedron number: %d\n' % atomA_number)
        f.write('the sum vertex number: %d\n' % count)
        f.write('vertex number/polyhedron: %.4f\n' % (count / atomA_number))
        return count / atomA_number


class Line():
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


if __name__ == "__main__":
    '''test'''

    P = Point(file='POSCAR', path='/media/one/My Passport/2020/C_Si_Ge/C/PHO/orig')
    P.vertex_number_in_polyhedron(center_element='C', coor_number=4, distance=[1.5474])

    # P = Point(file='POSCAR', path='/media/one/My Passport/2020/SnO2/SnO2_301/PHO/orig')
    # P.vertex_number_in_polyhedron(center_element='Sn', coor_number=6, distance=[2.0654, 2.0807, 2.0879, 2.0899, 2.1006, 2.1027])
