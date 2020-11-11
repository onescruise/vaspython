import os
import numpy as np
from collections import OrderedDict
from ase.io.vasp import read_vasp
from ase.io.vasp import write_vasp


class StructureMagic():
    '''此类用于对于结构文件进行操作


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
        self.cell = read_vasp(self.path + os.sep + self.file)

    def build_supercell(self, dim=[6, 6, 6], direct=True):
        '''超胞构建

        Args:
            dim:阔胞的大小，如[6,6,6]
            direct:布尔值,判断输出是否为direct
        Returns:
            name_supercell:超胞结构文件名称
        '''
        # 命名输出超胞的文件名
        name_supercell = str(self.cell.symbols) + '_supercell_' + str(dim[0]) + str(dim[1]) + str(dim[2])
        # 写入超胞
        write_vasp(self.path + os.sep + name_supercell, self.cell * np.array(dim), label=name_supercell,
                   direct=direct, sort=True, vasp5=True)
        # 超胞结构
        self.supercell = self.cell * np.array(dim)
        return name_supercell


class StructureBasicInfo():
    '''此类用于读取结构文件POSCAR中的基本信息'''

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

        f = open(self.path + os.sep + self.file, 'r')
        self.lines = f.readlines()
        f.close()

    def get_system_name(self):
        '''获取POSCAR的system名称

        Returns:
            system_name:system名称
        '''
        system_name = str(self.lines[0].strip())
        return system_name

    def get_scaling_factor(self):
        '''获取POSCAR的缩放系数

        Returns:
            scaling_factor:缩放系数
        '''
        scaling_factor = np.array(
            str(self.lines[1]).strip().split()).astype(np.float)[0]
        return scaling_factor

    def get_lattice_matrix(self):
        '''获取POSCAR的晶格常数

        Returns:
            lattice_matrix:晶格常数
        '''
        a = np.array(str(self.lines[2]).strip().split()).astype(np.float)
        b = np.array(str(self.lines[3]).strip().split()).astype(np.float)
        c = np.array(str(self.lines[4]).strip().split()).astype(np.float)
        lattice_matrix = np.array([a, b, c])
        return lattice_matrix

    def get_atoms_info(self):
        '''获取POSCAR的原子信息

        Returns:
            atoms_info:[字典]原子信息，如Sn:2;O:4
        '''
        atoms_info = OrderedDict()
        atoms_keys = self.lines[5].strip().split()
        atoms_number = self.lines[6].strip().split()
        for i in range(len(atoms_keys)):
            atoms_info[atoms_keys[i]] = int(atoms_number[i])
        return atoms_info

    def get_coordinate_type(self):
        '''获取POSCAR的坐标轴信息
        Returns:
            coordinate_type:Direct or Cartesian
        '''
        coordinate_type = str(self.lines[7].strip())
        return coordinate_type

    def get_atoms_position_matrix(self):
        '''获取POSCAR的所有原子坐标

        Returns:
            atoms_position_matrix:所有原子坐标矩阵
        '''
        atoms_sum = self.get_atoms_sum()
        atoms_position_matrix = np.zeros((atoms_sum, 3))
        for i in range(atoms_sum):
            atoms_position_matrix[i] = np.array(
                str(self.lines[i + 8]).strip().split())[0:3].astype(np.float)
        return atoms_position_matrix

    def calAngleBetween2Vectors(self, vector0, vector1):
        '''
        获取两个矢量的夹角
        '''
        angle = np.arccos(np.dot(vector0, vector1) /
                          (np.linalg.norm(vector0) * np.linalg.norm(vector1)))
        return angle

    def calLatticeMatrix_Transformation_2D(self):
        LatticeMatrix = self.getLatticeMatrix()
        a = LatticeMatrix[0]
        b = LatticeMatrix[1]
        angle = self.calAngleBetween2Vectors(a, b)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        a = np.array([np.cos(angle / 2) * a_norm, np.sin(angle / 2) * a_norm, 0])
        b = np.array([np.cos(angle / 2) * b_norm, np.sin(angle / 2) * b_norm * (-1), 0])
        return a, b

    def get_atoms_sum(self):
        '''获取POSCAR的总原子数目

        Returns:
            atoms_sum:总原子数目
        '''
        atoms_info = self.get_atoms_info()
        atoms_sum = 0
        for value in atoms_info.values():
            atoms_sum += value
        return atoms_sum

    def get_volume(self):
        '''获取POSCAR的晶胞体积

        Returns:
            volume:晶胞体积
        '''
        sf = self.get_scaling_factor()
        a = np.array(str(self.lines[2]).strip().split()).astype(np.float) * sf
        b = np.array(str(self.lines[3]).strip().split()).astype(np.float) * sf
        c = np.array(str(self.lines[4]).strip().split()).astype(np.float) * sf
        volume = np.dot(np.cross(a, b), c)
        return volume

    def get_elements_position_matrix(self):
        '''获取POSCAR的同一元素的原子坐标

        Returns:
            elements_position_matrix:同一元素的原子坐标矩阵
        '''
        atoms_info = self.get_atoms_info()
        atoms_position_matrix = self.get_atoms_position_matrix()

        elements_position_matrix = OrderedDict()
        count = 0
        for key, value in atoms_info.items():
            elements_position_matrix[key] = np.zeros((value, 3))
            for i in range(value):
                elements_position_matrix[key][i] = atoms_position_matrix[i + count]
            count += value
        return elements_position_matrix
