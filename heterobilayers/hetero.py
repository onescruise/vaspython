import os
import numpy as np
from collections import OrderedDict

'''
必要的调用函数
'''


class read_POSCAR():
    def __init__(self, path='./POSCAR'):
        self.path = path
        self.parameters = OrderedDict()

        f = open(self.path, 'r')
        self.lines = f.readlines()
        f.close()

    def addParameters(self, key, value):
        self.parameters[key] = value
        if 'Matrix' in key:
            print(str(key) + ':\n' + str(self.parameters[key]))
        else:
            print(str(key) + ':' + str(self.parameters[key]))

    def getSystemName(self):
        SystemName = str(self.lines[0].strip())
        self.addParameters(key='SystemName', value=SystemName)
        return SystemName

    def getScalingFactor(self):
        ScalingFactor = np.array(
            str(self.lines[1]).strip().split()).astype(np.float)[0]
        self.addParameters(key='ScalingFactor', value=ScalingFactor)
        return ScalingFactor

    def getLatticeMatrix(self):
        a = np.array(str(self.lines[2]).strip().split()).astype(np.float)
        b = np.array(str(self.lines[3]).strip().split()).astype(np.float)
        c = np.array(str(self.lines[4]).strip().split()).astype(np.float)
        LatticeMatrix = np.array([a, b, c])
        self.addParameters(key='LatticeMatrix', value=LatticeMatrix)
        return LatticeMatrix

    def getAtomsInfo(self):
        AtomsInfo = OrderedDict()
        AtomsKeys = self.lines[5].strip().split()
        AtomsNumber = self.lines[6].strip().split()
        for i in range(len(AtomsKeys)):
            AtomsInfo[AtomsKeys[i]] = int(AtomsNumber[i])
        self.addParameters(key='AtomsInfo', value=AtomsInfo)
        return AtomsInfo

    def getCoordinateType(self):
        CoordinateType = str(self.lines[7].strip())
        self.addParameters(key='CoordinateType', value=CoordinateType)
        return CoordinateType

    def getAtomsPositionMatrix(self):
        AtomsSum = self.calAtomsSum()
        AtomsPositionMatrix = np.zeros((AtomsSum, 3))
        for i in range(AtomsSum):
            AtomsPositionMatrix[i] = np.array(
                str(self.lines[i + 8]).strip().split())[0:3].astype(np.float)
        self.addParameters(key='AtomsPositionMatrix',
                           value=AtomsPositionMatrix)
        return AtomsPositionMatrix

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

    def calAtomsSum(self):
        AtomsInfo = self.getAtomsInfo()
        AtomsSum = 0
        for value in AtomsInfo.values():
            AtomsSum += value
        self.addParameters(key='AtomsSum', value=AtomsSum)
        return AtomsSum

    def calVolume(self):
        """
        Get unit cell volume
        """
        sf = self.getScalingFactor()
        a = np.array(str(self.lines[2]).strip().split()).astype(np.float) * sf
        b = np.array(str(self.lines[3]).strip().split()).astype(np.float) * sf
        c = np.array(str(self.lines[4]).strip().split()).astype(np.float) * sf
        Volume = np.dot(np.cross(a, b), c)
        self.addParameters(key='Volume', value=Volume)
        return Volume

    def calElementsPositionMatrix(self):
        AtomsInfo = self.getAtomsInfo()
        AtomsPositionMatrix = self.getAtomsPositionMatrix()

        ElementsPositionMatrix = OrderedDict()
        count = 0
        for key, value in AtomsInfo.items():
            ElementsPositionMatrix[key] = np.zeros((value, 3))
            for i in range(value):
                ElementsPositionMatrix[key][i] = AtomsPositionMatrix[i + count]
            count += value
        self.addParameters(key='ElementsPositionMatrix',
                           value=ElementsPositionMatrix)
        return ElementsPositionMatrix

    def Direct2Cartesian(self):
        return np.dot(self.getAtomsPositionMatrix()[:, :], self.getLatticeMatrix())


def writePOSCAR():
    def float2line(line):
        string = '%20.10f' % float(
            line[0]) + '%21.10f' % float(line[1]) + '%21.10f' % float(line[2]) + '\n'
        line = str(string)
        return line

    with open(parameters['SystemName'] + '_hetero_POSCAR', 'w') as f:
        f.write(parameters['SystemName'] + '\n')
        f.write(str(parameters['ScalingFactor']) + '\n')
        LatticeMatrix = parameters['LatticeMatrix']
        for i in range(len(LatticeMatrix)):
            f.write(float2line(LatticeMatrix[i]))

        AtomsInfo = parameters['AtomsInfo']
        string = ['', '']
        for key in AtomsInfo.keys():
            string[0] += key + '    '
            string[1] += str(AtomsInfo[key]) + '    '
        string[0] += '\n'
        string[1] += '\n'
        f.write(string[0])
        f.write(string[1])

        f.write(parameters['CoordinateType'] + '\n')

        ElementsPositionMatrix = parameters['ElementsPositionMatrix'].copy(
        )
        for key in ElementsPositionMatrix.keys():
            arr = ElementsPositionMatrix[key]
            for i in range(len(arr)):
                f.write(float2line(arr[i]))


# -----------------------------------------------------------------------------
'''
二维材料的结构信息获取(本部分用了vaspkit提供的异质结组合脚本)代码有点凌乱，小修了其中的一部分，后续还需要再修改
'''

# 3.分析每一层的结构信息，包括堆叠层数以及每层说对应的原子数


def dividing_layer(atos_ele_C_sort_A_or_B):
    z_coord = [0 for m in range(len(atos_ele_C_sort_A_or_B))]
    for i in range(len(atos_ele_C_sort_A_or_B)):
        z_coord[i] = atos_ele_C_sort_A_or_B[i][2]
    layer_coors = sorted(set(z_coord), key=z_coord.index)
    k = 0
    layer = [[] for o1 in range(len(layer_coors))]

    for layer_coor in layer_coors:
        for i in range(len(atos_ele_C_sort_A_or_B)):
            if abs(atos_ele_C_sort_A_or_B[i][2] - layer_coor) < 0.01:
                layer[k].append(atos_ele_C_sort_A_or_B[i][:])
        k += 1

    return layer

# 4.分别计算layer0与layer1中inter层间原子的最短键长


def cal_dis(coor1, coor2, q, w):
    distance = ((coor1[q][0] - coor2[w][0]) ** 2 + (coor1[q][1] - coor2[w][1])
                ** 2 + (coor1[q][2] - coor2[w][2]) ** 2) ** 0.5
    return distance


def cal_bond_length_1(atom_ele_C_A_or_B, atom_num_A_B0):
    atom_dis1 = []
    for i in range(atom_num_A_B0 - 1):
        atom_dis1.append(cal_dis(atom_ele_C_A_or_B, atom_ele_C_A_or_B, 0, i + 1))
        bond_length1 = min(atom_dis1)
    return bond_length1


def cal_bond_length_2(layer_A_or_B):
    atom_dis = []
    for m in range(len(layer_A_or_B[0])):
        for n in range(len(layer_A_or_B[1])):
            atom_dis.append(cal_dis(layer_A_or_B[0], layer_A_or_B[1], m, n))
    bond_length = min(atom_dis)
    return bond_length

# 5.计算layer0间层的间隔距离


def cal_dividing_layer_space(layer_A_or_B, bond_length_A_or_B):
    std_atom_infor = [0 for i in range(len(layer_A_or_B) - 1)]
    layer_std = [0 for i in range(len(layer_A_or_B) - 1)]
    for i in range(len(layer_A_or_B) - 1):
        for m in range(len(layer_A_or_B[i])):
            for n in range(len(layer_A_or_B[i + 1])):
                space = cal_dis(layer_A_or_B[i], layer_A_or_B[i + 1], m, n)
                if abs(space - bond_length_A_or_B) < 0.001:
                    layer_std[i] = layer_A_or_B[i][m]
                    std_atom_infor[i] = layer_A_or_B[i + 1][n]
    return layer_std, std_atom_infor

# 6.选择异质结的晶格矢量


def direct_to_cartesian(atos_D_A_or_B, lattice1, lattice2):
    atom1_latt2 = [[] for m in range(len(atos_D_A_or_B))]
    for i in range(len(atos_D_A_or_B)):
        atom1_latt2[i].append(atos_D_A_or_B[i][0] * lattice1[0][0] + atos_D_A_or_B[i]
                              [1] * lattice1[1][0] + atos_D_A_or_B[i][2] * lattice2[2][0])
        atom1_latt2[i].append(atos_D_A_or_B[i][0] * lattice1[0][1] + atos_D_A_or_B[i]
                              [1] * lattice1[1][1] + atos_D_A_or_B[i][2] * lattice2[2][1])
        atom1_latt2[i].append(atos_D_A_or_B[i][0] * lattice1[0][2] + atos_D_A_or_B[i]
                              [1] * lattice1[1][2] + atos_D_A_or_B[i][2] * lattice2[2][2])
        # atom1_latt2[i].append(atos_D_A_or_B[i][3])
    return atom1_latt2


def cartesian_to_direct(atomB_or_A_he_C, lattice1, lattice2):
    transit = [[] for i in range(len(atomB_or_A_he_C))]
    for i in range(len(atomB_or_A_he_C)):
        transit[i].append(atomB_or_A_he_C[i][0])
        transit[i].append(atomB_or_A_he_C[i][1])
        transit[i].append(atomB_or_A_he_C[i][2])
#    atom_C = np.transpose(transit)
    lattice = [[] for i in range(len(lattice1))]
    for i in range(len(lattice1) - 1):
        lattice[i].append(lattice1[i][0])
        lattice[i].append(lattice1[i][1])
        lattice[i].append(lattice1[i][2])
    lattice[2].append(lattice2[2][0])
    lattice[2].append(lattice2[2][1])
    lattice[2].append(lattice2[2][2])
    lattice_tran_mat = np.mat(np.transpose(lattice))
    convmat = np.linalg.inv(lattice_tran_mat)
    latt_inv = convmat.tolist()
    atomB_or_A_he_D = [[] for m in range(len(atomB_or_A_he_C))]
    for i in range(len(transit)):
        atomB_or_A_he_D[i].append(latt_inv[0][0] * transit[i][0] + latt_inv[0]
                                  [1] * transit[i][1] + latt_inv[0][2] * transit[i][2])
        atomB_or_A_he_D[i].append(latt_inv[1][0] * transit[i][0] + latt_inv[1]
                                  [1] * transit[i][1] + latt_inv[1][2] * transit[i][2])
        atomB_or_A_he_D[i].append(latt_inv[2][0] * transit[i][0] + latt_inv[2]
                                  [1] * transit[i][1] + latt_inv[2][2] * transit[i][2])
        # atomB_or_A_he_D[i].append(atomB_or_A_he_C[i][3])
    return atomB_or_A_he_D


def tran_of_each_layer(layer_std_A_or_B, atom_lattAB, std_atom_infor_A_or_B, bond_length_A_or_B):
    for i in range(len(layer_std_A_or_B)):
        for m in range(len(atom_lattAB)):
            for n in range(len(atom_lattAB)):
                if (np.array(layer_std_A_or_B[i]) == np.array(atom_lattAB[m])).all():
                    if (np.array(std_atom_infor_A_or_B[i]) == np.array(atom_lattAB[n])).all():
                        atom_tran = atom_lattAB[m][2] - (bond_length_A_or_B **
                                                         2 - (atom_lattAB[m][0] - atom_lattAB[n][0]) ** 2 - (atom_lattAB[m][1] -
                                                                                                             atom_lattAB[n][1]) ** 2) ** 0.5
                        tt = atom_lattAB[n][2]
                        for o in range(len(atom_lattAB)):
                            if atom_lattAB[o][2] == tt:
                                atom_lattAB[o][2] = atom_tran
    return atom_lattAB


def building_heterostructure(lattice_he, name_t, ele_A, ele_B, atom_num_A, atom_num_B, atom_A_new, atom_B_new):
    title = "Heterostructure generated by building_heterostructure"
    scaling = "1.0"
    fp4 = open(name_t, mode='w')
    float_width = 25
    str_width = 5
    fp4.write(title)
    fp4.write("\n")
    fp4.write(scaling)
    fp4.write("\n")
    fp4.write(str(lattice_he[0, 0]).rjust(float_width, ' '))
    fp4.write(str(lattice_he[0, 1]).rjust(float_width, ' '))
    fp4.write(str(lattice_he[0, 2]).rjust(float_width, ' '))
    fp4.write("\n")
    fp4.write(str(lattice_he[1, 0]).rjust(float_width, ' '))
    fp4.write(str(lattice_he[1, 1]).rjust(float_width, ' '))
    fp4.write(str(lattice_he[1, 2]).rjust(float_width, ' '))
    fp4.write("\n")
    fp4.write(str(lattice_he[2, 0]).rjust(float_width, ' '))
    fp4.write(str(lattice_he[2, 1]).rjust(float_width, ' '))
    fp4.write(str(lattice_he[2, 2]).rjust(float_width, ' '))
    fp4.write("\n")
    for m in range(len(ele_A)):
        fp4.write(str(ele_A[m]).rjust(str_width, ' '))
    for n in range(len(ele_B)):
        fp4.write(str(ele_B[n]).rjust(str_width, ' '))
    fp4.write("\n")
    for i in range(len(atom_num_A)):
        fp4.write(str(atom_num_A[i]).rjust(str_width, ' '))
    for i in range(len(atom_num_B)):
        fp4.write(str(atom_num_B[i]).rjust(str_width, ' '))
    fp4.write("\n")
    fp4.write("Direct")
    fp4.write("\n")
    atom_A_new1 = atom_A_new.tolist()
    atom_B_new1 = atom_B_new.tolist()
    for i in range(len(atom_A_new1)):
        fp4.write(str(atom_A_new1[i][0]).rjust(float_width, ' '))
        fp4.write(str(atom_A_new1[i][1]).rjust(float_width, ' '))
        fp4.write(str(atom_A_new1[i][2]).rjust(float_width, ' '))
        fp4.write("\n")
    for i in range(len(atom_B_new1)):
        fp4.write(str(atom_B_new1[i][0]).rjust(float_width, ' '))
        fp4.write(str(atom_B_new1[i][1]).rjust(float_width, ' '))
        fp4.write(str(atom_B_new1[i][2]).rjust(float_width, ' '))
        fp4.write("\n")
    fp4.close()
# --------------------------------------------------------------------------------


class establish_hetero():
    def __init__(self, path=None):
        if path == None:
            self.path = os.getcwd()
        else:
            self.path = path
        os.chdir(self.path)

    # 必要的调用函数
    def _create_cycle_matrix(self, cycle_list, cycle_number):
        '''
        创建循环矩阵

        '''
        cycle_list = np.array(cycle_list)
        cycle_length = len(cycle_list)
        cycle = np.zeros((cycle_length**cycle_number, cycle_number))

        for i in range(cycle_number):
            flag_cycle = np.repeat(cycle_list, cycle_length**(cycle_number - 1 - i), axis=0)
            for j in range(i):
                flag_cycle = np.tile(flag_cycle, (cycle_length, 1))
            cycle[:, i] = flag_cycle.reshape((1, -1))
        return cycle

    # 必要的调用函数

    def get_cell(self, inputfile, outputfile):
        print('get cell from ' + inputfile + ' ')
        os.system('cp ' + inputfile + ' POSCAR')
        os.system('echo 602 | vaspkit | grep 123')
        os.system('mv PRIMCELL.vasp ' + outputfile)
        os.system('rm POSCAR')
        print('Finished!')

    def get_lattice_dim(self, a, b, max_dim=5):
        '''
        https://blog.csdn.net/lxlong89940101/article/details/84314703
        https://blog.csdn.net/zyl1042635242/article/details/43052403
        '''

        # 1.1.给出一个单循环的一维数组
        dim_mn = np.arange(-max_dim, max_dim + 1)

        # 1.2.用numpy创建多循环矩阵
        cycle = self._create_cycle_matrix(cycle_list=dim_mn, cycle_number=4)

        # 1.3.构建扩包的晶格矢量
        vector_A = np.tile(cycle[:, 0], (2, 1)) * a[0: 2].reshape(-1, 1) + \
            np.tile(cycle[:, 1], (2, 1)) * b[0: 2].reshape(-1, 1)
        vector_B = np.tile(cycle[:, 2], (2, 1)) * a[0: 2].reshape(-1, 1) + \
            np.tile(cycle[:, 3], (2, 1)) * b[0: 2].reshape(-1, 1)

        # 1.4.求所扩包晶格的模
        vector_A_norm = np.sqrt(np.sum(vector_A ** 2, axis=0))  # 求所有情况下的A0的绝对值长度
        vector_B_norm = np.sqrt(np.sum(vector_B ** 2, axis=0))  # 求所有情况下的B0的绝对值长度

        # 1.5.删除不符合条件的晶格矢量，即不能矢量模长为0
        flag_logical = (vector_A_norm * vector_B_norm != 0)  # 排除某一边等于0的情况

        # 1.6.重新修正符合条件的晶格矢量与模长矩阵(把某一边等于0的情况删除之后的A0绝对值长度数组)
        vector_A = vector_A[:, flag_logical]  # 每行都用逻辑判断筛选
        vector_B = vector_B[:, flag_logical]
        vector_A_norm = vector_A_norm[flag_logical]
        vector_B_norm = vector_B_norm[flag_logical]
        cycle = cycle[flag_logical, :]

        # 1.7.计算对应的晶格矢量夹角,并进一步进行条件筛选
        angle_between_vectorAB = np.sum(vector_A * vector_B, axis=0) / \
            (vector_A_norm * vector_B_norm)
        flag_angle = (np.abs(angle_between_vectorAB) < 1)

        angle_between_vectorAB = angle_between_vectorAB[flag_angle]
        # vector_A = vector_A[:, flag_angle]  # 每行都用逻辑判断筛选
        # vector_B = vector_B[:, flag_angle]
        vector_A_norm = vector_A_norm[flag_angle]
        vector_B_norm = vector_B_norm[flag_angle]
        cycle = cycle[flag_angle, :]

        # 1.8.合并所有的参数cycle,vector_A_norm,vector_B_norm,angle_between_vectorAB
        lattice_dim_parameters = np.hstack((cycle, vector_A_norm.reshape(-1, 1)))
        lattice_dim_parameters = np.hstack((lattice_dim_parameters, vector_B_norm.reshape(-1, 1)))
        lattice_dim_parameters = np.hstack(
            (lattice_dim_parameters, np.degrees(np.arccos(angle_between_vectorAB).reshape(-1, 1))))

        # 返回所有的参数
        return lattice_dim_parameters

    def save_and_sort_couple(self, layer_dict):
        def float2line(line):
            string = ''
            for i in range(0, 4):
                string += str('%6.0f' % line[i])
            for i in range(4, 7):
                string += str('%7.2f' % line[i])
            for i in range(7, 11):
                string += str('%6.0f' % line[i])
            for i in range(11, 14):
                string += str('%7.2f' % line[i])
            for i in range(14, 17):
                string += str('%12.2f' % line[i])
            string += ('\n')
            return string

        couple = np.array(layer_dict['parameter'])
        couple = couple.reshape((int(couple.shape[0] / 17), 17))

        with open(layer_dict['hetero_name'], 'w') as f:
            # 合成波和蒸汽波 电子音乐 2020年8月4日看乐队的夏天第二季大波浪乐队 演唱 广仁师兄推荐
            f.write('hetero_name: ' + layer_dict['hetero_name'] + '\n')
            f.write('lattice_mismatch: ' + str(layer_dict['lattice_mismatch']) + '\n')
            f.write('angle_mismatch: ' + str(layer_dict['angle_mismatch']) + '\n')
            f.write('a0: ' + str(layer_dict['a0']) + '\n')
            f.write('b0: ' + str(layer_dict['b0']) + '\n')
            f.write('a1: ' + str(layer_dict['a1']) + '\n')
            f.write('b1: ' + str(layer_dict['b1']) + '\n')
            f.write('m_a0x  n_a0  m_b0_x  n_b0_y  A0  B0  angle0  ')
            f.write('m_a1x  n_a1  m_b1_x  n_b1_y  A1  B1  angle1  ')
            f.write('atom0_layer0  atom1_layer1  atom_hetero\n')
            for i in range(couple.shape[0]):
                f.write(float2line(couple[i]))

    def find_hetero(self, vasp0, vasp1, dim=5, lattice_mismatch=0.05, angle_mismatch=2, angle_need=None):
        # 1.读取POSCAR0.vasp与POSCAR1.vasp寻找单胞,输出POSCAR0，POSCAR1寻找合适的扩包参数
        self.get_cell(vasp0, 'POSCAR0')
        self.get_cell(vasp1, 'POSCAR1')

        # 2.读取POSCAR0/1得到单胞的晶格矢量a0(1),b0(1)(进行过坐标变换),计算单胞面积S0(1)(矢量叉乘)，并进行扩包参数筛选
        print('get cells vectors [a0,b0] and [a1,b1] (Transformation_2D)!\n')
        P = read_POSCAR('POSCAR0')
        atom0_cell = P.calAtomsSum()
        a0, b0 = P.calLatticeMatrix_Transformation_2D()
        s0 = np.linalg.norm(np.cross(a0, b0))

        P = read_POSCAR('POSCAR1')
        atom1_cell = P.calAtomsSum()
        a1, b1 = P.calLatticeMatrix_Transformation_2D()
        s1 = np.linalg.norm(np.cross(a1, b1))

        # 3.进行扩包参数选取
        ldp_0 = self.get_lattice_dim(a=a0, b=b0, max_dim=dim)
        ldp_1 = self.get_lattice_dim(a=a1, b=b1, max_dim=dim)

        # 4.将原胞基矢等信息存入字典
        layer_dict = {'hetero_name': vasp0[:-5] + '_&_' +
                      vasp1[:-5], 'lattice_mismatch': lattice_mismatch, 'angle_mismatch': angle_mismatch, 'a0': a0, 'b0': b0, 'a1': a1, 'b1': b1, 'parameter': []}

        # 5.寻找符合mismatch要求的异质结

        cycle_length_ldp_0 = ldp_0.shape[0]
        cycle_length_ldp_1 = ldp_1.shape[0]

        # mismatch_logical的每列对应angle,
        mismatch_logical = np.zeros((5, cycle_length_ldp_1))
        for i in range(cycle_length_ldp_0):

            '''
            筛选符合异质结条件的组合
            满足angle_mismatch
            满足lattice_mismatch
            '''

            # 筛选符合angle mismatch的组合布尔值 mismatch_logical[0,:]
            mismatch_logical[0, :] = (
                np.abs(np.repeat(ldp_0[i, 6], cycle_length_ldp_1, axis=0) - ldp_1[:, 6]) < angle_mismatch)

            # 筛选符合lattice mismatch的A边组合布尔值 mismatch_logical[1,:],mismatch_logical[2,:]
            # 筛选符合lattice mismatch的B边组合布尔值 mismatch_logical[3,:],mismatch_logical[4,:]
            for j in range(4, 6):
                lattice_diff = np.abs(
                    np.repeat(ldp_0[i, j], cycle_length_ldp_1, axis=0) - ldp_1[:, j])
                mismatch_logical[2 * j - 7, :] = (lattice_diff / np.repeat(ldp_0[i, j],
                                                                           cycle_length_ldp_1, axis=0) <= lattice_mismatch)
                mismatch_logical[2 * j - 6, :] = (lattice_diff / ldp_1[:, j] <= lattice_mismatch)

            # 总结所有mismatch组合的布尔值
            total_mismatch_logical = np.prod(mismatch_logical, axis=0).astype(int)
            # print(total_mismatch_logical)

            # 对合适的组合给出相关的结构信息
            if np.sum(total_mismatch_logical):
                # np.where() 返回索引 https://www.cnblogs.com/massquantity/p/8908859.html
                index_true = np.array(np.where(total_mismatch_logical == 1))

                # print(index_true.shape[1])
                A0 = ldp_0[i, 0] * np.tile(a0, (index_true.shape[1], 1)) + \
                    ldp_0[i, 1] * np.tile(b0, (index_true.shape[1], 1))
                B0 = ldp_0[i, 2] * np.tile(a0, (index_true.shape[1], 1)) + \
                    ldp_0[i, 3] * np.tile(b0, (index_true.shape[1], 1))

                A1 = np.tile(ldp_1[index_true[0], 0], (3, 1)).T * a1 + \
                    np.tile(ldp_1[index_true[0], 1], (3, 1)).T * b1
                B1 = np.tile(ldp_1[index_true[0], 2], (3, 1)).T * a1 + \
                    np.tile(ldp_1[index_true[0], 3], (3, 1)).T * b1

                # 记下扩包后的每层对应的原子数
                atom0_layer0 = np.linalg.norm(np.cross(A0, B0), axis=1) / s0 * atom0_cell
                atom1_layer1 = np.linalg.norm(np.cross(A1, B1), axis=1) / s1 * atom1_cell
                atom_hetero = atom0_layer0 + atom1_layer1

                '''
                总结统计所有的符合条件的信息：
                扩包系数 共八个,a0,b0,a1,b1,atom0_layer0,atom1_layer1,lattice_mismatch=0.05, angle_mismatch=2
                '''

                line = np.tile(ldp_0[i], (index_true.shape[1], 1))  # 7个元素
                line = np.hstack((line, ldp_1[index_true][0]))  # 7个元素
                line = np.hstack((line, atom0_layer0.reshape(-1, 1)))  # 1个
                line = np.hstack((line, atom1_layer1.reshape(-1, 1)))  # 1个
                line = np.hstack((line, atom_hetero.reshape(-1, 1)))  # 一个元素
                layer_dict['parameter'] += list(line.flatten())

        # 6.整理组合的参数进字典
        couple = np.array(layer_dict['parameter'])
        couple = couple.reshape((int(couple.shape[0] / 17), 17))
        couple = couple[(couple[:, -2] != 0)]
        couple = couple[(couple[:, -3] != 0)]
        couple = couple[np.where(np.abs((couple[:, 6] + couple[:, 13]) / 2 - 90) <= 60)]
        sort_parameter0 = couple[np.argsort(couple[:, -1])]
        if angle_need is None:
            sort_parameter = sort_parameter0
        else:
            sort_parameter1 = sort_parameter0[np.argsort(np.abs(sort_parameter0[:, 6] + sort_parameter0[:, 13] - 2 * angle_need))]
            sort_number = len(sort_parameter1[np.where(np.abs(sort_parameter1[:, 6] + sort_parameter1[:, 13] - 2 * angle_need) < 5)])
            sort_parameter1[:sort_number] = sort_parameter1[:sort_number][np.argsort(sort_parameter1[:sort_number, -1])]
            if (sort_parameter1[0][-1] / sort_parameter0[0][-1]) <= 1.2:
                sort_parameter = sort_parameter1
            else:
                sort_parameter = sort_parameter0

        layer_dict['parameter'] = list(sort_parameter.flatten())
        self.layer_dict = layer_dict
        self.save_and_sort_couple(layer_dict)
        return layer_dict

    def redefine_lattice(self, inputfile, outputfile, couple):
        # 保持坐标轴的手性
        z = 1
        if couple[0] * couple[3] - couple[2] * couple[1] < 0:
            z = -1

        os.system('cp ' + inputfile + ' POSCAR')
        os.system('(echo 400; echo %d %d 0; echo %d %d 0; echo 0 0 %d) | vaspkit | grep 123' % (
            int(couple[0]), int(couple[1]), int(couple[2]), int(couple[3]), z))
        os.system('mv SUPERCELL.vasp POSCAR')
        os.system('echo 921 | vaspkit')
        os.system('mv POSCAR_REV %s' % outputfile)
        os.system('rm POSCAR')

    def build_hetero(self, lattice_select=-1, layer_spacing=3, vac_thickness=20, cp=0):
        '''
        lattice:
        0:用下层的晶格参数
        1:用上层的晶格参数
        -1:用平均晶格参数
        '''
        # 1.获取layer的组合信息
        layer_dict = self.layer_dict
        couple = np.array(layer_dict['parameter'])
        line = couple.reshape((int(couple.shape[0] / 17), 17))[cp]
        couple0 = line[0:4]
        couple1 = line[7:11]
        A0 = line[4]
        B0 = line[5]
        angle0 = line[6]
        A1 = line[11]
        B1 = line[12]
        angle1 = line[13]

        # 2.用vaspkit导出来为Direct坐标系,通过read_POSCAR()类方法或许所需结构信息
        self.redefine_lattice('POSCAR0', 'super_POSCAR0', couple0)
        self.redefine_lattice('POSCAR1', 'super_POSCAR1', couple1)
        P0 = read_POSCAR('super_POSCAR0')
        lattice0 = P0.getLatticeMatrix()
        atom0_Direct = P0.getAtomsPositionMatrix()
        atom0_Cartesian = P0.getAtomsPositionMatrix()
        atom0_Info = P0.getAtomsInfo()

        P1 = read_POSCAR('super_POSCAR1')
        lattice1 = P1.getLatticeMatrix()
        atom1_Direct = P1.getAtomsPositionMatrix()
        atom1_Cartesian = P1.getAtomsPositionMatrix()
        atom1_Info = P1.getAtomsInfo()
        # 3.分析每一层的结构信息，包括堆叠层数以及每层说对应的原子数
        layer0_layer_info = dividing_layer(atom0_Cartesian[np.argsort(-atom0_Cartesian[:, 2],)])
        layer1_layer_info = dividing_layer(atom1_Cartesian[np.argsort(-atom1_Cartesian[:, 2],)])

        # 4.分别计算layer0与layer1中inter层间原子的最短键长
        if len(layer0_layer_info) == 1:
            layer0_bond_length = cal_bond_length_1(atom0_Cartesian, list(atom0_Info.values())[0])
        else:
            layer0_bond_length = cal_bond_length_2(layer0_layer_info)

        if len(layer1_layer_info) == 1:
            layer1_bond_length = cal_bond_length_1(atom1_Cartesian, list(atom1_Info.values())[0])
        else:
            layer1_bond_length = cal_bond_length_2(layer1_layer_info)
        print(layer0_bond_length)
        print(layer1_bond_length)

        # 5.计算layer间的间隔距离
        [layer0_std, std_atom0_info] = cal_dividing_layer_space(
            layer0_layer_info, layer0_bond_length)
        [layer1_std, std_atom1_info] = cal_dividing_layer_space(
            layer1_layer_info, layer1_bond_length)

        # 6.选择异质结的晶格矢量,矢量变换
        vector = np.zeros((3, 3))
        if lattice_select == -1:
            A = (A0 + A1) / 2
            B = (B0 + B1) / 2
            angle = np.radians(angle0 + angle1) / 2
            vector[0] = [np.cos(angle / 2) * A,
                         np.sin(angle / 2) * A, 0]  # 平均情况下的A矢量
            vector[1] = [np.cos(angle / 2) * B,
                         np.sin(angle / 2) * B * (-1), 0]  # 平均情况下的B矢量
            vector[2] = [0, 0, 0]
            print(vector)
        if lattice_select == 0:
            A = A0
            B = B0
            angle = np.radians(angle0)
        if lattice_select == 1:
            A = A1
            B = B1
            angle = np.radians(angle1)

        atom0_Cartesian_In_Lattice1 = direct_to_cartesian(atom0_Direct, lattice1, lattice0)
        atom1_Cartesian_In_Lattice0 = direct_to_cartesian(atom1_Direct, lattice0, lattice1)

        atom0_Cartesian_In_hetero = direct_to_cartesian(atom0_Direct, vector, lattice0)
        atom1_Cartesian_In_hetero = direct_to_cartesian(atom1_Direct, vector, lattice1)

        atom1_hetero_C = tran_of_each_layer(
            layer1_std, atom1_Cartesian_In_Lattice0, std_atom1_info, layer1_bond_length)
        atom1_hetero_D = cartesian_to_direct(atom1_hetero_C, lattice0, lattice1)
        atom0_hetero_C = tran_of_each_layer(
            layer0_std, atom0_Cartesian_In_Lattice1, std_atom0_info, layer0_bond_length)
        atom0_hetero_D = cartesian_to_direct(atom0_hetero_C, lattice1, lattice0)

        atom1_hehe_C = tran_of_each_layer(
            layer1_std, atom1_Cartesian_In_hetero, std_atom1_info, layer1_bond_length)
        atom1_hehe_D = cartesian_to_direct(atom1_hehe_C, vector, lattice1)
        atom0_hehe_C = tran_of_each_layer(
            layer0_std, atom0_Cartesian_In_hetero, std_atom0_info, layer0_bond_length)
        atom0_hehe_D = cartesian_to_direct(atom0_hehe_C, vector, lattice0)

        if lattice_select == -1:
            atom0 = np.mat(atom0_hehe_D)
            atom1 = np.mat(atom1_hehe_D)
        if lattice_select == 0:
            atom0 = np.mat(atom0_Direct)
            atom1 = np.mat(atom1_hetero_D)
        if lattice_select == 1:
            atom0 = np.mat(atom0_hetero_D)
            atom1 = np.mat(atom1_Direct)
        # --------------------------------------------------------------------------------------------------------------------
        '''
        (本部分用了vaspkit提供的异质结组合脚本)代码有点凌乱，小修了其中的一部分，后续还需要再修改
        '''
        dis_A = (atom0[:, 2].max(axis=0) - atom0[:, 2].min(axis=0)) * abs(lattice0[2, 2])  # 单层之间的距离
        dis_B = (atom1[:, 2].max(axis=0) - atom1[:, 2].min(axis=0)) * abs(lattice1[2, 2])  # 单层之间的距离
        print(dis_A, dis_B)

        C = dis_A + dis_B + vac_thickness + layer_spacing
        C_A = abs(lattice0[2, 2])
        C_B = abs(lattice1[2, 2])
        vector = np.mat(vector)
        vector[2, 2] = C
        lattice_aver = vector
        atom0_new = atom0
        atom1_new = atom1
        slab_pos = vac_thickness / 2
        distance_A_Direct = atom0[:, 2].min(axis=0) - (slab_pos / C_A)
        atom0_new[:, 2] = (atom0[:, 2] - distance_A_Direct) * C_A / C
        A1_max = atom0_new[:, 2].max(axis=0)
        B_min_Cartesian = (atom1[:, 2].min(axis=0)) * C_B
        B1_min_Cartesian = A1_max * C + layer_spacing
        distance_B_Cartesian = B_min_Cartesian - B1_min_Cartesian
        atom1_new[:, 2] = (atom1[:, 2] * C_B - distance_B_Cartesian) / C
        lattice0[2, 2] = C
        lattice1[2, 2] = C
        name_t = str(layer_dict['hetero_name']) + '_' + str(lattice_select) + \
            '_' + str(vac_thickness) + '_' + str(layer_spacing) + '.vasp'

        if lattice_select == -1:
            building_heterostructure(lattice_aver, name_t=name_t, ele_A=list(
                atom0_Info.keys()), ele_B=list(atom1_Info.keys()), atom_num_A=list(
                    atom0_Info.values()), atom_num_B=list(atom1_Info.values()), atom_A_new=atom0_new, atom_B_new=atom1_new)
        # if lattice_select == 0:
        #     building_heterostructure(lattice0, name_t=name_t, ele_A=list(
        #         atom0_Info.keys()), ele_B=list(atom0_Info.keys()))
        # if lattice_select == 1:
        #     building_heterostructure(lattice1, name_t=name_t, ele_A=list(
        #         atom0_Info.keys()), ele_B=list(atom0_Info.keys()))

        # --------------------------------------------------------------------------------------------------------------------
        return name_t


if __name__ == "__main__":
    E = establish_hetero()
    E.find_hetero('00-BiSeCl-BiTeI.vasp', '02-Tl2Cl2O2-FeOCl.vasp', dim=3, lattice_mismatch=0.05, angle_mismatch=2, angle_need=90)
    E.build_hetero(cp=0)
