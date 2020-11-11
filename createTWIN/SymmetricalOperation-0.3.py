#!/usr/bin/python
# coding=utf-8
'''
Created on 14:35, Dec. 12, 2019

Recently updated on 20:19, Apr. 8, 2020

@author: Yilin Zhang

These files are need: POSCAR
'''

import os
import numpy as np
from collections import OrderedDict
# from ReadPOSCAR import POSCAR
import math


class POSCAR():
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
                str(self.lines[i + 8]).strip().split()).astype(np.float)
        self.addParameters(key='AtomsPositionMatrix',
                           value=AtomsPositionMatrix)
        return AtomsPositionMatrix

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

    def main(self):
        self.getSystemName()  # 获取体系名称
        self.getScalingFactor()  # 获取缩放系数
        self.getLatticeMatrix()  # 获取晶格参数矩阵
        self.getAtomsInfo()  # 获取原子(数量)信息
        self.getCoordinateType()  # 获取坐标类型
        self.getAtomsPositionMatrix()  # 获取原子位置矩阵

        self.calAtomsSum()  # 计算原子总数
        self.calVolume()  # 计算晶体面积
        self.calElementsPositionMatrix()  # 获取每种元素的位置矩阵

        os.system('clear')  # 注释掉此行可以方便脚本检查
        for key, value in self.parameters.items():
            if 'Matrix' in key:
                print(str(key) + ':\n' + str(value))
            else:
                print(str(key) + ':' + str(value))

        return self.parameters


class SOP():
    def __init__(self, readpath):
        self.readpath = readpath
        p = POSCAR(self.readpath)
        POSCARparameters = p.main()
        self.parameters = POSCARparameters
        self.process = []
        os.system('clear')

    def writePOSCAR(self, filename, path, vesta=True):
        def float2line(line):
            string = '%20.10f' % float(
                line[0]) + '%21.10f' % float(line[1]) + '%21.10f' % float(line[2]) + '\n'
            line = str(string)
            return line
        if not os.path.exists(path):
            os.mkdir(path)

        f = open(path + os.sep + filename + '_SOP', 'w')

        f.write(filename + '\n!BEGIN\n')

        f.write('---------------------------------\n')
        for i in range(len(self.process)):
            f.write(str(i + 1) + '.' + self.process[i] + '\n')
            f.write('********************************\n')
        f.write('FINISHED!\n')
        f.write('---------------------------------\n\n')
        f.write('=================================\n')
        f.write('\nthe input file:' + self.readpath + '\n')
        pos = open(self.readpath)
        lines = pos.readlines()
        pos.close()

        for line in lines:
            f.write(line)
        f.close()

        with open(path + os.sep + filename + '_POSCAR', 'w') as f:
            f.write(self.parameters['SystemName'] + '\n')
            f.write(str(self.parameters['ScalingFactor']) + '\n')

            LatticeMatrix = self.parameters['LatticeMatrix']
            for i in range(len(LatticeMatrix)):
                f.write(float2line(LatticeMatrix[i]))

            AtomsInfo = self.parameters['AtomsInfo']
            string = ['', '']
            for key in AtomsInfo.keys():
                string[0] += key + '    '
                string[1] += str(AtomsInfo[key]) + '    '
            string[0] += '\n'
            string[1] += '\n'
            f.write(string[0])
            f.write(string[1])

            f.write(self.parameters['CoordinateType'] + '\n')

            ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
            )
            for key in ElementsPositionMatrix.keys():
                arr = ElementsPositionMatrix[key]
                for i in range(len(arr)):
                    f.write(float2line(arr[i]))

        if vesta:
            os.chdir(path)
            os.system('VESTA ' + filename + '_POSCAR')

    def Direct2Cartesian(self):
        if self.parameters['CoordinateType'] == 'Cartesian':
            self.process.append('Direct2Cartesian:NO')
            return 0

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        LatticeMatrix = self.parameters['LatticeMatrix']

        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            for i in range(len(matrix)):
                matrix[i] = np.dot(matrix[i], LatticeMatrix)
            ElementsPositionMatrix[key] = matrix
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix
        self.parameters['CoordinateType'] = 'Cartesian'

        self.process.append('Direct2Cartesian:YES')

    def Cartesian2Direct(self):
        if self.parameters['CoordinateType'] == 'Direct':
            self.process.append('Cartesian2Direct:NO')
            return 0

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        LatticeMatrix = self.parameters['LatticeMatrix'].copy()
        for i in range(3):
            for j in range(3):
                if LatticeMatrix[i][j] != 0:
                    LatticeMatrix[i][j] = 1.0 / LatticeMatrix[i][j]

        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            for i in range(len(matrix)):
                matrix[i] = np.dot(matrix[i], LatticeMatrix)
            ElementsPositionMatrix[key] = matrix
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix
        self.parameters['CoordinateType'] = 'Direct'

        self.process.append('Cartesian2Direct:YES')

    def transformMatrixR(self, fixedAxis, angle):
        '''
        fixedAxis=[[0,0,0],[0,0,1]]:对应的旋转轴的起点和旋转轴的终点
        angle:绕轴旋转的角度
        注意：尽量用abc相等的包进行旋转，因为用的是分数坐标系
        '''
        angle = math.radians(angle)
        fixedAxis = np.array(fixedAxis)

        a = fixedAxis[0][0]
        b = fixedAxis[0][1]
        c = fixedAxis[0][2]

        vector = fixedAxis[1] - fixedAxis[0]

        u = vector[0] / np.linalg.norm(vector)
        v = vector[1] / np.linalg.norm(vector)
        w = vector[2] / np.linalg.norm(vector)

        cosA = np.cos(angle)
        sinA = np.sin(angle)

        K1 = [u * u + (v * v + w * w) * cosA, u * v * (1 - cosA) - w * sinA, u * w * (1 - cosA) +
              v * sinA, (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - cosA) + (b * w - c * v) * sinA]
        K2 = [u * v * (1 - cosA) + w * sinA, v * v + (u * u + w * w) * cosA, v * w * (1 - cosA) -
              u * sinA, (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - cosA) + (c * u - a * w) * sinA]
        K3 = [u * w * (1 - cosA) - v * sinA, v * w * (1 - cosA) + u * sinA, w * w + (u * u + v * v)
              * cosA, (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - cosA) + (a * v - b * u) * sinA]
        K4 = [0, 0, 0, 1]
        return np.array([K1, K2, K3, K4])

    def move2point(self, refElement, point):
        '''
        将某个元素的第几个原子refElement移动到某个点point
        refElement=['Sn',1]
        '''
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        refElementPositionMatrix = ElementsPositionMatrix[refElement[0]
                                                          ][refElement[1] - 1]
        delta_r = np.array(point) - refElementPositionMatrix
        for key in ElementsPositionMatrix.keys():
            ElementsPositionMatrix[key] += delta_r

        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('move2point:\n' + '-refElement:'
                            + str(refElement) + '\n-point:' + str(point))

    def modifyingAtomicCoordinates(self, refElement, cor):
        '''
        将某个元素的第几个原子refElement坐标改为cor
        refElement=['Sn',1]
        '''
        LatticeMatrix = self.parameters['LatticeMatrix']
        if self.parameters['CoordinateType'] == 'Direct':
            a = b = c = 1.0
        else:
            a = np.linalg.norm(LatticeMatrix[0])
            b = np.linalg.norm(LatticeMatrix[1])
            c = np.linalg.norm(LatticeMatrix[2])
        basicvector = np.array([a, b, c])

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        ElementsPositionMatrix[refElement[0]
                               ][refElement[1] - 1] = np.array(cor) * basicvector

        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('modifyingAtomicCoordinates:\n'
                            + '-refElement:' + str(refElement) + '\n-cor:' + str(cor))

    def delOneAtom(self, refElement):
        '''
        将某个元素的第几个原子refElement坐标改为cor
        refElement=['Sn',1]
        '''
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        ElementsPositionMatrix[refElement[0]] = np.delete(
            ElementsPositionMatrix[refElement[0]], refElement[1] - 1, axis=0)

        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        AtomsInfo = self.parameters['AtomsInfo'].copy()
        AtomsInfo[refElement[0]] -= 1
        self.parameters['AtomsInfo'] = AtomsInfo

        self.process.append('delAtom:\n'
                            + '-refElement:' + str(refElement))

    def simpleDIM(self, dim):
        '''
        简单的扩胞,[x,y,z]对应x,y,z方向上的扩胞倍数
        '''
        dim = np.array(dim)
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )

        LatticeMatrix = self.parameters['LatticeMatrix']
        if self.parameters['CoordinateType'] == 'Direct':
            a = np.array([1, 0, 0])
            b = np.array([0, 1, 0])
            c = np.array([0, 0, 1])

        for key in ElementsPositionMatrix.keys():
            for i in range(1, dim[0]):
                ElementsPositionMatrix[key] = np.vstack(
                    (ElementsPositionMatrix[key], ElementsPositionMatrix[key] + i * a))
            for i in range(1, dim[1]):
                ElementsPositionMatrix[key] = np.vstack(
                    (ElementsPositionMatrix[key], ElementsPositionMatrix[key] + i * b))
            for i in range(1, dim[2]):
                ElementsPositionMatrix[key] = np.vstack(
                    (ElementsPositionMatrix[key], ElementsPositionMatrix[key] + i * c))
            ElementsPositionMatrix[key] /= np.array(dim)
            self.parameters['AtomsInfo'][key] = len(
                ElementsPositionMatrix[key])

        self.parameters['LatticeMatrix'] = [
            LatticeMatrix[0] * dim[0], LatticeMatrix[1] * dim[1], LatticeMatrix[2] * dim[2]]
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('simpleDIM:\n' + str(np.array(dim)))

        self.cutLattice([[0, 1], [0, 1], [0, 1]], flag=False)

    def rotationRoundFixedAxis(self, fixedAxis, angle):

        matrixR = self.transformMatrixR(fixedAxis, angle)

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            matrix = np.insert(matrix, 3, [1], axis=1)
            for i in range(len(matrix)):
                matrix[i] = np.dot(matrix[i], matrixR.T)
            ElementsPositionMatrix[key] = matrix[:, 0:-1]
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('rotationRoundFixedAxis:\n' + '-fixedAxis:\n'
                            + str(np.array(fixedAxis)) + '\n-angle:' + str(angle))

        self.cutLattice([[0, 1], [0, 1], [0, 1]], flag=False)

        # LatticeMatrix=self.parameters['LatticeMatrix']
        # basicvector=np.array([LatticeMatrix[0][0],LatticeMatrix[1][1],LatticeMatrix[2][2],1])
        # vector=np.array(fixedAxis[1])-np.array(fixedAxis[0])
        # matrixR = transformMatrixR([[0,0,0],vector], angle)
        # basicvector = np.dot(basicvector, matrixR.T)
        # basicvector =basicvector[0:-1]
        # a=[basicvector[0],0,0]
        # b=[0,basicvector[1],0]
        # c=[0,0,basicvector[2]]
        # self.parameters['LatticeMatrix'] = np.array([a,b,c])

    def rotationRoundFixedPlane(self, fixedPlane, parallelAxis, angle, postionRange):
        '''
        fixedPlane:[[1,1,0],[0,1,0],[0,1,1]],平面由三点构成
        parallelAxis:平行轴[0，0，1]
        angle:绕平行轴转动的角度


        尚未写入的功能：选取某个范围绕着给平面的平行轴作个转动
        postionRange：[[0.5,1],[0.5,1],[0.5,1]]转动范围
        '''
        def getNormalVector(fixedPlane):
            fixedPlane = np.array(fixedPlane)
            vector1 = fixedPlane[1] - fixedPlane[0]
            vector2 = fixedPlane[2] - fixedPlane[0]
            normalVector = np.cross(vector1, vector2)
            normalVector = normalVector / np.linalg.norm(normalVector)
            return normalVector, fixedPlane[0]

        LatticeMatrix = self.parameters['LatticeMatrix']
        basicvector = np.array(
            [LatticeMatrix[0][0], LatticeMatrix[1][1], LatticeMatrix[2][2]])

        normalVector, inPoint = getNormalVector(
            np.array(fixedPlane) * basicvector)

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        fixedAxis = np.zeros((2, 3))
        postionRange = np.array(postionRange)
        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            for i in range(len(matrix)):
                point = matrix[i]
                if (point[0] >= postionRange[0][0] and point[0] <= postionRange[0][1]):
                    if (point[1] >= postionRange[1][0] and point[1] <= postionRange[1][1]):
                        if (point[2] >= postionRange[2][0] and point[2] <= postionRange[2][1]):
                            point *= basicvector
                            dropfoot = point + normalVector * \
                                (normalVector * (inPoint - point))
                            fixedAxis = [dropfoot, dropfoot
                                         + np.array(parallelAxis)]
                            matrixR = self.transformMatrixR(fixedAxis, angle)
                            matrix[i] = np.dot(
                                np.insert(point, 3, [1], axis=0), matrixR.T)[:-1]
                            matrix[i] /= basicvector
            ElementsPositionMatrix[key] = matrix
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('rotationRoundFixedPlane:\n' + '-fixedPlane:\n' + str(np.array(
            fixedPlane)) + '\n-parallelAxis:' + str(np.array(parallelAxis)) + '\n-angle:' + str(angle) + '\n-postionRange:\n' + str(np.array(postionRange)))

        self.cutLattice([[0, 1], [0, 1], [0, 1]], flag=False)

    def cutLattice(self, cutRange, flag=True):
        '''
        cutRange:需要切割的范围
        '''
        LatticeMatrix = self.parameters['LatticeMatrix']
        if self.parameters['CoordinateType'] == 'Direct':
            a = b = c = 1.0
        else:
            a = np.linalg.norm(LatticeMatrix[0])
            b = np.linalg.norm(LatticeMatrix[1])
            c = np.linalg.norm(LatticeMatrix[2])
        basicvector = np.array([a, b, c])

        cutRange = np.array(cutRange)
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            for i in range(3):
                matrix = matrix[matrix[:, i]
                                >= cutRange[i][0] * basicvector[i], :]
                matrix = matrix[matrix[:, i]
                                <= cutRange[i][1] * basicvector[i], :]
            ElementsPositionMatrix[key] = matrix
            self.parameters['AtomsInfo'][key] = len(
                ElementsPositionMatrix[key])
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix
        if self.parameters['CoordinateType'] == 'Cartesian':
            delta = np.array(cutRange[:, 1] - cutRange[:, 0])
            self.parameters['LatticeMatrix'] = [LatticeMatrix[0] * delta[0],
                                                LatticeMatrix[1] * delta[1], LatticeMatrix[2] * delta[2]]
        if flag:
            self.process.append('cutLattice:\n' + str(np.array(cutRange)))

    def localAtomicCoordinatesComponentReset(self, zoom, reset):
        '''
        zoom:需要修改的范围
        reset:需要修正的坐标分量['a', 0.]
        '''
        if reset[0] == 'a':
            axis = 0
        if reset[0] == 'b':
            axis = 1
        if reset[0] == 'c':
            axis = 2

        component = float(reset[1])

        LatticeMatrix = self.parameters['LatticeMatrix']

        zoom = np.array(zoom)
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )

        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            for i in range(len(matrix)):
                point = matrix[i]
                if (point[0] >= zoom[0][0] and point[0] <= zoom[0][1]):
                    if (point[1] >= zoom[1][0] and point[1] <= zoom[1][1]):
                        if (point[2] >= zoom[2][0] and point[2] <= zoom[2][1]):
                            matrix[i][axis] = component
            ElementsPositionMatrix[key] = matrix
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('localAtomicCoordinatesComponentReset:\n'
                            + str(np.array(zoom)) + '\nreset:' + str(np.array(reset)))

    def screen(self, screenPlane, justScreen=True):
        '''
        镜面对称：
        screenPlane:[]
        '''

        def getNormalVector(fixedPlane):
            fixedPlane = np.array(fixedPlane)
            vector1 = fixedPlane[1] - fixedPlane[0]
            vector2 = fixedPlane[2] - fixedPlane[0]
            normalVector = np.cross(vector1, vector2)
            normalVector = normalVector / np.linalg.norm(normalVector)
            return -normalVector, fixedPlane[0]

        normalVector, inPoint = getNormalVector(screenPlane)
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            screenpoint = np.zeros((len(matrix), 3))
            for i in range(len(matrix)):
                point = matrix[i]
                dropfoot = point + normalVector * \
                    (normalVector * (inPoint - point))
                screenpoint[i] = 2 * dropfoot - point
            ElementsPositionMatrix[key] = screenpoint
            if not justScreen:
                ElementsPositionMatrix[key] = np.unique(
                    np.vstack((matrix, screenpoint)), axis=0)

            self.parameters['AtomsInfo'][key] = len(
                ElementsPositionMatrix[key])
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('screen:\n' + str(np.array(screenPlane)))

    def autoMove2Zero(self):
        '''
        自动移动到原点
        '''
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )

        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            minA = min(matrix[:, 0])
            minB = min(matrix[:, 1])
            minC = min(matrix[:, 2])
            delta = np.array([minA, minB, minC])
            for key1 in ElementsPositionMatrix.keys():
                ElementsPositionMatrix[key1] -= delta
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('autoMove2Zero:YES')

        self.cutLattice([[0, 1], [0, 1], [0, 1]], flag=False)

    def autoMove2ZeroByElement(self, element):
        '''
        通过某个元素自动移动到原点
        '''
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )

        matrix = ElementsPositionMatrix[element]
        minA = min(matrix[:, 0])
        minB = min(matrix[:, 1])
        minC = min(matrix[:, 2])
        delta = np.array([minA, minB, minC])
        for key in ElementsPositionMatrix.keys():
            ElementsPositionMatrix[key] -= delta
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('autoMove2ZeroByElement:' + element)

        self.cutLattice([[0, 1], [0, 1], [0, 1]], flag=False)

    def autoCutLatticeByElement(self, element, cor):
        LatticeMatrix = self.parameters['LatticeMatrix']
        a = np.linalg.norm(LatticeMatrix[0])
        b = np.linalg.norm(LatticeMatrix[1])
        c = np.linalg.norm(LatticeMatrix[2])
        basicvector = np.array([a, b, c])

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        matrix = ElementsPositionMatrix[element]

        delta = np.ones(3)
        scale = np.ones(3)

        corlist = {'a': 0, 'b': 1, 'c': 2}
        i = int(corlist[cor])
        delta[i] = max(matrix[:, i]) - min(matrix[:, i])
        if delta[i] != 0:
            scale[i] = delta[i] / basicvector[i]
        LatticeMatrix[i] *= scale[i]
        self.parameters['LatticeMatrix'] = LatticeMatrix

        self.process.append('autoMove2ZeroByElement:' + element + '_' + cor)

    def getAngle(self, refElement1, refElement2):
        '''
        选定某个元素的两个原子，来给出他们相对的夹角，前者作为参考系
        refElement1=['Sn',12]
        '''
        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        LatticeMatrix = self.parameters['LatticeMatrix']
        basicvector = np.array(
            [LatticeMatrix[0][0], LatticeMatrix[1][1], LatticeMatrix[2][2]])
        refElement1 = np.array(
            ElementsPositionMatrix[refElement1[0]][refElement1[1] - 1])
        refElement2 = np.array(
            ElementsPositionMatrix[refElement2[0]][refElement2[1] - 1])
        vector = (refElement2 - refElement1) * basicvector
        normalVector = vector / np.linalg.norm(vector)

        gamma = np.degrees(np.arccos(normalVector[2]))
        lineXY = np.sqrt(normalVector[0]**2 + normalVector[1]**2)
        alpha = np.degrees(np.arccos(normalVector[0] / lineXY))
        beta = np.degrees(np.arccos(normalVector[1] / lineXY))

        Angle = np.array([alpha, beta, gamma])

        self.process.append('getAngle:\n' + '-refElement:\n' + str(refElement1) + '\n' + str(refElement2) +
                            '\n*Return:' + '\nalpha:' + str(Angle[0]) + '\nbeta :' + str(Angle[1]) + '\ngamma:' + str(Angle[2]))

        print(Angle)
        return Angle


if __name__ == "__main__":
    # 读入初始的vasp文件，来获取构建参数
    if False:
        path = '/media/ones/My Passport/MATERIAL/Ge/two_bilayer'
        s = SOP(readpath=path + '/POSCAR')
        # s.Cartesian2Direct()
        s.simpleDIM(([6, 6, 6]))
        s.rotationRoundFixedAxis([[0.5, 0.5, 0.5], [0.5, 1, 0.5]], angle=45)
        s.rotationRoundFixedAxis(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 1]], angle=35.26438786)
        s.rotationRoundFixedAxis([[0.5, 0.5, 0.5], [1, 0.5, 0.5]], angle=60)
        s.cutLattice([[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]])
        s.Direct2Cartesian()
        s.cutLattice(
            [[0.30754, 0.59622], [0.36391, 0.56803], [0.26429, 0.38214]])
        s.writePOSCAR(filename='step1_Ge', path=path, vesta=True)

    if False:
        path = '/media/ones/My Passport/MATERIAL/Ge/two_bilayer/step2'
        s = SOP(readpath=path + '/POSCAR')
        s.simpleDIM(([4, 1, 1]))
        plane_a = (0.45384 + 0.51634) / 2
        s.cutLattice([[0, plane_a], [0, 1], [0, 1]])
        s.screen(screenPlane=[[plane_a, 0, 1], [plane_a, 1, 0], [
                 plane_a, 0, 0]], justScreen=False)
        s.cutLattice(
            [[0.34967 / 2 + 0.28717 / 2, 0.62051 / 2 + 0.68301 / 2], [0, 1], [0, 1]])
        s.move2point(['Ge', 1], [0.34967 / 2 - 0.28717 / 2, 0.28287, 0.74267])
        s.Direct2Cartesian()
        s.cutLattice(
            [[0, 0.30208 + 0.34967 / 2 - 0.28717 / 2], [0, 1], [0, 1]])
        s.writePOSCAR(filename='step2_Ge', path=path, vesta=True)

    if False:
        path = '/media/ones/My Passport/MATERIAL/Ge/four_bilayer'
        s = SOP(readpath=path + '/POSCAR')
        s.simpleDIM(([4, 1, 1]))
        plane_a = (0.45384 + 0.51634) / 2
        s.cutLattice([[0, plane_a], [0, 1], [0, 1]])
        s.screen(screenPlane=[[plane_a, 0, 1], [plane_a, 1, 0], [
                 plane_a, 0, 0]], justScreen=False)
        s.cutLattice(
            [[0.26634 / 2 + 0.20384 / 2, 0.76634 / 2 + 0.70384 / 2], [0, 1], [0, 1]])
        s.move2point(['Ge', 1], [0.26634 / 2 - 0.20384 / 2, 0.11620, 0.24266])
        s.Direct2Cartesian()
        s.cutLattice(
            [[0, 0.46875 + 0.26634 / 2 - 0.20384 / 2], [0, 1], [0, 1]])
        s.writePOSCAR(filename='four_bilayers_Ge', path=path, vesta=True)

    if False:
        path = 'F:/MATERIAL/ZnF2'
        s = SOP(readpath=path + '/CONTCAR')
        s.simpleDIM(([4, 4, 6]))
        # s.getAngle(refElement1=['Sn', 210], refElement2=['Sn', 319])
        s.rotationRoundFixedAxis([[0.5, 0.5, 0.5], [0.5, 0.5, 1.0]], angle=45)
        s.rotationRoundFixedAxis(
            [[0.5, 0.5, 0.5], [0.5, 1, 0.5]], angle=115.36604174)
        s.rotationRoundFixedAxis(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 1]], angle=23.07186614)

        s.move2point(['Zn', 72], [0.5, 0.50034, 0.89087])
        # #
        plane_a = 0.5
        s.cutLattice([[0, plane_a], [0, 1], [0, 1]])
        # s.simpleDIM(([2, 1, 1]))
        # s.cutLattice([[0, 0.32], [0, 1], [0, 1]])
        plane_b = 0.5
        s.screen(screenPlane=[[plane_b, 0, 1], [plane_b, 1, 0], [
                 plane_b, 0, 0]], justScreen=False)
        #
        s.rotationRoundFixedAxis(
            [[1, 0.5, 0.5], [0.5, 0.5, 0.5]], angle=39.551623820925315)
        s.cutLattice([[0, 1], [0.37, 0.63], [0.32, 0.63]])
        # s.autoMove2ZeroByElement('Zn')
        #
        s.move2point(['Zn', 1], [0, 0.37615, 0.32686])
        s.move2point(['Zn', 2], [0.00040, 0, 0.62732])
        s.move2point(['Zn', 1], [0.00000, 0.00084, 0])
        # # # # s.move2point(['Ge', 1], [0.26634/2-0.20384/2, 0.11620, 0.24266])
        s.Direct2Cartesian()
        s.cutLattice([[0,  0.8345], [0, 0.2509], [0, 0.3012]])
        s.writePOSCAR(filename='ZnF2101TWIN', path=path, vesta=True)

    # s.cutLattice([[0.25816,0.75818], [0,1], [0,1]])
    # s.move2point(['C', 3], [(0.02131+0.08362)/2+0.0001,0.28287,0.74266])
    # s.Direct2Cartesian()
    # s.cutLattice([[0,0.50000], [0,1], [0,1]])
    # s.autoCutLatticeByElement('C','a')
    # s.autoCutLatticeByElement('C','b')
    # s.autoCutLatticeByElement('C','c')
    # s.autoMove2Zero()
    # s.cutLattice([[0,1], [0.22221,1], [0 ,1]])
    # s.cutLattice([[0.28349,0.71651], [0,1], [0.43136 ,0.63136]])

    # s.screen(screenPlane=[[0, 0.5, 1], [1, 0.5, 0],[1, 0.5, 1]], justScreen=False)
    #
    #
    # s.rotationRoundFixedPlane(fixedPlane=[[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]], parallelAxis=[0, 1, 0], angle=-10, postionRange=[[0, 1], [0, 1], [0, 0.5]])
    # s.rotationRoundFixedPlane(fixedPlane=[[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5]], parallelAxis=[0, 1, 0], angle=10, postionRange=[[0, 1], [0, 1], [0.5, 1]])
    # s.cutLattice([[0.25463,0.50463], [0,1], [0.19647 ,0.78736 ]])
    # s.cutLattice([[0.25379,0.74621], [0,1], [0,1]])
    # s.move2point(['P', 4], [0,0,-0.88647+0.90264])
    #
    # s.cutLattice([[0.25463,0.50463], [0,1], [0.19647 ,0.78736 ]])
    # # s.autoMove2ZeroByElement('P')
    # s.autoCutLatticeByElement('P','a')
    # s.autoCutLatticeByElement('P','c')

    # s.modifyingAtomicCoordinates(['P', 4], [1.00000,0.02427,0.02051])
    # s.move2point(['C', 1], [0.58143,0,0.75000 ])
    if False:
        path = 'F:/workstation/SnO2_301twin'
        nameProject = 'SnO2_301twin'
        s = SOP(readpath=path + '/CONTCAR')
        s.simpleDIM(([4, 1, 6]))
        # s.getAngle(refElement1=['Sn', 38], refElement2=['Sn', 7])
        s.rotationRoundFixedAxis(
            [[0.5, 0.5, 0.5], [0.5, 1.0, 0.5]], angle=-144.88996438 + 90)
        # s.getAngle(refElement1=['Sn', 9], refElement2=['Sn', 28])
        s.rotationRoundFixedAxis(
            [[0.5, 0.5, 0.5], [0.5, 1, 0.5]], angle=82.16022023 - 90)

        # s.rotationRoundFixedAxis(
        #     [[0.5, 0.5, 0.5], [0.5, 0.5, 1]], angle=23.07186614)
        #
        # s.move2point(['Zn', 72], [0.5, 0.50034, 0.89087])
        # # #
        plane_a = 0.512
        s.cutLattice([[0, 1], [0, 1], [0, plane_a]])
        s.cutLattice([[0.22, 0.83], [0, 1], [0, 1]])
        s.localAtomicCoordinatesComponentReset(
            zoom=[[0,  1], [0, 1], [0.45, 1]], reset=['c', 0.5])
        s.localAtomicCoordinatesComponentReset(
            zoom=[[0, 0.271], [0, 1], [0, 1]], reset=['a', 0.237])
        s.localAtomicCoordinatesComponentReset(
            zoom=[[0.779, 1], [0, 1], [0, 1]], reset=['a', 0.813])
        s.localAtomicCoordinatesComponentReset(
            zoom=[[0,  1], [0, 1], [0, 0.062]], reset=['c', 0.052])
        s.screen(screenPlane=[[1, 0, 0.5], [1, 1, 0.5],
                              [0, 1, 0.5]], justScreen=False)

        s.autoMove2ZeroByElement('Sn')
        s.Direct2Cartesian()
        s.cutLattice([[0, 0.57], [0, 1], [0, 0.89]])
        s.writePOSCAR(filename=nameProject, path=path, vesta=True)

    if False:
        path = 'F:/workstation/SnO2_doubletwin'
        nameProject = 'SnO2_doubletwin'
        s = SOP(readpath=path + '/CONTCAR')
        s.simpleDIM([2, 1, 1])
        s.cutLattice([[0, 0.51], [0, 1], [0, 1]])
        s.screen(screenPlane=[[0, 0, 1], [0, 1, 0],
                              [0, 1, 1]], justScreen=False)
        s.Direct2Cartesian()
        s.writePOSCAR(filename=nameProject, path=path, vesta=True)

    if True:
        path = 'F:/2020/SnO2/SnO2_301minus'
        nameProject = 'SnO2_301twinMinus'
        s = SOP(readpath=path + '/POSCAR')
        s.cutLattice([[0, 1], [0, 1], [0.24,  0.76]])
        s.localAtomicCoordinatesComponentReset(
            zoom=[[0,  1], [0, 1], [0.24, 0.26]], reset=['c', 0.25])
        s.localAtomicCoordinatesComponentReset(
            zoom=[[0,  1], [0, 1], [0.68, 1]], reset=['c', 0.753])
        # s.move2point(['Sn', 6], [0.25514, 1.00000, 0])
        # s.cutLattice([[0, 1], [0, 1], [0, 0.5]])
        s.Direct2Cartesian()
        s.cutLattice([[0, 1], [0, 1], [0.25, 0.75]])
        s.writePOSCAR(filename=nameProject, path=path, vesta=True)
        # s.simpleDIM(([4, 1, 6]))
        # # s.getAngle(refElement1=['Sn', 38], refElement2=['Sn', 7])
        # s.rotationRoundFixedAxis(
        #     [[0.5, 0.5, 0.5], [0.5, 1.0, 0.5]], angle=-144.88996438 + 90)
        # # s.getAngle(refElement1=['Sn', 9], refElement2=['Sn', 28])
        # s.rotationRoundFixedAxis(
        #     [[0.5, 0.5, 0.5], [0.5, 1, 0.5]], angle=82.16022023 - 90)
        #
        # # s.rotationRoundFixedAxis(
        # #     [[0.5, 0.5, 0.5], [0.5, 0.5, 1]], angle=23.07186614)
        # #
        # # s.move2point(['Zn', 72], [0.5, 0.50034, 0.89087])
        # # # #
        # plane_a = 0.512
        # s.cutLattice([[0, 1], [0, 1], [0, plane_a]])
        # s.cutLattice([[0.22, 0.83], [0, 1], [0, 1]])
        # s.localAtomicCoordinatesComponentReset(
        #     zoom=[[0,  1], [0, 1], [0.45, 1]], reset=['c', 0.5])
        # s.localAtomicCoordinatesComponentReset(
        #     zoom=[[0, 0.271], [0, 1], [0, 1]], reset=['a', 0.237])
        # s.localAtomicCoordinatesComponentReset(
        #     zoom=[[0.779, 1], [0, 1], [0, 1]], reset=['a', 0.813])
        # s.localAtomicCoordinatesComponentReset(
        #     zoom=[[0,  1], [0, 1], [0, 0.062]], reset=['c', 0.052])
        # s.screen(screenPlane=[[1, 0, 0.5], [1, 1, 0.5],
        #                       [0, 1, 0.5]], justScreen=False)
        #
        # s.autoMove2ZeroByElement('Sn')
