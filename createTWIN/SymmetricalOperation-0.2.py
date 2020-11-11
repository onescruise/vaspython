#!/usr/bin/python
# coding=utf-8
'''
Created on 14:35, Dec. 12, 2019

Recently updated on 17:45, Dec. 24, 2019

@author: Yilin Zhang

These files are need: POSCAR
'''

import os
import numpy as np
from collections import OrderedDict
from ReadPOSCAR import POSCAR
import math


class SOP():
    def __init__(self, readpath):
        self.readpath=readpath
        p = POSCAR(self.readpath)
        POSCARparameters = p.main()
        self.parameters = POSCARparameters
        self.process = []
        os.system('clear')

    def writePOSCAR(self, filename, path,vesta = True):
        def float2line(line):
            string = '%20.10f' % float(line[0]) + '%21.10f' % float(line[1]) + '%21.10f' % float(line[2]) + '\n'
            line = str(string)
            return line
        if not os.path.exists(path):
            os.mkdir(path)

        f = open(path + os.sep + filename + '_SOP', 'w')

        f.write(filename + '  !BEGIN\n')
        f.write('---------------------------------\n')
        f.write('the input file:'+self.readpath+'\n')
        f.write('---------------------------------\n')
        for i in range(len(self.process)):
            f.write(str(i + 1) + '.' + self.process[i] + '\n')
            f.write('********************************\n')
        f.write('FINISHED!')
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

            ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy()
            for key in ElementsPositionMatrix.keys():
                arr = ElementsPositionMatrix[key]
                for i in range(len(arr)):
                    f.write(float2line(arr[i]))

        if vesta:
            os.chdir(path)
            os.system('VESTA '+filename + '_POSCAR')

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

        K1 = [u * u + (v * v + w * w) * cosA, u * v * (1 - cosA) - w * sinA, u * w * (1 - cosA)
              + v * sinA, (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - cosA) + (b * w - c * v) * sinA]
        K2 = [u * v * (1 - cosA) + w * sinA, v * v + (u * u + w * w) * cosA, v * w * (1 - cosA)
              - u * sinA, (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - cosA) + (c * u - a * w) * sinA]
        K3 = [u * w * (1 - cosA) - v * sinA, v * w * (1 - cosA) + u * sinA, w * w + (u * u + v * v) *
              cosA, (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - cosA) + (a * v - b * u) * sinA]
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

        self.process.append('move2point:\n' + '-refElement:'+str(refElement)+'\n-point:'+str(point))

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

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy()
        ElementsPositionMatrix[refElement[0]][refElement[1] - 1]=np.array(cor)*basicvector

        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('modifyingAtomicCoordinates:\n' + '-refElement:'+str(refElement)+'\n-cor:'+str(cor))

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
        basicvector=np.array([LatticeMatrix[0][0],LatticeMatrix[1][1],LatticeMatrix[2][2]])

        normalVector, inPoint = getNormalVector(np.array(fixedPlane)*basicvector)

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy()
        fixedAxis = np.zeros((2, 3))
        postionRange = np.array(postionRange)
        for key in ElementsPositionMatrix.keys():
            matrix = ElementsPositionMatrix[key]
            for i in range(len(matrix)):
                point = matrix[i]
                if (point[0] >= postionRange[0][0] and point[0] <= postionRange[0][1]):
                    if (point[1] >= postionRange[1][0] and point[1] <= postionRange[1][1]):
                        if (point[2] >= postionRange[2][0] and point[2] <= postionRange[2][1]):
                            point*=basicvector
                            dropfoot = point + normalVector * \
                                (normalVector * (inPoint - point))
                            fixedAxis = [dropfoot, dropfoot +
                                         np.array(parallelAxis)]
                            matrixR = self.transformMatrixR(fixedAxis, angle)
                            matrix[i] = np.dot(np.insert(point, 3, [1], axis=0), matrixR.T)[:-1]
                            matrix[i]/=basicvector
            ElementsPositionMatrix[key] = matrix
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix

        self.process.append('rotationRoundFixedPlane:\n' + '-fixedPlane:\n' + str(np.array(
            fixedPlane)) + '\n-parallelAxis:' + str(np.array(parallelAxis)) + '\n-angle:' + str(angle)+'\n-postionRange:\n'+str(np.array(postionRange)))

        self.cutLattice([[0,1], [0,1], [0,1]])

    def cutLattice(self, cutRange):
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
                matrix = matrix[matrix[:, i] >=
                                cutRange[i][0] * basicvector[i], :]
                matrix = matrix[matrix[:, i] <=
                                cutRange[i][1] * basicvector[i], :]
            ElementsPositionMatrix[key] = matrix
            self.parameters['AtomsInfo'][key] = len(
                ElementsPositionMatrix[key])
        self.parameters['ElementsPositionMatrix'] = ElementsPositionMatrix
        if self.parameters['CoordinateType'] == 'Cartesian':
            delta = np.array(cutRange[:, 1] - cutRange[:, 0])
            self.parameters['LatticeMatrix'] = [LatticeMatrix[0] * delta[0],
                                                LatticeMatrix[1] * delta[1], LatticeMatrix[2] * delta[2]]

        self.process.append('cutLattice:\n' + str(np.array(cutRange)))

    def screen(self, screenPlane, justScreen=True):

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

        # self.process.append('autoMove2Zero:YES‘)

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

    def autoCutLatticeByElement(self, element,cor):
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

        corlist={'a':0,'b':1,'c':2}
        i=int(corlist[cor])
        delta[i] = max(matrix[:, i]) - min(matrix[:, i])
        if delta[i] != 0:
            scale[i] = delta[i] / basicvector[i]
        LatticeMatrix[i] *= scale[i]
        self.parameters['LatticeMatrix'] = LatticeMatrix

        self.process.append('autoMove2ZeroByElement:' + element+'_'+cor)

if __name__ == "__main__":
    # 读入初始的vasp文件，来获取构建参数
    s = SOP(readpath='/media/ones/New Volume/WorkPlatform/P(black).vasp')

    # 进行构建孪晶的操作
    s.Cartesian2Direct()
    s.cutLattice([[0,1], [0.3,0.7], [0, 1]])
    s.simpleDIM(([1, 1, 10]))
    # s.rotationRoundFixedAxis([[0.5,0.5,0.5],[0.5,0.5,1]], angle=45)
    # s.screen(screenPlane=[[0, 0.5, 1], [1, 0.5, 0],[1, 0.5, 1]], justScreen=False)
    #
    # s.autoMove2Zero()
    s.rotationRoundFixedPlane(fixedPlane=[[0, 0, 0.50821], [1, 0, 0.50821], [0, 1, 0.50821]], parallelAxis=[1, 0, 0], angle=-10, postionRange=[[0, 1], [0, 1], [0, 0.50821]])
    s.rotationRoundFixedPlane(fixedPlane=[[0, 0, 0.54179], [1, 0, 0.54179], [0, 1, 0.54179]], parallelAxis=[1, 0, 0], angle=10, postionRange=[[0, 1], [0, 1], [0.54179, 1]])
    # s.cutLattice([[0,1], [0,1], [0,1]])
    s.cutLattice([[0,1], [0,1], [(0.11429+0.09812)/2,(0.88647+0.90264)/2]])
    s.move2point(['P', 4], [0,0,-0.88647+0.90264])
    s.Direct2Cartesian()
    # s.autoMove2ZeroByElement('P')
    # s.autoCutLatticeByElement('P','b')
    # s.autoCutLatticeByElement('P','c')
    s.cutLattice([[0,1], [0,1], [0,0.78836]])
    s.modifyingAtomicCoordinates(['P', 4], [1.00000,0.02427,0.02051])
    # s.move2point(['C', 1], [0.58143,0,0.75000 ])
    s.writePOSCAR(filename='blackPhosphorous_singleLayer_armchair_even', path='/media/ones/New Volume/WorkPlatform',vesta = False)
