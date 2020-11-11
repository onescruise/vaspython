#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
from collections import OrderedDict


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
        ScalingFactor = np.array(str(self.lines[1]).strip().split()).astype(np.float)[0]
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


if __name__ == "__main__":
    p = POSCAR(path='./SnO2.vasp')
    PoscarParameters = p.main()  # 获得所需的参数
