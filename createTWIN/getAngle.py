# coding=utf-8
import os
import numpy as np
from collections import OrderedDict
from ReadPOSCAR import POSCAR
import math


class POST():
    def __init__(self, readpath):
        self.readpath = readpath
        p = POSCAR(self.readpath)
        POSCARparameters = p.main()
        self.parameters = POSCARparameters
        self.process = []
        os.system('clear')

    def writePOST(self, filename, path):
        f = open(path + os.sep + filename + '_POST', 'w')

        f.write(filename + '  !BEGIN\n')
        f.write('---------------------------------\n')
        f.write('the input file:' + self.readpath + '\n')
        f.write('---------------------------------\n')
        for i in range(len(self.process)):
            f.write(str(i + 1) + '.' + self.process[i] + '\n')
            f.write('********************************\n')
        f.write('FINISHED!')
        f.close()

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

        return Angle


if __name__ == "__main__":
    path = 'F:/workstation/SnO2[111]Twin/Structure'
    path += '/four_bilayers_Ge_POSCAR11'
    post = POST(readpath=path)
    angle = post.getAngle(refElement1=['Sn', 80], refElement2=['Sn', 81])
    print(angle)
    # post.writePOST(filename='test',path='./5_Degree')
