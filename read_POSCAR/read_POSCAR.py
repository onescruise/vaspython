

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
