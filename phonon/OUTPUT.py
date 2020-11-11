import os


class POSCAR():
    def __init__(self,path=os.getcwd()):
        self.path=path

    def getVolume(self):
        """
        Get unit cell volume
        """
        import numpy as np
        f = open(self.path+'/CONTCAR', 'r')
        lines = f.readlines()
        sf = np.array(str(lines[1]).strip().split()).astype(np.float)
        a = np.array(str(lines[2]).strip().split()).astype(np.float) * sf
        b = np.array(str(lines[3]).strip().split()).astype(np.float) * sf
        c = np.array(str(lines[4]).strip().split()).astype(np.float) * sf
        Volume = np.dot(np.cross(a, b), c)
        return Volume


class OUTCAR():
    def __init__(self,path=os.getcwd()):
        self.path = path
        self.parameter = {}

    def getParameter(self):
        '''
        get some parameters in OUTCAR
        '''
        import re

        f = open(self.path+"/OUTCAR", 'r')
        lines = f.readlines()
        for line in lines:
            if "LORBIT" in line:
                LORBIT = re.compile(
                    r"(?<=LORBIT =)\s*\d+\.?\d*").findall(line)
                LORBIT = list(map(int, LORBIT))
                if LORBIT != []:
                    self.parameter['LORBIT'] = LORBIT[0]  # print(line)
            if "ISPIN" in line:
                ISPIN = re.compile(
                    r"(?<=ISPIN  =)\s*\d+\.?\d*").findall(line)
                ISPIN = list(map(int, ISPIN))
                if ISPIN != []:
                    self.parameter['ISPIN'] = ISPIN[0]
            if "energy  without entropy=" in line:
                Enthalpy = re.compile(
                    r"(?<=energy  without entropy=)\s*\-\d+\.?\d*").findall(line)
                Enthalpy = list(map(float, Enthalpy))
                if Enthalpy != []:
                    self.parameter['Enthalpy'] = Enthalpy[0]

        f.close()
        # print(self.parameter['ISPIN'])
        # test
        return self.parameter


if __name__ == "__main__":
    poscar = POSCAR()
    print(poscar.getVolume())

    outcar = OUTCAR()
    print(outcar.getParameter())
