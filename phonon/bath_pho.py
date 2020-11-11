
import os
import shutil
import tools
import pho

path = '/home/one/Downloads/OPT'


class Bath_pho():
    def __init__(self, path):
        '''
        Set basic parameters
        '''
        self.number_minusFiles = 5
        self.number_plusFiles = 5

        self.DimNumberA = 1
        self.DimNumberB = 3
        self.DimNumberC = 4

        self.path = path

        self.folder = []
        for i in range(self.number_minusFiles):
            self.folder.append('minus' + str(self.number_minusFiles - i))
        for i in range(self.number_plusFiles):
            self.folder.append('plus' + str(i + 1))

    def CreatePOSCAR_number_File(self):
        """
        Create disp-{number} files.
        """

        files = os.listdir(os.getcwd())

        number_dispFile = 0
        for filename in files:
            if (filename.find('POSCAR-') != -1):
                number_dispFile += 1
        for i in range(1, number_dispFile + 1):
            if i < 100:
                if i < 10:
                    order = '00' + str(i)
                else:
                    order = '0' + str(i)
            else:
                order = str(i)

            newfile = 'disp-' + order
            if not os.path.exists(newfile):
                os.mkdir(newfile)

            shutil.copy2('POSCAR-' + order, newfile)
            shutil.move('POSCAR-' + order, newfile + '/POSCAR')
            # os.remove('POSCAR-' + order)
            if not os.path.exists('POSCAR-' + order):
                tools.Readme("Creating folders of POSCAR-" +
                             order + " is finished!")

    def Mkdir(self):

        os.chdir(self.path)

        if not os.path.exists('bath_pho'):
            os.mkdir('bath_pho')

        os.chdir('bath_pho')
        for folder in self.folder:
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.copy2(self.path + os.sep + folder + '/CONTCAR', folder)
            os.chdir(folder)
            os.rename('CONTCAR', 'POSCAR')
            os.system('''phonopy -d --dim="%d %d %d" --tolerance=1e-3''' %
                      (self.DimNumberA, self.DimNumberB, self.DimNumberC))
            self.CreatePOSCAR_number_File()
            os.chdir('..')
        os.chdir('..')


if __name__ == "__main__":
    os.mkdir(path + '/bath_pho')
    shutil.copy2('submit_bath_pho.py', path + '/bath_pho')
    bath_pho = Bath_pho(path)
    bath_pho.Mkdir()


# def sumbit():
#     filelist = ['minus3', 'minus2', 'minus1', 'plus1', 'plus2', 'plus3']
#
#     dispNumber = 1
#
#     def getDispList(dispNumber):
#         dispList = []
#         for i in range(1, dispNumber + 1):
#             if i < 10:
#                 number = str('00') + str(i)
#             elif i < 100:
#                 number = str('0') + str(i)
#             else:
#                 number = str(i)
#             dispList.append('disp-' + number)
#         return dispList
#
#     for folder in filelist:
#         os.chdir(folder)
#         for disp in getDispList(dispNumber):
#             shutil.copy2('../' + 'INCAR', disp)
#             shutil.copy2('../' + 'POTCAR', disp)
#             shutil.copy2('../' + 'vaspwz.pbs', disp)
#             os.chdir(file)
#             try:
#                 os.system('qsub vaspwz.pbs')
#             except IOError:
#                 print(file + ' Submit failed')
#             os.chdir('..')
#         os.chdir('..')
#
# if __name__ == "__main__":
#     sumbit()
