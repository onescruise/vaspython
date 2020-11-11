#!/usr/bin/env python
# coding=utf-8
import os
import tools
import shutil


def Dim(path):
    """
    Broad cell
    """
    os.chdir(path)

    fp = open('POSCAR', "r")
    file = fp.readlines()
    for i in range(6):
        print(file[i])
    print('!----------------------------------------------------------------------------------!')
    print('! Notes: make number_a X a = number_b X b = number_c X c equal as much as possible !')
    print('!----------------------------------------------------------------------------------!')
    print("Please input number_{a,b,c} [int type]")
    number_a = int(input('number_a = '))
    number_b = int(input('number_b = '))
    number_c = int(input('number_c = '))
    print("------------------------------------------")
    print('number_a = %s number_b = %s number_c = %s' %
          (number_a, number_b, number_c))
    os.system('''phonopy -d --dim="%d %d %d" --tolerance=1e-3''' %
              (number_a, number_b, number_c))

    f = open('Readme', 'a')
    f.write('\n')
    f.write('''phonopy -d --dim="%d %d %d"''' %
            (number_a, number_b, number_c))
    f.write('\n')
    f.close()


def CreatePOSCAR_number_File(path):
    """
    Create disp-{number} files.
    """
    import shutil
    os.chdir(path)

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
            tools.Readme("Creating folders of POSCAR-"
                         + order + " is finished!")


def CSF():
    '''
    Calculation of Sets of forces
    '''

    files = os.listdir(os.getcwd())

    number_dispFile = 0
    for filename in files:
        if (filename.find('disp-') != -1):
            number_dispFile += 1
    if(number_dispFile < 99):
        str_number_dispFile = '0' + str(number_dispFile)
    else:
        str_number_dispFile = str(number_dispFile)

    os.system('''phonopy -f disp-{001..'''
              + str_number_dispFile + '''}/vasprun.xml''')
    tools.Readme('phoCSF: Calculation of Sets of forces!')




if __name__ == "__main__":

    path = input('请输入声子计算所需POSCAR所在的文件夹路径：\n')
    shutil.copy2('submit_pho.py', path)
    Dim(path)
    CreatePOSCAR_number_File(path)
    CSF()
