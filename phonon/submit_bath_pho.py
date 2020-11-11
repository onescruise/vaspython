#!/usr/bin/python
# coding=utf-8
import time
import shutil
import os
"""
note：写入提交时的备注
filelist：对应所需提交的当前目录下的文件
dispNumber：对应与filelist文件中的disp-*的文件数

author: Yilin ZHANG
lasted time: Jan, 8th, 2020

填写下面提交信息：
"""

filelist = ['minus3', 'minus2', 'minus1', 'plus1', 'plus2', 'plus3',  'plus4']
note = input('# note:')
dispNumber = int(input('dispNumber:'))
KPOINTS = False


def sumbit():
    def getDispList(dispNumber):
        dispList = []
        for i in range(1, dispNumber + 1):
            if i < 10:
                number = str('00') + str(i)
            elif i < 100:
                number = str('0') + str(i)
            else:
                number = str(i)
            dispList.append('disp-' + number)
        return dispList

    for folder in filelist:
        os.chdir(folder)
        for disp in getDispList(dispNumber):
            shutil.copy2('../' + 'INCAR', disp)
            shutil.copy2('../' + 'POTCAR', disp)
            shutil.copy2('../' + 'vaspwz.pbs', disp)
            if KPOINTS:
                shutil.copy2('../' + 'KPOINTS', disp)
            os.chdir(disp)
            try:
                os.system('qsub vaspwz.pbs')
            except IOError:
                print(disp + ' Submit failed')
            os.chdir('..')
        os.chdir('..')


def writeReadme(note):
    f = open('readme', 'w')
    f.write('[' + time.asctime(time.localtime(time.time())) + ']\n')
    f.write('* ' + note + '\n')
    f.close()


if __name__ == "__main__":
    flag = input(
        'INCAR,POTCAR,pbs file (if KOPINTS,KPOINTS = True)?[Y/N]')
    print(flag)
    if flag == 'Y' or 'y':
        sumbit()
        writeReadme(note)
