#!/usr/bin/python
# coding=utf-8
import time
import os
import shutil

note = input('# NOTE: ')

KPOINTS = False

def submit(fileslist):
    for file in fileslist:
        shutil.copy2('INCAR', file)
        shutil.copy2('POTCAR', file)
        shutil.copy2('vaspwz.pbs', file)
        if KPOINTS:
            shutil.copy2('KPOINTS', file)

        os.chdir(file)
        try:
            os.system('qsub vaspwz.pbs')
        except IOError:
            print(file + ' Submit failed')
        os.chdir('..')
    os.system('qstat')


def getFileList(filenumber):

    fileslist = []
    for i in range(1, filenumber + 1):
        if i < 10:
            number = str('00') + str(i)
        elif i < 100:
            number = str('0') + str(i)
        else:
            number = str(i)
        fileslist.append('disp-' + number)
    return fileslist


def writeReadme(note):
    f = open('readme', 'a')
    f.write('[' + time.asctime(time.localtime(time.time())) + ']\n')
    f.write('* ' + note + '\n')
    f.close()


if __name__ == "__main__":
    print(note+'\n')
    flag=input('请把INCAR，POTCAR，pbs文件(可能需要的KOPINTS,即KPOINTS = True)与本提交脚本放置在同一目录下？[Y/N]')
    print(flag)
    if flag == 'Y' or 'y':
        filenumber = int(input('请输入disp-的文件数: '))
        submit(getFileList(filenumber))
        writeReadme(note)
