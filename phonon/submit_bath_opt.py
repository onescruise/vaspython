#!/usr/bin/python
# coding=utf-8
note=input('# Note:')
fileslist=['minus3', 'minus2', 'minus1', 'plus1', 'plus2', 'plus3', 'plus4']

import time
import os
import shutil

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

def writeReadme(note):
    f = open('readme', 'w')
    f.write('[' + time.asctime(time.localtime(time.time())) + ']\n')
    f.write('* ' + note + '\n')
    f.close()

if __name__ == "__main__":
    submit(fileslist)
    writeReadme(note)
