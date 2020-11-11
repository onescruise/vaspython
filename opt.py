#!/usr/bin/env python
# coding=utf-8
import os
import sys


def opt():
    """
    Script submission for structural optimization

    Arguments:
        dir: Required dir to submit
    """

    sys.path.append('/home/ylzhang/bin/vaspython')
    import tools

    flag = False
    while(not flag):
        flag=tools.YesNo(input('Files Ready?[y/n] '))
        flag = tools.FindFile('POSCAR')
        if flag:
            flag = tools.FindFile('INCAR')
        if flag:
            flag = tools.FindFile('POTCAR')
        if flag:
            flag = tools.FindFile('KPOINTS')
        if flag:
            flag = tools.FindFile('Readme')
        if flag:
            flag = tools.FindFile('vaspwz.pbs')
        if flag:
            break

    tools.WriteDiary()
    os.system('qsub vaspwz.pbs')
    os.system('qstat')
    exit()


if __name__ == '__main__':
    opt()
