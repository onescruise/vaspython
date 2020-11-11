
#!/usr/bin/python
# coding=utf-8
'''
Created on 11:35, May. 11, 2020

Recently updated on 15:37, May. 12, 2020

@author: Yilin Zhang

These files are need:
    *vasprun.xml* or *OUTCAR* (both after optics caculation)
'''


import os
import numpy as np


def inOutflilePath():
    '''Specify input and output path'''
    # how to use: inPath, outPath = inOutflilePath()

    flag = input('use the Default Path(.)?[1/0]')
    if flag:
        inPath = '.'
        outPath = '.'
    if not flag:
        inPath = str(input('the Path of input file'))
        outPath = str(input('the Path of output file'))
    print('inPath:' + inPath)
    print('outPath:' + outPath)
    # how to use: inPath, outPath = inOutflilePath()
    return inPath, outPath


def contextIn2markLine(lines, markStart, markEnd, show=False):
    '''get the context of lines between markStart and markEnd line'''
    # how to use: context = contextIn2markLine(lines, markStart='Direct', markEnd='', show=False)
    context = []
    indexStart = len(lines)
    # print(indexStart)
    indexEnd = -1
    for line in lines:
        indexEnd += 1
        if markStart in line:
            indexStart = indexEnd
        if len(markEnd.strip().split()) == 0:
            if len(line.strip().split()) == 0:
                if (indexEnd - indexStart) > 0:
                    break
        else:
            if markEnd in line:
                if (indexEnd - indexStart) > 0:
                    break

    print('indexStart:' + str(indexStart), 'indexEnd:' + str(indexEnd) + '\n')

    for i in range(indexStart + 1, indexEnd):
        context.append(lines[i].strip().split())
    if show:
        print('context between ' + markStart + ' and ' + markEnd + ':')
        print(len(context))
        for i in range(len(context)):
            print(context[i])

    return context


def float2line(line,):
    # print(line)
    string = ''
    for i in range(len(line)):
        string += '%25.20f' % float(line[i])
    string += '\n'
    return string


def getDielectricMatrix():
    print('Need files:')
    print('*vasprun.xml* or *OUCTAR*(after optics caculation)')

    inPath, outPath = inOutflilePath()

    infilename = 'OUTCAR'  # vasprun.xml
    infile = open(inPath + os.sep + infilename, 'r')
    lines = infile.readlines()
    infile.close()

    with open(outPath + os.sep + 'REAL.in', 'w') as f:
        context = contextIn2markLine(
            lines, markStart='REAL DIELECTRIC FUNCTION', markEnd='', show=False)

        context = context[2:]
        context = np.array(context).astype(np.float)

        print('REAL line:' + str(len(context)) + '\n')

        for i in range(len(context)):
            line = float2line(context[i])
            f.write(line)

    with open(outPath + os.sep + 'IMAG.in', 'w') as f:
        context = contextIn2markLine(
            lines, markStart='IMAGINARY DIELECTRIC FUNCTION', markEnd='', show=False)

        context = context[2:]
        context = np.array(context).astype(np.float)
        print('IMAG line:' + str(len(context)) + '\n')

        for i in range(len(context)):
            line = float2line(context[i])
            f.write(line)


if __name__ == "__main__":
    getDielectricMatrix()
