# coding=UTF-8
import os
README = False
KSPACING = True


def FindFile(file, dir=os.getcwd()):
    """
    Determine if the 'file' is in the 'dir'.

    Arguments:
        file:
        dir:

    return：
        Ture: find 'file' in 'dir'
        False: not find 'file' in 'dir'

    """
    files = os.listdir(dir)
    if file in files:
        return True
    else:
        print("!Not find " + file + " file! Please check it!")
        return False


def _getPOSCAR():
    """
    get the cell's name.

    return:
        cellname
    """
    file = open("POSCAR", "r")
    lines = file.readlines()
    line5 = str(lines[5]).strip().split()
    line6 = str(lines[6]).strip().split()
    cellname = ''
    for i in range(len(line5)):
        cellname += (line5[i] + line6[i])
    return cellname


def _CopyWriteDiary(copyfile, diaryname):
    diary = open(diaryname, "a")
    if(FindFile(copyfile)):
        diary.write('###' + " " + copyfile + '\n')
        diary.write('```\n')
        fp = open(copyfile, 'r')
        for line in fp:
            diary.write(line)
        diary.write('\n```\n\n')
    diary.close()


def WriteDiary():
    """
    Record the files and parameters used in the calculation,including INCAR POTCAR POSCAR KPOINTS and Readme.
    """
    from datetime import datetime

    diaryname = _getPOSCAR()
    diary = open(diaryname, "w")
    diary.write('***' + str(datetime.now()) + '***' + '\n')
    diary.write('## ' + diaryname + '\n')
    diary.close()
    if README:
        _CopyWriteDiary('Readme', diaryname)
    _CopyWriteDiary('INCAR', diaryname)
    if not KSPACING:
        _CopyWriteDiary('KPOINTS', diaryname)
    _CopyWriteDiary('POSCAR', diaryname)
    _CopyWriteDiary('POTCAR', diaryname)
    os.rename(diaryname, diaryname + '.md')


if __name__ == "__main__":
    WriteDiary()
