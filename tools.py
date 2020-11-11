# coding=UTF-8
import os


def Readme(string):
    '''
    Write the operation to 'Readme'
    '''
    import os
    import time
    f = open('Readme', 'a')
    f.write('\n')
    f.write(string)
    f.write('  [' + time.asctime(time.localtime(time.time())) + ']')
    f.write('\n')
    f.close()


def YesNo(flag):
    while(True):
        if(flag == 'y'):
            return True
        elif(flag == 'n'):
            return False
        else:
            flag = input("!!!Please input[y/n] ")


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


def FindFileBySuffix(dir, suffix):
    files = os.listdir(dir)
    for i in range(len(files)):
        files[i] = os.path.splitext(files[i])[1]
    if suffix in files:
        print("!Find " + suffix + " file!")
        return True
    else:
        print("!Not find " + suffix + " file! Please check it!")
        return False


def RenameFileBySuffix(dir, suffix, newname):
    files = os.listdir(dir)
    for i in range(len(files)):
        if suffix in os.path.splitext(files[i])[1]:
            os.rename(dir + os.sep + files[i], dir + os.sep + newname)
            print("!Successfully " + suffix
                  + " file  was modified" + "to" + newname)
            return 0
    print("!Not find " + suffix + " file! Please check it!")


def PrintDir(path=os.getcwd()):
    files = os.listdir(path)
    print(files)


def PrintFile(file):
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        print(line)


def Rebackdir(path):
    os.chdir(path)
    path = os.getcwd()
    print(str(path))


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
    _CopyWriteDiary('Readme', diaryname)
    _CopyWriteDiary('INCAR', diaryname)
    _CopyWriteDiary('KPOINTS', diaryname)
    _CopyWriteDiary('POSCAR', diaryname)
    _CopyWriteDiary('POTCAR', diaryname)
    os.rename(diaryname, diaryname + '.md')


def CreateProject(projectName='project'):
    """
    create the project mkdir

    Arguments:
        projectName: is the name of project
    """
    projectName = input('''The project's name: ''')
    if not os.path.exists(projectName):
        os.mkdir(projectName)
    else:
        print('There is a file with the same name.')

    for dir in ['OPT', 'SCF', 'PHO']:
        if not os.path.exists(projectName + os.sep + dir):
            os.mkdir(projectName + os.sep + dir)


def ReplacementText(file, oldstr, newstr):
    import re
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    f = open(file, 'w')
    for line in lines:
        everyline = re.sub(oldstr, newstr, line)
        f.writelines(everyline)
    f.close()
    return True


def ReplacementLine(file, line_number, newstr):

    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    lines[line_number] = str(newstr) + '\n'

    f = open(file, 'w')
    for line in lines:
        f.writelines(line)
    f.close()


def getScalingFactor(dir=os.getcwd()):
    '''
    need POSCAR file
    '''
    f = open(dir, 'r')
    lines = f.readlines()
    f.close()
    # print(type(float(lines[1].strip()))) #test
    return float(lines[1].strip())


def Submit(folderHeadMark):
    files = os.listdir(os.getcwd())
    for filename in files:
        if filename.find(folderHeadMark) != -1:
            os.chdir(filename)
            try:
                os.system('qsub vaspwz.pbs')
            except IOError:
                print(filename + ' Submit failed')
            os.chdir('..')


if __name__ == "__main__":
    delta_SF = -(1 + 1) * 0.03
    # ReplacementLine('PrepareFiles/POSCAR', 1, delta_SF)
    getScalingFactor()
