
'''Specify input and output path'''
# how to use: inPath, outPath = inOutflilePath()

def inOutflilePath():
    flag = input('use the Default Path?')
    if not flag:
        inAddress = string(input('the Path of input file'))
        outAddress = string(input('the Path of output file'))
    print('inPath:' + inAddress)
    print('outPath:' + outAddress)
    # how to use: inPath, outPath = inOutflilePath()
    return inPath, outPath
