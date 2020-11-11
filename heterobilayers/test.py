import os
os.system('ase convert *.cif POSCAR')

with open('POSCAR', 'r') as f:
    lines = f.readlines()

with open('POSCAR', 'w') as f:
    for i in range(len(lines)):
        f.write(lines[i])
        if i == 4:
            f.write(lines[0])
