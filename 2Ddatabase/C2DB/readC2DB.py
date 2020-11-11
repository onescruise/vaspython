# import ase.db
# # from ase.io import write
# import os
import numpy as np
import pandas as pd
import os


def prehandle_c2db():
    import json
    import sqlite3
    # Step1： connect database and get the features
    con = sqlite3.connect("c2db.db")
    df = pd.read_sql_query("SELECT * FROM systems", con)

    # Step2: convert database to csv
    key_value_pairs = []
    for line in df.key_value_pairs:
        key_value_pairs.append(json.loads(line))
    key_value_pairs = pd.DataFrame(key_value_pairs)

    # Step3: rename formula
    formula_values = list(key_value_pairs['folder'].str.split('/'))
    flag = []
    for line in formula_values:
        flag.append(line[-2])
    key_value_pairs['formula'] = flag

    # Step4： sorted the vbm
    key_value_pairs['vbm_to_evac'] = key_value_pairs['vbm'] - key_value_pairs['evac']
    key_value_pairs['cbm_to_evac'] = key_value_pairs['cbm'] - key_value_pairs['evac']
    hasBandGap = key_value_pairs[key_value_pairs['vbm_to_evac'] < 0]
    hasBandGap = hasBandGap[hasBandGap['gap'] > 0.3]

    hasBandGap = hasBandGap[hasBandGap['ehull'] <= 0]
    hasBandGap = hasBandGap[hasBandGap['dynamic_stability_level'] == 3]
    hasBandGap = hasBandGap.sort_values(by=['vbm_to_evac'], axis=0)
    hasBandGap.to_csv('sorted_by_vbm_to_evac.csv')

    # Step5: band alignment
    count = 0
    bandAlignment_CSV_names = []
    duplicate = []

    for i in range(len(hasBandGap)):
        # 获取名称，方便建立文件
        flag = str(hasBandGap.iloc[i]['uid'])
        x = hasBandGap[hasBandGap['cbm_to_evac'] -
                       hasBandGap.iloc[i]['vbm_to_evac'] < 0.1]
        x = x[x['cbm_to_evac'] - hasBandGap.iloc[i]['vbm_to_evac'] > -0.1]
        x = x.sort_values(by=['cbm_to_evac'], axis=0)
        x = x[~x['uid'].isin(duplicate)]  # 删除重复项
        x = x.append(hasBandGap.iloc[i])
        if len(x) > 1:
            if 'Se' in flag:
                if count < 10:
                    folder = '0' + str(count) + '-' + flag
                else:
                    folder = str(count) + '-' + flag
                if not os.path.exists(folder):
                    os.mkdir(folder)
                bandAlignment_CSV_names.append(folder + '.csv')

                x.to_csv(folder + os.sep + folder + '.csv')
                count += 1
                duplicate.append(flag)

    # step6: write CSVs
    with open('bandAlignment_CSV_names', 'w') as f:
        for i in bandAlignment_CSV_names:
            f.write(str(i) + '\n')


def after_handle():
    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    # Step1: read bandAlignment_CSV_names
    with open('bandAlignment_CSV_names', 'r') as f:
        lines = f.readlines()
    # Step2：Visualization
    for i in range(len(lines)):
        os.chdir(lines[i].strip()[:-4])
        csv = pd.read_csv(lines[i].strip())
        labels = []
        vbms = []
        cbms = []

        # labels.append(csv.iloc[-1]['formula'])
        # vbms.append(csv.iloc[-1]['vbm_to_evac'])
        # cbms.append(csv.iloc[-1]['cbm_to_evac'])
        for j in range(-1, len(csv) - 1):
            labels.append(csv.iloc[j]['formula'])
            vbms.append(csv.iloc[j]['vbm_to_evac'])
            cbms.append(csv.iloc[j]['cbm_to_evac'])

        x = np.arange(len(vbms)) + 0.5
        emin = np.floor(np.min(vbms)) - 1.0

        # With and height in pixels
        ppi = 100
        figw = 800
        figh = 400

        fig = plt.figure(figsize=(figw / ppi, figh / ppi), dpi=ppi)
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(x, np.array(vbms) - emin, bottom=emin)
        ax.bar(x, -np.array(cbms), bottom=cbms)
        ax.set_xlim(0, len(labels))
        ax.set_ylim(emin, 0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=10)

        plt.title(lines[i], fontsize=12)
        plt.ylabel('Energy relative to vacuum [eV]', fontsize=10)
        plt.tight_layout()
        plt.savefig(lines[i].strip() + '.png')
        # plt.show()
        os.chdir('..')

################################################################################


def writePOSCAR(atom):

    with open(path + os.sep + filename + '_POSCAR', 'w') as f:
        f.write(self.parameters['SystemName'] + '\n')
        f.write(str(self.parameters['ScalingFactor']) + '\n')

        LatticeMatrix = self.parameters['LatticeMatrix']
        for i in range(len(LatticeMatrix)):
            f.write(float2line(LatticeMatrix[i]))

        AtomsInfo = self.parameters['AtomsInfo']
        string = ['', '']
        for key in AtomsInfo.keys():
            string[0] += key + '    '
            string[1] += str(AtomsInfo[key]) + '    '
        string[0] += '\n'
        string[1] += '\n'
        f.write(string[0])
        f.write(string[1])

        f.write(self.parameters['CoordinateType'] + '\n')

        ElementsPositionMatrix = self.parameters['ElementsPositionMatrix'].copy(
        )
        for key in ElementsPositionMatrix.keys():
            arr = ElementsPositionMatrix[key]
            for i in range(len(arr)):
                f.write(float2line(arr[i]))


def output_structure():
    import ase.db
    from ase.io import write
    dbpath = os.path.join(os.getcwd(), 'c2db.db')
    db = ase.db.connect(dbpath)

    def float2line(mart):
        '''
        将矩阵转换成可输出的字符串形式
        '''
        string = ''
        for line in mart:
            for s in line:
                string += '%21.10f' % float(s)
            string += '\n'
        return string

    def count_element(mart):
        string = ['', '']
        elements = sorted(set(mart), key=mart.index)
        for key in elements:
            string[0] += key + '    '
            string[1] += str(mart.count(key)) + '    '
        string[0] += '\n'
        string[1] += '\n'
        return string

    with open('bandAlignment_CSV_names', 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        os.chdir(lines[i].strip()[:-4])
        csv = pd.read_csv(lines[i].strip())
        for j in range(len(csv)):
            flag = (j + 1) % len(csv)
            if flag < 10:
                mark = str('0') + str(flag) + '-'
            else:
                mark = str(flag) + '-'

            atom = db.get_atoms(uid=csv.iloc[j]['uid'], add_additional_information=True)
            write(mark + '{}.cif'.format(csv.iloc[j]['formula']), atom, format='cif')
            with open(mark + csv.iloc[j]['formula'] + '.vasp', 'w') as f:
                f.write(csv.iloc[j]['uid'] + '\n')
                f.write('1.0\n')
                f.write(float2line(atom.get_cell()))
                atomInfo = count_element(atom.get_chemical_symbols())
                f.write(atomInfo[0])
                f.write(atomInfo[1])
                f.write('Cartesian\n')
                f.write(float2line(atom.get_positions()))
        os.chdir('..')


def heterojunction():
    from hetero import establish_hetero
    import ase.db
    dbpath = os.path.join(os.getcwd(), 'c2db.db')
    db = ase.db.connect(dbpath)

    with open('bandAlignment_CSV_names', 'r') as f:
        lines = f.readlines()
    for table in range(len(lines)):
        os.chdir(lines[table].strip()[:-4])
        csv = pd.read_csv(lines[table].strip())
        # atom0 = db.get_atoms(uid=csv.iloc[len(csv) - 1]['uid'], add_additional_information=True)
        # atom0_cell = np.array(atom0.get_cell())
        structure0 = '00-' + csv.iloc[len(csv) - 1]['formula'] + '.vasp'
        print(structure0)

        for i in range(len(csv) - 1):
            flag = (i + 1) % len(csv)
            if flag < 10:
                mark = str('0') + str(flag) + '-'
            else:
                mark = str(flag) + '-'

            # atom1 = db.get_atoms(uid=csv.iloc[i]['uid'], add_additional_information=True)
            structure1 = mark + csv.iloc[i]['formula'] + '.vasp'

            E = establish_hetero()
            dim = 5
            layer_dict = E.find_hetero(structure0, structure1, dim=dim, lattice_mismatch=0.04, angle_mismatch=1, angle_need=90)
            while layer_dict['parameter'] == []:
                j = 2
                layer_dict = E.find_hetero(structure0, structure1, dim=dim * j, lattice_mismatch=0.04, angle_mismatch=1, angle_need=90)
                j += 1
            file_name = E.build_hetero(lattice_select=-1, layer_spacing=3, vac_thickness=20, cp=0)
            if os.path.exists(file_name):
                os.rename(file_name, '00-' + mark + '.vasp')

        os.chdir('..')


if __name__ == "__main__":
    prehandle_c2db()
    after_handle()
    output_structure()
    heterojunction()
