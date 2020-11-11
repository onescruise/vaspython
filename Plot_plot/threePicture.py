import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rc('font', family='Times New Roman')
# from matplotlib.pyplot import subplot
# matplotlib.use('Agg')

config = {
    "font.family": 'Times New Roman',
    # "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def A_BCD_Scatter(name, data, save=False, outpath=None, filename=None, show=False):
    '''
    画A-B,A-C,A-D图
    '''
    # 数据np化
    data = np.array(data)

    plt.figure(figsize=(25, 6), dpi=600)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    # 第一幅图
    for i in range(3):
        ax.scatter(data[:, 0], data[:, i + 1])
        ax.set_ylabel(name[i + 1])
        ax.set_xlabel(name[0])
        # ax.legend(fontsize=10)

    plt.tick_params(labelsize='18')
    if show:
        plt.show()
    if save:
        plt.savefig(outpath + '/' + filename)
    return plt
