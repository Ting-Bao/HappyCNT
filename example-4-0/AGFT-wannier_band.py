'''
Benshu Fan 2021/5/31
Wannier band of a material
Description: This python code is designed for construction the Hamiltonian
                and plot the band structure use the vasp wannier90_hr.dat file.
              ::Input File::
                - wannier90_hr.dat
                - KPOINTS
                - POSCAR
              ::Output File::
                - wannier band.png
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator  # 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔


class WannierBand():
    def __init__(self, lines, num_wan, nrpts, n, name, lv, K_point_path, K_label, kn, E_fermi, ymin, ymax):
        self.lines = lines
        self.num_wan = num_wan
        self.nrpts = nrpts
        self.n = n
        self.name = name
        self.lv = lv
        self.K_point_path = K_point_path
        self.K_label = K_label
        self.kn = kn
        self.E_fermi = E_fermi
        self.ymin = ymin
        self.ymax = ymax

    # 根据实空间基矢获取倒格矢
    def reciprocal(self):
        V = np.dot(self.lv[0], np.cross(self.lv[1], self.lv[2]))
        self.rec = [np.cross(self.lv[1], self.lv[2]) * 2 * np.pi / V,
                    np.cross(self.lv[2], self.lv[0]) * 2 * np.pi / V,
                    np.cross(self.lv[0], self.lv[1]) * 2 * np.pi / V]

    # 构建K空间能带作图路径
    # 这里K空间路径越长，撒点越多，能带图越真实，注意endpoint=False的使用，避免重复计数与赋值
    def k_path(self):
        self.k = np.zeros((self.n * self.kn, 3), dtype=np.float16)
        for i in range(len(self.K_point_path)):
            self.K_point_path[i] = self.K_point_path[i][0] * self.rec[0] + self.K_point_path[i][1] * self.rec[1] + \
                                   self.K_point_path[i][2] * self.rec[2]

        for i in range(self.n):
            if i < self.n - 1:
                self.k[i * self.kn:(i + 1) * self.kn, ...] = np.linspace(self.K_point_path[i], self.K_point_path[i + 1],
                                                                         self.kn, endpoint=False)
            else:
                self.k[i * self.kn:(i + 1) * self.kn, ...] = np.linspace(self.K_point_path[i], self.K_point_path[i + 1],
                                                                         self.kn)

    # 依照K空间路径构建作图步长
    def length(self):
        k_length = [0]
        for i in range(self.n * self.kn - 1):
            if i < (self.n - 1) * self.kn:
                k_length.append(np.sqrt(
                    ((self.K_point_path[int(i / self.kn) + 1] - self.K_point_path[int(i / self.kn)]) ** 2).sum(
                        axis=-1)) / self.kn + k_length[i])
            else:
                k_length.append(np.sqrt(
                    ((self.K_point_path[int(i / self.kn) + 1] - self.K_point_path[int(i / self.kn)]) ** 2).sum(
                        axis=-1)) / (self.kn - 1) + k_length[i])
        self.k_length = k_length

        '''
        构建哈密顿量矩阵元，将代码矢量化，第一个维度是最后作图时k点个数，然后是wannier_hr.dat文件里的实空间R的个数，
        在BP里即为15*13=195，后面两个维度是实际哈密顿矩阵维度，在BP里即为16*16，而对于R和K，又多加了一个维度1，其实是为了后续与
        h的乘法唯独匹配，不具有实际物理意义，（1，3）表示的就是实空间和K空间的矢量维度，一行三列（三分量）
        '''

    def matrix_element(self):
        h = np.zeros((self.n * self.kn, self.nrpts, self.num_wan, self.num_wan), dtype=np.complex64)
        R = np.zeros((self.n * self.kn, self.nrpts, 1, 1, 3), dtype=np.float16)
        Degen = np.zeros((self.n * self.kn, self.nrpts, 1, 1), dtype=np.uint8)
        wan_centre = np.zeros((self.n * self.kn, self.nrpts, self.num_wan, self.num_wan, 3), dtype=np.float16)
        # 读取the degeneracy of each Wigner-Seitz grid point
        for ir in range(3, 4 + (self.nrpts - 1) // 15):
            if ir != 3 + (self.nrpts - 1) // 15:
                for jr in range(15):
                    Degen[:, (ir - 3) * 15 + jr, :] = self.lines[ir].split()[jr]
            else:
                for jr in range(1 + (self.nrpts - 1) % 15):
                    Degen[:, (ir - 3) * 15 + jr, :] = self.lines[ir].split()[jr]
        # 读取wannier90_centres.xyz文件 采用AGFT
        with open("wannier90_centres.xyz", "r") as fw:
            lines = fw.readlines()
            for i in range(self.num_wan):
                for j in range(self.num_wan):
                    wan_centre[..., i, j, :] = np.array(
                        list(map(float, lines[i + 2].strip().split()[1:4]))) - np.array(
                        list(map(float, lines[j + 2].strip().split()[1:4])))
            fw.close()
        if self.nrpts % 15 == 0:
            x = self.nrpts // 15 + 3
        else:
            x = self.nrpts // 15 + 4
        for i in range(x, len(self.lines)):
            h[:, int(np.floor((i - x) / self.num_wan ** 2)), int(self.lines[i].split()[3]) - 1,
            int(self.lines[i].split()[4]) - 1] = \
                float(self.lines[i].split()[5]) + 1j * float(self.lines[i].split()[6])
        for m in range(self.nrpts):
            R[:, m, ...] = float(self.lines[x + m * (self.num_wan ** 2)].split()[0]) * np.array(self.lv[0]) + float(
                self.lines[x + m * (self.num_wan ** 2)].split()[1]) * np.array(self.lv[1]) + float(
                self.lines[x + m * (self.num_wan ** 2)].split()[2]) * np.array(self.lv[2])
        H = (np.exp(1j * (R * self.k[:, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * (wan_centre * self.k[:, None, None, None, :]).sum(axis=-1)) * h) / Degen).sum(
            axis=1)
        self.H = H

    def plot(self):
        eigenvalue, _ = np.linalg.eigh(self.H)
        self.eigenvalue_k = np.sort(eigenvalue) - self.E_fermi
        dim = self.eigenvalue_k.shape[-1]
        fig, ax = plt.subplots()
        for dim0 in range(dim):
            plt.plot(self.k_length[0:self.n * self.kn], self.eigenvalue_k[:, dim0])

        # 注意最后一个坐标位置
        xticks = []
        for i in range(self.n + 1):
            if i != self.n:
                xticks.append(self.k_length[i * self.kn])
            else:
                xticks.append(self.k_length[i * self.kn - 1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(self.K_label)
        ax.set_title('Wannier band of {}'.format(self.name))
        ax.set_xlabel("Wave vector  "r"$\vec{k}$")
        ax.set_ylabel(r"$E - E_{fermi}$"' (eV)')
        plt.xlim([0, self.k_length[self.n * self.kn - 1]])
        plt.ylim([self.ymin, self.ymax])
        # y_major_locator = MultipleLocator(1.25)  # 把y轴的刻度间隔设置为10，并存在变量里
        # ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数

        plt.plot([0, self.k_length[self.n * self.kn - 1]], [0, 0], color='black', linestyle='--')
        plt.grid(axis='x', c='r', linestyle='--')
        plt.savefig('wannier band of {}.jpg'.format(self.name), bbox_inches='tight', dpi=600, pad_inches=0.0)  # bbox…去掉图外边框
        plt.show()


def main():
    begin = time.time()
    # 读取wannier90_hr.dat文件
    with open("wannier90_hr.dat", "r") as fw:
        lines = fw.readlines()
        # 获取投影的wannier band
        num_wan = int(lines[1].strip().split()[0])
        # 获取the number of Wigner-Seitz grid-points
        nrpts = int(lines[2].strip().split()[0])
        fw.close()
    # 根据POSCAR，获取实空间基矢
    with open("POSCAR", "r") as fp:
        lines_p = fp.readlines()
        # 获取体系名称
        name = lines_p[0].strip()
        lv = []
        for i in range(2, 5):
            lv.append((np.array(list(map(float, lines_p[i].strip().split()))) * float(
                lines_p[1].strip().split()[0])).tolist())
        fp.close()
    # 根据KPOINTS，获取能带在K空间高对称点所取路径
    with open("KPOINTS", "r") as fk:
        K_point_path = []
        K_label = []
        lines_k = fk.readlines()
        K_list = []
        for i in range(len(lines_k)):
            if len(lines_k[i].strip().split()) == 0:
                continue
            else:
                K_list.append(lines_k[i].strip().split())
        fk.close()
    for i in range(4, len(K_list), 2):
        if i != len(K_list) - 2:
            K_point_path.append(list(map(float, K_list[i][0:-1])))
            K_label.append(K_list[i].pop(-1))
        else:
            K_point_path.append(list(map(float, K_list[i][0:-1])))
            K_label.append(K_list[i].pop(-1))
            K_point_path.append(list(map(float, K_list[i + 1][0:-1])))
            K_label.append(K_list[i + 1].pop(-1))
    for i in range(len(K_label)):
        if K_label[i] == 'GAMMA' or K_label[i] == 'G':
            K_label[i] = '$\Gamma$'

    # 获取高对称线个数
    n = len(K_point_path) - 1
    # 每个高对称线撒点个数
    kn = 100
    # 从自洽步获取费米能，grep fermi OUTCAR
    E_fermi = -3.97520335
    ymin = -5
    ymax = 5
    kernel = WannierBand(lines, num_wan, nrpts, n, '4-0', lv, K_point_path, K_label, kn, E_fermi, ymin, ymax)
    kernel.reciprocal()
    kernel.k_path()
    kernel.length()
    kernel.matrix_element()
    print('time:', time.time() - begin)
    kernel.plot()


if __name__ == '__main__':  # 如果是当前文件直接运行，执行main()函数中的内容；如果是import当前文件，则不执行。
    main()
