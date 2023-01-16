'''
Benshu Fan 2021/5/31
Ting Bao 2022/11/22
Wannier band of a material
Description: This python code is designed for construction the Hamiltonian
                and plot the band structure use the vasp wannier90_hr.dat file.
             Additionally, the code can be used to deal with the 1D system adding magnetic flux
                and get the Hamiltonian matrix based on kz and phi
              ::Input File::
                - wannier90_hr.dat
                - wannier90_centres.xyz
                - KPOINTS
                - POSCAR
              ::Output File::
                - wannier band.png
'''

#%%
import scipy.constants as cst
import time
import numpy as np
import scipy
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator  # 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔


class WannierBand():
    def __init__(self,  name, E_fermi, ymin, ymax, source, CNT_r = None):
        self.E_fermi = E_fermi  # 费米能
        self.ymin = ymin  # y轴坐标下限
        self.ymax = ymax
        self.name = name
        self.source = source
        self.CNT_r=CNT_r  # CNT的半径，用于计算加入磁通时候的截面面积, Unit:1 Ang = 1e-10 m
        self.get_parameters()
        self.reciprocal()

    def get_parameters(self):
        '''
        read parameters from files 
        '''
        # 读取wannier90_hr.dat文件
        with open(self.source+"wannier90_hr.dat", "r") as fw:
            lines = fw.readlines()
            # 获取投影的wannier band
            num_wan = int(lines[1].strip().split()[0])
            # 获取the number of Wigner-Seitz grid-points
            nrpts = int(lines[2].strip().split()[0])

        # 根据POSCAR，获取实空间基矢
        with open(self.source+"POSCAR", "r") as fp:
            lines_p = fp.readlines()
            # 获取体系名称
            #name = lines_p[0].strip()
            lv = []
            for i in range(2, 5):
                lv.append((np.array(list(map(float, lines_p[i].strip().split()))) * float(
                    lines_p[1].strip().split()[0])).tolist())

        # 根据KPOINTS，获取能带在K空间高对称点所取路径
        with open(self.source+"KPOINTS", "r") as fk:
            K_point_path = []
            K_label = []
            lines_k = fk.readlines()
            K_list = []
            for i in range(len(lines_k)):#this is by fbs
                if len(lines_k[i].strip().split()) == 0:
                    continue
                else:
                    K_list.append(lines_k[i].strip().split())
            kn=int(lines_k[1].strip()) # 每个高对称线撒点个数
            # print(kn,type(kn))

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

        self.lv = lv  # lattice vector
        self.K_point_path = K_point_path   #高对称点坐标的列表
        self.K_label = K_label  #高对称点符号的列表
        self.kn = kn  #每段高对称路径的撒点数量
        self.lines = lines   # wannier90_hr.dat每行
        self.num_wan = num_wan    # wannier带的数目
        self.nrpts = nrpts   #实空间截断矩阵的个数
        self.n = n    # 高对称线的个数（段数），例如G-Z-A算作两段

    # 根据实空间基矢获取倒格矢
    def reciprocal(self):
        '''
        get the reciprocal lattice vector
        '''
        V = np.dot(self.lv[0], np.cross(self.lv[1], self.lv[2]))
        self.rec = [np.cross(self.lv[1], self.lv[2]) * 2 * np.pi / V,
                    np.cross(self.lv[2], self.lv[0]) * 2 * np.pi / V,
                    np.cross(self.lv[0], self.lv[1]) * 2 * np.pi / V]

    # 构建K空间能带作图路径
    # 这里K空间路径越长，撒点越多，能带图越真实，注意endpoint=False的使用，避免重复计数与赋值
    def k_path(self): #self.k 指的是kpath的路径
        '''
        get the kpath for band calculation
        '''
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

    # 依照K空间路径构建作图步长，“步长”指的是沿着k空间高对称路径走过的长度
    def band_length(self):
        '''
        get the step length of kpath
        '''
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
        '''
        generate the matrix and do the Fourier transformation
        finally get the H(k)
        '''
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
        with open(self.source+"wannier90_centres.xyz", "r") as fw:
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
        '''
        plot the band along k path
        '''
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
        plt.savefig(self.source+'AGFT wannier band of {}.jpg'.format(self.name), bbox_inches='tight', dpi=600, pad_inches=0.2)  # bbox…去掉图外边框
        #plt.show()

    def mag_matrix_element(self, kz = 0.0, coo='fractional'):
        '''
        take (0,0,kz) in in the k space, default kz is in the fractional coo
        '''
        if coo=='cartesian':
            kz_car=kz
        elif coo=='fractional':
            kz_car=kz*self.rec[-1][-1]
        else:
            raise AttributeError()

        kz_car=np.array([0,0,kz_car])

        ###
        self.B = np.array(np.linspace(0, 4*self.period_B, 3000),dtype=np.float64)
        self.B_reduce=self.B / (cst.h / (2*cst.e))
        ###
        #               k路径撒点数量      实空间矩阵数   wannier基数   wannier基数
        h = np.zeros((self.n * self.kn, self.nrpts, self.num_wan, self.num_wan), dtype=np.complex128)
        R = np.zeros((self.n * self.kn, self.nrpts, 1, 1, 3), dtype=np.float32) 
        Degen = np.zeros((self.n * self.kn, self.nrpts, 1, 1), dtype=np.int8) # 简并程度
        wan_centre = np.zeros((self.n * self.kn, self.nrpts, self.num_wan, self.num_wan, 3), dtype=np.float32)
        S = np.zeros((self.num_wan, self.num_wan), dtype=np.float32)
        # 读取the degeneracy of each Wigner-Seitz grid point
        print(self.nrpts)

        # mod 15 是因为记录wannier90_hr.dat中的第四行开始的记录简并度的行，一行最多有15个数字
        for ir in range(3, 4 + (self.nrpts - 1) // 15):   # ir 判断了写简并度的行数有几行
            if ir != 3 + (self.nrpts - 1) // 15:
                for jr in range(15):
                    Degen[:, (ir - 3) * 15 + jr, :] = self.lines[ir].split()[jr]
            else:
                for jr in range(1 + (self.nrpts - 1) % 15):
                    Degen[:, (ir - 3) * 15 + jr, :] = self.lines[ir].split()[jr]
        
        # 读取wannier90_centres.xyz文件 采用AGFT
        with open(self.source+"wannier90_centres.xyz", "r") as fw:
            lines = fw.readlines()#self.lines 和 lines 是两个变量。。。
            for i in range(self.num_wan):
                for j in range(self.num_wan):
                    R1 = np.array(list(map(float, lines[i + 2].strip().split()[1:4])))
                    R2 = np.array(list(map(float, lines[j + 2].strip().split()[1:4])))
                    wan_centre[..., i, j, :] = (R1 - R2) # 把一个三维矢量，赋给了一个三阶张量                                      
                    R1[-1] = 0.0
                    R2[-1] = 0.0
                    sgn = np.sign(np.cross(R1, R2)[-1])

                    R_avg=self.CNT_r
                    theta = np.arccos(round(np.dot(R1, R2) / (np.linalg.norm(R1) * np.linalg.norm(R2)), 8))
                    S[i, j] = sgn * theta * ((1e-10 * R_avg) ** 2) / 2 #use meters as unit, adjust for the sign problem
                    # 记录ij间考虑符号的面积

        # x 用于判断wannier90_hr.dat中第几行开始有矩阵元
        if self.nrpts % 15 == 0:
            x = self.nrpts // 15 + 3
        else:
            x = self.nrpts // 15 + 4
        
        # h 为实空间的哈密顿量矩阵
        for i in range(x, len(self.lines)):
            h[:, int(np.floor((i - x) / self.num_wan ** 2)), int(self.lines[i].split()[3]) - 1,
            int(self.lines[i].split()[4]) - 1] = \
                float(self.lines[i].split()[5]) + 1j * float(self.lines[i].split()[6])

        # 每个实空间矩阵的对应的晶格矢量    
        for m in range(self.nrpts):
            R[:, m, ...] = float(self.lines[x + m * (self.num_wan ** 2)].split()[0]) * np.array(self.lv[0]) + float(
                self.lines[x + m * (self.num_wan ** 2)].split()[1]) * np.array(self.lv[1]) + float(
                self.lines[x + m * (self.num_wan ** 2)].split()[2]) * np.array(self.lv[2])
        '''
        H = (np.exp(1j * (R[None, -1, ...] * self.k[None, -1, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * (wan_centre[None, -1, ...] * self.k[None, -1, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * S[None, None, None, ...] * self.B[:, None, None, None, None])) * h[None, -1, ...]) / Degen[
                 None, -1, ...]).sum(axis=2)
        '''
        H = (np.exp(1j * (R[None, -1, ...] * kz_car[None, None, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * (wan_centre[None, -1, ...] * kz_car[None, None, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * S[None, None, None, ...] * self.B_reduce[:, None, None, None, None])) * h[None, -1, ...]) / Degen[
                 None, -1, ...]).sum(axis=2)

        H = H.sum(axis=1)
        
        self.mag_H = H
        self.mag_H_kz=kz_car[-1]


    def mag_plot(self):
        '''
        plot the band structure along mag flux phi at fix k point
        '''
        eigenvalue, _ = np.linalg.eigh(self.mag_H)
        self.eigenvalue_k = np.sort(eigenvalue) - self.E_fermi
        dim = self.eigenvalue_k.shape[-1]
        fig, ax = plt.subplots()
        for dim0 in range(dim):
            plt.plot(self.B_reduce, self.eigenvalue_k[:, dim0])

        ax.set_title('Wannier band of {} vs mag flux'.format(self.name))
        ax.set_xlabel(r"magnetic strength $B$")
        ax.set_ylabel(r"$E - E_{fermi}$"' (eV)')
        plt.ylim([self.ymin, self.ymax])
        plt.savefig(self.source+'Mag band of {} at kz= {:.2f}.png'.format(self.name,self.mag_H_kz), bbox_inches='tight', dpi=600, pad_inches=0.2)  # bbox…去掉图外边框
        #plt.show()
        plt.close()

    def get_boundary(self):
        '''
        Get the peoriod of kz and mag strength B (and mag flux phi)
        '''
        #self.CNT_r
        self.period_kz = self.rec[-1][-1]
        unitphi= 2* cst.pi * cst.h / ( 2 * cst.e)
        self.period_phi = unitphi
        self.period_B = unitphi/(cst.pi*(self.CNT_r*1e-10)**2)
        print("period kz and B is {} and {} respectively.".format(self.period_kz,self.period_B))
    
    def get_full_matrix(self, kz_mesh=100, phi_mesh=100):
        '''
        Get the full matrix H(kz,phi)
        kz_mesh, phi_mesh defines the number of mesh grids.
        '''

        self.kz_mesh = kz_mesh
        self.phi_mesh = phi_mesh
        if not hasattr(self, 'period_kz'):
            self.get_boundary()

        ### mesh
        self.B = np.array(np.linspace(0, self.period_B, phi_mesh),dtype=np.float64) # 磁场强度，单位特斯拉
        self.B_reduce=self.B / (cst.h / (2*cst.e))
        self.kz = np.array([[0,0,i]for i in np.array(np.linspace(0, self.period_kz, kz_mesh),dtype=np.float64)])
        ###

        #               k路径撒点数量      实空间矩阵数   wannier基数   wannier基数
        h = np.zeros((self.n * self.kn, self.nrpts, self.num_wan, self.num_wan), dtype=np.complex128)
        R = np.zeros((self.n * self.kn, self.nrpts, 1, 1, 3), dtype=np.float32) 
        Degen = np.zeros((self.n * self.kn, self.nrpts, 1, 1), dtype=np.int8) # 简并程度
        wan_centre = np.zeros((self.n * self.kn, self.nrpts, self.num_wan, self.num_wan, 3), dtype=np.float32)
        S = np.zeros((self.num_wan, self.num_wan), dtype=np.float32)
        # 读取the degeneracy of each Wigner-Seitz grid point
        print(self.nrpts)

        # mod 15 是因为记录wannier90_hr.dat中的第四行开始的记录简并度的行，一行最多有15个数字
        for ir in range(3, 4 + (self.nrpts - 1) // 15):   # ir 判断了写简并度的行数有几行
            if ir != 3 + (self.nrpts - 1) // 15:
                for jr in range(15):
                    Degen[:, (ir - 3) * 15 + jr, :] = self.lines[ir].split()[jr]
            else:
                for jr in range(1 + (self.nrpts - 1) % 15):
                    Degen[:, (ir - 3) * 15 + jr, :] = self.lines[ir].split()[jr]
        
        # 读取wannier90_centres.xyz文件 采用AGFT
        with open(self.source+"wannier90_centres.xyz", "r") as fw:
            lines = fw.readlines()#self.lines 和 lines 是两个变量。。。
            for i in range(self.num_wan):
                for j in range(self.num_wan):
                    R1 = np.array(list(map(float, lines[i + 2].strip().split()[1:4])))
                    R2 = np.array(list(map(float, lines[j + 2].strip().split()[1:4])))
                    wan_centre[..., i, j, :] = (R1 - R2) # 把一个三维矢量，赋给了一个三阶张量                                      
                    R1[-1] = 0.0
                    R2[-1] = 0.0
                    sgn = np.sign(np.cross(R1, R2)[-1])

                    R_avg=self.CNT_r
                    theta = np.arccos(round(np.dot(R1, R2) / (np.linalg.norm(R1) * np.linalg.norm(R2)), 8))
                    S[i, j] = sgn * theta * ((1e-10 * R_avg) ** 2) / 2 #use meters as unit, adjust for the sign problem
                    # 记录ij间考虑符号的面积

        # x 用于判断wannier90_hr.dat中第几行开始有矩阵元
        if self.nrpts % 15 == 0:
            x = self.nrpts // 15 + 3
        else:
            x = self.nrpts // 15 + 4
        
        # h 为实空间的哈密顿量矩阵
        for i in range(x, len(self.lines)):
            h[:, int(np.floor((i - x) / self.num_wan ** 2)), int(self.lines[i].split()[3]) - 1,
            int(self.lines[i].split()[4]) - 1] = \
                float(self.lines[i].split()[5]) + 1j * float(self.lines[i].split()[6])

        # 每个实空间矩阵的对应的晶格矢量    
        for m in range(self.nrpts):
            R[:, m, ...] = float(self.lines[x + m * (self.num_wan ** 2)].split()[0]) * np.array(self.lv[0]) + float(
                self.lines[x + m * (self.num_wan ** 2)].split()[1]) * np.array(self.lv[1]) + float(
                self.lines[x + m * (self.num_wan ** 2)].split()[2]) * np.array(self.lv[2])
                
        '''
        对于下面的tensor，扩充维度后前两个维度axis是固定不动的phimesh和kzmesh
        R,h,wan_center中因为历史遗留问题被扩充到了band的k点数量，是不必要的，这一维度被指定0而去除
        '''

        # 以下都提早进行了过度广播，浪费内存，有空可以优化

        H1 = np.exp(1j * (R[None,None, 0, ...] * self.kz[None, :, None, None, None, None, :]).sum(axis=-1))
        # 广播扩维后shape(1,kmesh,1, 实空间矩阵数量nrpts，1 ，1， 3)， 最后一维对应三个方向被sum掉了。
        # H1.shape() = (1,kmesh,1, 实空间矩阵数量nrpts，1 ，1)
        
        H2 = np.exp(1j * (wan_centre[None,None, 0, ...] * self.kz[None, :,None, None, None, None, :]).sum(axis=-1)) \
        # 广播扩维后shape(1,kmesh,1, 实空间矩阵数量nrpts，num_wan ，num_wan， 3)， 最后一维对应三个方向被sum掉了。
        # H2.shape() = (1,kmesh,1, 实空间矩阵数量nrpts，num_wan ，num_wan)

        H3 = np.exp(1j * S[None, None, None, None, ...] * self.B_reduce[:, None, None, None, None, None]) * h[None,None, 0, ...]/ Degen[None, None, 0, ...]
        # 广播扩维后shape(phimesh,1,1, 实空间矩阵数量nrpts，num_wan ，num_wan)
        # H3.shape() = (phimesh,1,1, 实空间矩阵数量nrpts，num_wan ，num_wan)

        H=(H1*H2*H3).sum(axis=2)
        # 广播扩维后shape(phimesh, kmesh, 1, 实空间矩阵数量nrpts，num_wan ，num_wan)
        # 求和后 H.shape()= (phimesh, kmesh, 实空间矩阵数量nrpts，num_wan ，num_wan)

        del H1,H2,H3
        '''
        Htemp = (np.exp(1j * (R[None,None, -1, ...] * self.kz[:,None, None, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * (wan_centre[None,None, -1, ...] * self.kz[:, None,None, None, None, None, :]).sum(axis=-1)) * (np.exp(
            1j * S[None, None, None, None, ...] * self.B_reduce[None, :, None, None, None, None])) * h[None,None, -1, ...]) / Degen[
            None, None, -1, ...]).sum(axis=2)
        print(np.allclose(Htemp,H))
        '''
        H = H.sum(axis=2) # 关于实空间晶格矢求和
        
        print(np.allclose(H[:,0,...],H[:,12,...]))
        print(np.allclose(H[:,33,...],H[:,99,...]))
        print(np.allclose(H[0,...],H[1,...]))

        self.full_H = H

    def full_H_plot(self, show=False, plot_max=3, plot_min=-3):
        '''
        To avoid confused with mag_H part, run 'get_full_matrix' before this function
        plot the "band structure"
        this is a 3D contour
        '''
        eigenvalue, _ = np.linalg.eigh(self.full_H)
        self.eigenvalue_full = np.sort(eigenvalue) - self.E_fermi
        '''
        The eigenvalue_full is in the shape (phi_mesh, kz_mesh, num_wan)
        '''
        B,kz = np.meshgrid(self.B,self.kz[:,-1],indexing='ij') # self.kz is in shape (kzmesh, 3)
        temp=self.kz[:,-1]
        print (self.kz[:,-1])
        ## plot 3D fig and save
        fig = plt.figure()
        fig.set_size_inches(5, 6)
        ax = fig.add_subplot(projection='3d')

        # manually cut off the data out of range
        
        eigenvalue_full_cut=self.eigenvalue_full
        eigenvalue_full_cut[(eigenvalue_full_cut>plot_max)|(eigenvalue_full_cut<plot_min)]=None
        '''
        print(np.allclose(H[:,0,:],H[:,12,:]))
        print(np.allclose(H[:,33,:],H[:,99,:]))
        print(np.allclose(H[0,,:],H[1,,:]))
        '''
        for i in range(self.num_wan):
            im=ax.plot_surface(B, kz, eigenvalue_full_cut[...,i], cmap=cm.nipy_spectral, vmin= plot_min,vmax = plot_max,linewidth=0, antialiased=False)
        
        ax.set_zlim(plot_min, plot_max)
        fig.colorbar(im, location='left', extend='both', aspect=50, pad=0.01,shrink =0.5,label='Eigen Energy Ek')
        plt.xlabel(r'$B/T$')
        plt.ylabel(r'$k_z$')
        temp='temp'
        plt.title(temp)
        plt.savefig('./temp.jpg',dpi=800)
        if show==True:
            plt.show()
        input('hh')



def AGFT(source='./', name='example',E_fermi = 0.0, ymin = -5, ymax = 5):
    '''
    func AGFT reproduce the band structure from the Wannier model
    '''
    # 从自洽步获取费米能，grep fermi OUTCAR
    begin = time.time()
    kernel = WannierBand(name=name, E_fermi=E_fermi, ymin=ymin, ymax=ymax, source=source)
    kernel.k_path()
    kernel.band_length()
    kernel.matrix_element()
    print('time:', time.time() - begin)
    kernel.plot()

def mag_band(source='./', name='example',E_fermi = 0.0, ymin = -5, ymax = 5, CNT_r=1.0, kz = 0.0, coo='cartesian'):
    '''
    func mag_band reproduce the band structure at the specific kz point
    '''
    begin = time.time()
    kernel = WannierBand(name=name, E_fermi=E_fermi, ymin=ymin, ymax=ymax, source=source, CNT_r=CNT_r)
    kernel.get_boundary()
    kernel.mag_matrix_element(kz = kz, coo=coo)
    print('time:', time.time() - begin)
    kernel.mag_plot()

def full_hamiltonian(source='./', name='example',E_fermi = 0.0, ymin = -5, ymax = 5, CNT_r=1.0, to_file = False):
    '''
    get the full hamiltonian H(k, phi), take the full hamiltonian as the return
    '''
    kernel = WannierBand(name=name, E_fermi=E_fermi, ymin=ymin, ymax=ymax, source=source, CNT_r=CNT_r)
    kernel.get_full_matrix(kz_mesh=123)
    kernel.full_H_plot(show=True)

    if to_file==True:
        # https://blog.csdn.net/mr_songw/article/details/124222383
        np.save(source+'full_H',kernel.full_H)
    
    # read the stored full_H
    # full_H = np.load(source+'full_H.npy')




if __name__ == '__main__':  # 如果是当前文件直接运行，执行main()函数中的内容；如果是import当前文件，则不执行。
    # AGFT(source='work/example-3-3/',name='3-3',E_fermi=-2.4239)
    # mag_band(source='work/example-3-3/',name='3-3',E_fermi=-2.4239, CNT_r=2.03, kz = 0.5, coo='fractional')
    full_hamiltonian(source='work/example-3-3/',name='3-3',E_fermi=-2.4239, CNT_r=2.03)
