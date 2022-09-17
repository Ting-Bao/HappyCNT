'''
Author: Benshu Fan
Date: 2021.7.27
Description: This python code is designed for plot the band structure
             (including DFT band, wannier band, spin up and down)
              ::Input File::
                - BAND.dat
                - KLABELS
                - wannier90_band.dat
              ::Output File::
                - DFT band.png (spin up and spin down)
                - Wannier band.png
                - Fitting band.png
'''

'''
This file is silghtly modified by Ting Bao for simplicity
'''


import matplotlib.pyplot as plt


class DFTBand():
    def Klabels(self):
        # 获得高对称点信息：hsk[]是高对称点符号列表，hsk_coors=[]是高对称点坐标列表
        self.hsk = []
        self.hsk_coors = []
        with open('KLABELS') as frp:
            lines = frp.readlines()
        for i in range(1, len(lines)):
            if len(lines[i].strip().split()) == 0:
                break
            self.hsk.append(lines[i].strip().split()[0])
            self.hsk_coors.append(float(lines[i].strip().split()[1]))
        for i in range(len(self.hsk)):
            if self.hsk[i] == 'GAMMA' or self.hsk[i] == 'G' or self.hsk[i] == 'Gamma':
                self.hsk[i] = '$\Gamma$'
        return (self.hsk, self.hsk_coors)

    def Band(self):
        self.spin_num = 1
        with open("BAND.dat", "r") as frp:
            lines = frp.readlines()
        # 删除文件内空行
        copy_lines = []
        for i in range(len(lines)):
            if len(lines[i].strip().split()) == 0:
                continue
            copy_lines.append(lines[i].strip().split())
        # 判断是否是自旋极化的计算，是：spin_num=2 否：spin_num=1
        if len(copy_lines[3]) == 3:
            self.spin_num = 2
        # 获取能带指标（能带条数）：band_number
        band_number = 0
        index = []
        for i in range(len(copy_lines)):
            if 'Band' in copy_lines[i] or 'Band-index' in copy_lines[i] or 'Band-Index' in copy_lines[i] or '#Band' in \
                    copy_lines[i] or '#Band-index' in copy_lines[i] or '#Band-Index' in copy_lines[i]:
                index.append(i)
                band_number = int(copy_lines[i][-1])
        # 获取高对称点路径上K点个数：Kpoints_number；获取K点在高对称点路径上的坐标：Kpoints_coors
        Kpoints_number = 0
        self.Kpoints_coors = []
        for i in range(index[0] + 1, len(copy_lines)):
            if 'Band' in copy_lines[i] or 'Band-index' in copy_lines[i] or 'Band-Index' in copy_lines[i] or '#Band' in \
                    copy_lines[i] or '#Band-index' in copy_lines[i] or '#Band-Index' in copy_lines[i]:
                break
            self.Kpoints_coors.append(float(copy_lines[i][0]))
            Kpoints_number += 1
        # Band data
        if self.spin_num == 1:
            self.band_energy = [[0 for i in range(Kpoints_number)] for j in range(band_number)]
        elif self.spin_num == 2:
            self.spin_up_band_energys = [[0 for i in range(Kpoints_number)] for j in range(band_number)]
            self.spin_dn_band_energys = [[0 for i in range(Kpoints_number)] for j in range(band_number)]
        line_index = 0
        for i in range(index[0] + 1, len(copy_lines)):
            if 'Band' in copy_lines[i] or 'Band-index' in copy_lines[i] or 'Band-Index' in copy_lines[i] or '#Band' in \
                    copy_lines[i] or '#Band-index' in copy_lines[i] or '#Band-Index' in copy_lines[i]:
                continue
            band_index = line_index // Kpoints_number
            Kpoint_index = line_index % Kpoints_number
            # 修正文件里的反序排列
            if band_index % 2 == 1:
                Kpoint_index = Kpoints_number - Kpoint_index - 1
            if self.spin_num == 1:
                self.band_energy[band_index][Kpoint_index] = float(copy_lines[i][1])
            elif self.spin_num == 2:
                self.spin_up_band_energys[band_index][Kpoint_index] = float(copy_lines[i][1])
                self.spin_dn_band_energys[band_index][Kpoint_index] = float(copy_lines[i][2])
            line_index += 1
        if self.spin_num == 1:
            return (self.Kpoints_coors, self.band_energy)
        else:
            return (self.Kpoints_coors, self.spin_up_band_energys, self.spin_dn_band_energys)

    def DFT_band_plot(self):
        fig, ax = plt.subplots()
        if self.spin_num == 1:
            for band in self.band_energy:
                plt.plot(self.Kpoints_coors, band, 'r-', linewidth=1.0)
        elif self.spin_num == 2:
            for band in self.spin_up_band_energys:
                plt.plot(self.Kpoints_coors, band, 'r-', linewidth=1.0)
            for band in self.spin_dn_band_energys:
                plt.plot(self.Kpoints_coors, band, '-', color='blue', linewidth=1.0)
        plt.plot([0, self.Kpoints_coors[-1]], [0, 0], color='black', linestyle='--')
        plt.grid(axis='x', c='grey', linestyle='--')
        ax.set_xticks(self.hsk_coors)
        ax.set_xticklabels(self.hsk)
        ax.set_title('DFT band of AA')
        ax.set_xlabel("Wave vector  "r"$\vec{k}$")
        ax.set_ylabel(r"$E - E_{fermi}$"' (eV)')
        plt.xlim([0, self.Kpoints_coors[-1]])
        # plt.ylim([-1.5, 1.5])
        plt.savefig('DFT band of AA', bbox_inches='tight', dpi=600, pad_inches=0.0)
        plt.show()


class WannierBand():
    def __init__(self, E_fermi):
        self.E_fermi = E_fermi

    def Band(self):
        with open("wannier90_band.dat", "r") as frp:
            lines = frp.readlines()
        # 获取能带指标（能带条数）：band_number
        band_number = 0
        for i in range(len(lines)):
            if len(lines[i].strip().split()) == 0:
                band_number += 1
        # 获取高对称点路径上K点个数：Kpoints_number；获取K点在高对称点路径上的坐标：Kpoints_coors
        Kpoints_number = 0
        self.Kpoints_coors = []
        for i in range(len(lines)):
            if len(lines[i].strip().split()) == 0:
                break
            self.Kpoints_coors.append(float(lines[i].strip().split()[0]))
            Kpoints_number += 1
        # Band data
        self.band_energy = [[0 for i in range(Kpoints_number)] for j in range(band_number)]
        line_index = 0
        for i in range(len(lines)):
            if len(lines[i].strip().split()) == 0:
                continue
            band_index = line_index // Kpoints_number
            Kpoint_index = line_index % Kpoints_number
            self.band_energy[band_index][Kpoint_index] = float(lines[i].strip().split()[1]) - self.E_fermi
            line_index += 1
        return (self.Kpoints_coors, self.band_energy)

    def Wannier_band_plot(self):
        fig, ax = plt.subplots()
        for band in self.band_energy:
            plt.plot(self.Kpoints_coors, band, 'r-', linewidth=1.2)
        ax.set_title('Wannier band of AA')
        ax.set_xlabel("Wave vector  "r"$\vec{k}$")
        ax.set_ylabel(r"$E - E_{fermi}$"' (eV)')
        plt.plot([0, self.Kpoints_coors[-1]], [0, 0], color='black', linestyle='--')
        plt.xlim([0, self.Kpoints_coors[-1]])
        # plt.ylim([-1.5, 1.5])
        plt.savefig('wannier band of AA', bbox_inches='tight', dpi=600, pad_inches=0.0)
        plt.show()


def Band_fitting(kernel1, kernel2):
    fig, ax = plt.subplots()
    for band in kernel1.Band()[1]:
        if band != kernel1.Band()[1][-1]:
            plt.plot(kernel1.Band()[0], band, alpha=0.5, color='black', linewidth=1)
            continue
        plt.plot(kernel1.Band()[0], band, alpha=0.5, color='black', linewidth=1, label='DFT-Band')
    for band in kernel2.Band()[1]:
        if band != kernel2.Band()[1][-1]:
            plt.scatter(kernel2.Band()[0], band, s=0.8, color='red', )
            continue
        plt.scatter(kernel2.Band()[0], band, s=0.8, color='red', label='Wannier-Band')
    ax.set_title("Fitting Band")
    plt.plot([0, kernel1.Band()[0][-1]], [0, 0], color='black', linestyle='--')
    plt.grid(axis='x', c='grey', linestyle='--')
    ax.set_xlabel("Wave vector  "r"$\vec{k}$")
    ax.set_ylabel(r"$E - E_{fermi}$"' (eV)')
    ax.set_xticks(kernel1.Klabels()[1])
    ax.set_xticklabels(kernel1.Klabels()[0])
    plt.xlim([0, kernel1.Band()[0][-1]])
    plt.ylim([-5, 5])
    plt.legend(loc='upper right', prop={'size': 8})  # 加图例
    plt.savefig('fitband.jpg', bbox_inches='tight', dpi=600, pad_inches=0.1)
    plt.show()


def main():
    E_fermi = float(input())  # -2.6112
    kernel1 = DFTBand()
    kernel2 = WannierBand(E_fermi)
    Band_fitting(kernel1, kernel2)


if __name__ == "__main__":
    main()
