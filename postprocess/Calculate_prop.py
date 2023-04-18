'''
Ting Bao 2023/02/13
Read the stored full matrix H(phi,kz) and do postprocess, including:
- Berry curvature/phase and Chern number
- mag-related chi
'''

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('QtAgg')

class HamiltonianProcess():
    def __init__(self,source, name='example_name', E_fermi=0.0, CNT_r=4):
        self.source = source
        self.name = name
        self.E_fermi = E_fermi
        self.CNT_r = CNT_r
        # read in the full H matrix in npy format
        self._read_full_H()
        self.berry_curvature()
        print('h')

    def _read_full_H(self):
        ''' read full_H.npy in the source file'''
        self.full_H=np.load(os.path.join(self.source,'full_H.npy'))
        print("Reading Hamiltonian finished! H shape:",self.full_H.shape)

    def calc_chi()
    def berry_curvature(self,method='willson_loop'):
        ''' Calculate the berry curvature
        method = {willson_loop, normal_formula}
        '''
        if method ==  'willson_loop':
            # TODO
            self.berry_curvature()
        elif method== 'normal_formula':
            # TODO
            self.berry_curvature()
        else:
            raise AttributeError('Wrong method!')






def main():
    kernel = HamiltonianProcess(source='work/example-3-3/',
                                name='3-3CNT',
                                E_fermi=-2.4239,
                                CNT_r=2.03)


if __name__ == '__main__':
    main()