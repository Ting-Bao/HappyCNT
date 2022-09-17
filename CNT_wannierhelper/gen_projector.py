'''
Ting BAO @ Tsinghua University
bao-ting@foxmail.com
2022.09.08
'''


from pymatgen.core.structure import Structure
import numpy as np
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--file',default='POSCAR',type=str,help='specify the CNT POSCAR filename, should be periodical in z axis')
parser.add_argument('--newposcar',default='POSCAR_centered',type=str,help='specify the new structure file')
parser.add_argument('--projectorfile',default='projector.txt',type=str,help='specify the text file to put the projector info in wannier90 format')
args=parser.parse_args()

def move_to_00(file,newposcar):
    old_struct=Structure.from_file(file)
    sites=old_struct.cart_coords
    x_center = np.average(sites[:,0])
    y_center = np.average(sites[:,1])
    new_sites=[[i[0]-x_center,i[1]-y_center,i[2]] for i in sites]
    new_struct=Structure(old_struct.lattice, old_struct.species, new_sites, coords_are_cartesian=True,to_unit_cell=False)
    new_struct.to('poscar',newposcar)

def gen_CNT_projecteor(newposcar,projectorfile):
    ''' newposcar is the centered file
    '''
    struc=Structure.from_file(newposcar)
    sites=struc.cart_coords
    
    # pz part
    pz_part=[]
    for i in sites:
        temp='c= {:> 8.4f}, {:>8.4f}, {:>8.4f} :pz :z=  {:>8.4f}, {:>8.4f},  0.0000 :x=0,0,1'.format(i[0],i[1],i[2],i[0],i[1])
        pz_part.append(temp)
    
    # spart, using PBC!
    s_part=[]
    for i in range(len(sites)):
        for j in range(i+1,len(sites)):
            # the C-C bond is 1.42 A, use 1.6 here for judgement
            if np.linalg.norm(sites[i]-sites[j])< 1.6:
                bc=(sites[i]+sites[j])/2 # bond center
                temp='c= {:>8.4f}, {:>8.4f}, {:>8.4f} :s'.format(bc[0],bc[1],bc[2])
                s_part.append(temp)
            if np.linalg.norm(sites[i]-sites[j]+[0,0,struc.lattice.c])< 1.6:
                bc=(sites[i]+sites[j]+[0,0,struc.lattice.c])/2
                temp='c= {:>8.4f}, {:>8.4f}, {:>8.4f} :s'.format(bc[0],bc[1],bc[2])
                s_part.append(temp)
            if np.linalg.norm(sites[i]-sites[j]-[0,0,struc.lattice.c])< 1.6:
                bc=(sites[i]+sites[j]-[0,0,struc.lattice.c])/2
                temp='c= {:>8.4f}, {:>8.4f}, {:>8.4f} :s'.format(bc[0],bc[1],bc[2])
                s_part.append(temp)
            
    s_part.sort(key=lambda x:float(x.split()[-2]))
    # sort along z axis

    projector_text=['Begin Projections','Ang']
    projector_text.extend(pz_part)
    projector_text.extend(s_part)
    projector_text.append('End Projections')
    projector_text=[i+'\n' for i in projector_text]
    with open(projectorfile,'w',encoding='utf-8') as f:
        f.writelines(projector_text)

def main():
    move_to_00(args.file, args.newposcar)
    gen_CNT_projecteor(args.newposcar,args.projectorfile)

if __name__=='__main__':
    main()