#!/bin/sh
#PBS -N example
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -q deeph

nodes_num=$(cat ${PBS_NODEFILE} | wc -l)

cd ${PBS_O_WORKDIR}
module load intel/18u2
module load intel/20u1


# for i in 3.60 3.65 3.70; do
# for i in 3.15 3.20 3.25 3.30 3.32 3.35 3.37 3.40 3.45 3.50 3.55; do
# cp -f POSCAR$i POSCAR

# mpirun -np ${nodes_num} /home/liyang1/Software/CalcProg/VASP/Main/vasp-544-patched/vasp_intel18u2/bin/vasp_std >> output.$PBS_JOBID
# mpirun -np ${nodes_num} /home/liyang1/Software/CalcProg/VASP/Main/vasp-544-patched/vasp_w90-1.2_intel18u2/bin/vasp_ncl >> output.$PBS_JOBID
mpirun -np 12 /home/liyang1/Software/CalcProg/Wannier90/platform/w003/wannier90-3.1.0_20u1/wannier90.x wannier90
#mpirun -np ${nodes_num} /home/liyang1/Software/CalcProg/WannierTools/wanniertools-2.5.1_intel18u2/bin/wt.x

# E=`tail -1 OSZICAR` ; echo $i $E >>SUMMARY
# done
# done
