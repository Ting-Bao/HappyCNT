#PBS -N example
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -q deeph
module load impi/5.1.3.258
module load intel/16.4.258
module load mkl/16.4.258

ulimit -Ss unlimited
cd $PBS_O_WORKDIR
cp $PBS_NODEFILE node
#NCORE=`cat node | wc -l`
NCORE=12
date > output_vaspwann.$PBS_JOBID
export P4_RSHCOMMAND=/opt/pbs/default/bin/pbs_remsh
mpirun -machinefile node -np $NCORE /home/lijh/A.liuyz/apps/1.vasp_wannier/2.soc_100/bin/vasp_ncl >> output_vaspwann.$PBS_JOBID
date >> output_vaspwann.$PBS_JOBID

