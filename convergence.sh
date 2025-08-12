#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --time=23:59:00
######SBATCH --time=1:00:00
######SBATCH -p debug
#SBATCH --job-name convergence
#SBATCH --output=Convergence_a_%j.txt
#SBATCH --mail-type=END

#export OMP_NUM_THREADS=10
export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true

module load CCEnv StdEnv/2020 gcc/9.3.0 vtk/9.0.1 python/3.7.7
source $HOME/.virtualenvs/toolsenv/bin/activate

python convergence.py A
#(python convergence.py A && echo "case_A complete")&
#(python convergence.py B && echo "case_B complete")&
#(python convergence.py C && echo "case_C complete")&

#wait