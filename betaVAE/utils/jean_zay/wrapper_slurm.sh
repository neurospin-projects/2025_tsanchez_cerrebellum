#!/bin/bash

source /dev/stdin

cat <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --output=$output
#SBATCH --error=$error

#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3

#SBATCH -A tgu@v100

##SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=timothee.sanchez@cea.fr

module purge

module load pytorch-gpu/py3/2.4.0
set -x

srun python main.py kl=$kl n=$n lr=$lr nb_epoch=$nb_epoch batch_size=$batch_size weights=$weights gradient_max_norm=$gradient_max_norm weight_decay=$weight_decay suffix_dir=$RANDOM
EOF
