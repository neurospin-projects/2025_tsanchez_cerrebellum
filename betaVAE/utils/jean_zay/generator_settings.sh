declare -i count=1

declare -a kl_list=(6 8 10)
declare -a lr_list=("1e-4" "1e-5" "5e-6" "5e-4")
declare -a n_list=(64 75 128)

for kl in "${kl_list[@]}"
do
	for lr in "${lr_list[@]}"
	do
		for n in "${n_list[@]}"
		do
			cat <<EOF >>setting_grid_$count.txt
			job_name="small_grid_$count"
			output="/lustre/fswork/projects/rech/tgu/uyf32do/Runs/01_hyperparameter_betaVAE/Output/output_%j.out"
			error="/lustre/fswork/projects/rech/tgu/uyf32do/Runs/01_hyperparameter_betaVAE/Output/error_%j.err"

			kl=$kl
			lr=$lr
			n=$n
			nb_epoch=250
			batch_size=32
			weights=[1,2]
			gradient_max_norm=100
			weight_decay=0.0001

EOF
count=$((count + 1))
		done
	done
done
