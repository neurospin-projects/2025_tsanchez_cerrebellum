list_files=(slurm_settings/*.txt)

for file in "${list_files[@]}"
do
	bash wrapper_slurm.sh < $file
done
