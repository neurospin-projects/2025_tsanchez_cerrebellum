data_folder="/neurospin/dico/data/cerebellum/datasets/Ukbio"
ssh_connection="uyf32do@jean-zay.idris.fr"
output_dir="/lustre/fswork/projects/rech/tgu/uyf32do/Runs/01_hyperparameter_betaVAE/Input/Ukbio"

subjects=$(ls $data_folder)

echo "Copying $(ls $data_folder | wc -w) files"

for subject in $subjects;
do 
    input_folder=$data_folder/$subject
    output_folder="$output_dir/$subject"
    ssh $ssh_connection "mkdir -p $output_folder"
    rsync -aP $input_folder/*.npy "$ssh_connection:$output_folder/"
    ssh $ssh_connection "ls $output_folder"
done
