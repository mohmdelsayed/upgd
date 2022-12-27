#!/bin/bash
for f in *.txt
do
echo "#!/bin/bash" > ${f%.*}.sh
echo -e "#SBATCH --signal=USR1@90" >> ${f%.*}.sh
echo -e "#SBATCH --job-name="${f%.*}"\t\t\t# single job name for the array" >> ${f%.*}.sh
echo -e "#SBATCH --mem=4G\t\t\t# maximum memory 100M per job" >> ${f%.*}.sh
echo -e "#SBATCH --time=00:30:00\t\t\t# maximum wall time per job in d-hh:mm or hh:mm:ss" >> ${f%.*}.sh
if [[ "${f%.*}" != *'SGD'* ]]; then
echo "#SBATCH --array=1-240" >> ${f%.*}.sh
else
echo "#SBATCH --array=1-40" >> ${f%.*}.sh
fi
echo -e "#SBATCH --gres=gpu:1\t\t\t# Number of GPUs (per node)" >> ${f%.*}.sh
echo -e "#SBATCH --account=def-ashique" >> ${f%.*}.sh
echo -e "#SBATCH --output=%x%A%a.out\t\t\t# standard output (%A is replaced by jobID and %a with the array index)" >> ${f%.*}.sh
echo -e "#SBATCH --error=%x%A%a.err\t\t\t# standard error\n" >> ${f%.*}.sh

echo "FILE=\"\$SCRATCH/GT-learners/grid_search_command_scripts/${f%.*}.txt\""  >> ${f%.*}.sh
echo "SCRIPT=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" \$FILE)"  >> ${f%.*}.sh
echo "module load python/3.7.9" >> ${f%.*}.sh
echo "source \$SCRATCH/GT-learners/.gt-learners/bin/activate" >> ${f%.*}.sh
echo "srun \$SCRIPT" >> ${f%.*}.sh
done