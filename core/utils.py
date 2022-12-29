from core.task.static_mnist import StaticMNIST
from core.task.label_permuted_mnist import LabelPermutedMNIST
from core.task.summer_with_sign_change import SummerWithSignChange
from core.task.summer_with_signals_change import SummerWithSignalsChange
from core.task.utility_task import UtilityTask

from core.network.fcn_leakyrelu import FullyConnectedLeakyReLU
from core.network.fcn_relu import FullyConnectedReLU
from core.network.fcn_tanh import FullyConnectedTanh, SmallFullyConnectedTanh
from core.network.fcn_linear import FullyConnectedLinear

from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.learner.upgd import UPGDv2LearnerFO, UPGDv2LearnerSO, UPGDv1LearnerFO, UPGDv1LearnerSO
from core.learner.search import SearchLearnerNormalFO, SearchLearnerNormalSO, SearchLearnerAntiCorrFO, SearchLearnerAntiCorrSO
from core.utilities.fo_utility import FirstOrderUtility
from core.utilities.so_utility import SecondOrderUtility
from core.utilities.weight_utility import WeightUtility
from core.utilities.oracle_utility import OracleUtility
from core.utilities.fo_nvidia_utility import NvidiaUtilityFO
from core.utilities.so_nvidia_utility import NvidiaUtilitySO
from core.utilities.random_utility import RandomUtility

import torch
import numpy as np

tasks = {
    "summer_with_sign_change": SummerWithSignChange,
    "summer_with_signals_change": SummerWithSignalsChange,
    "label_permuted_mnist": LabelPermutedMNIST,
    "static_mnist": StaticMNIST,
    "utility_task": UtilityTask,
}

networks = {
    "fully_connected_leakyrelu": FullyConnectedLeakyReLU,
    "fully_connected_relu": FullyConnectedReLU,
    "fully_connected_tanh": FullyConnectedTanh,
    "small_fully_connected_tanh": SmallFullyConnectedTanh,
    "fully_connected_linear": FullyConnectedLinear,
}

learners = {
    "sgd": SGDLearner,
    "anti_pgd": AntiPGDLearner,
    "pgd": PGDLearner,
    "upgdv1_normalized_fo": UPGDv1LearnerFO,
    "upgdv1_normalized_so": UPGDv1LearnerSO,
    "upgdv2_normalized_fo": UPGDv2LearnerFO,
    "upgdv2_normalized_so": UPGDv2LearnerSO,
    "search_fo_normal": SearchLearnerNormalFO,
    "search_so_normal": SearchLearnerNormalSO,
    "search_fo_anticorr": SearchLearnerAntiCorrFO,
    "search_so_anticorr": SearchLearnerAntiCorrSO,
}

criterions = {
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "mse_hesscale": torch.nn.MSELoss,
}

utility_factory = {
    "first_order": FirstOrderUtility,
    "second_order": SecondOrderUtility,
    "weight": WeightUtility,
    "oracle": OracleUtility,
    "nvidia_fo": NvidiaUtilityFO,
    "nvidia_so": NvidiaUtilitySO,
    "random": RandomUtility,
}

def compute_spearman_rank_coefficient(approx_utility, oracle_utility):
    approx_list = []
    oracle_list = []
    for fo, oracle in zip(approx_utility, oracle_utility):
        oracle_list += list(oracle.ravel().numpy())
        approx_list += list(fo.ravel().numpy())

    overall_count = len(approx_list)
    approx_list = np.argsort(np.asarray(approx_list))
    oracle_list = np.argsort(np.asarray(oracle_list))

    difference = np.sum((approx_list - oracle_list) ** 2)
    coeff = 1 - 6.0 * difference / (overall_count * (overall_count**2-1))
    return coeff


def create_script_generator(path):
    cmd='''#!/bin/bash
for f in *.txt
do
echo \"#!/bin/bash\" > ${f%.*}.sh
echo -e \"#SBATCH --signal=USR1@90\" >> ${f%.*}.sh
echo -e \"#SBATCH --job-name=\"${f%.*}\"\\t\\t\\t# single job name for the array\" >> ${f%.*}.sh
echo -e \"#SBATCH --mem=4G\\t\\t\\t# maximum memory 100M per job\" >> ${f%.*}.sh
echo -e \"#SBATCH --time=00:30:00\\t\\t\\t# maximum wall time per job in d-hh:mm or hh:mm:ss\" >> ${f%.*}.sh
if [[ \"${f%.*}\" != *'SGD'* ]]; then
echo \"#SBATCH --array=1-240\" >> ${f%.*}.sh
else
echo \"#SBATCH --array=1-40\" >> ${f%.*}.sh
fi
echo -e \"#SBATCH --gres=gpu:1\\t\\t\\t# Number of GPUs (per node)\" >> ${f%.*}.sh
echo -e \"#SBATCH --account=def-ashique\" >> ${f%.*}.sh
echo -e \"#SBATCH --output=%x%A%a.out\\t\\t\\t# standard output (%A is replaced by jobID and %a with the array index)\" >> ${f%.*}.sh
echo -e \"#SBATCH --error=%x%A%a.err\\t\\t\\t# standard error\\n\" >> ${f%.*}.sh

echo \"FILE=\\"\$SCRATCH/GT-learners/grid_search_command_scripts/${f%.*}.txt\\"\"  >> ${f%.*}.sh
echo \"SCRIPT=\$(sed -n \\"\${SLURM_ARRAY_TASK_ID}p\\" \$FILE)\"  >> ${f%.*}.sh
echo \"module load python/3.7.9\" >> ${f%.*}.sh
echo \"source \$SCRATCH/GT-learners/.gt-learners/bin/activate\" >> ${f%.*}.sh
echo \"srun \$SCRIPT\" >> ${f%.*}.sh
done'''

    with open(f"{path}/create_scripts.bash", "w") as f:
        f.write(cmd)

    
def create_script_runner(path):
    cmd='''#!/bin/bash
for f in *.sh
do sbatch $f
done'''
    with open(f"{path}/run_all_scripts.bash", "w") as f:
        f.write(cmd)
