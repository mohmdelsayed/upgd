from core.task.static_mnist import StaticMNIST
from core.task.label_permuted_mnist import LabelPermutedMNIST
from core.task.summer_with_sign_change import SummerWithSignChange
from core.task.summer_with_signals_change import SummerWithSignalsChange
from core.task.utility_task import UtilityTask

from core.network.fcn_leakyrelu import FullyConnectedLeakyReLU, SmallFullyConnectedLeakyReLU, FullyConnectedLeakyReLUGates
from core.network.fcn_relu import FullyConnectedReLU, SmallFullyConnectedReLU, FullyConnectedReLUGates
from core.network.fcn_tanh import FullyConnectedTanh, SmallFullyConnectedTanh, FullyConnectedTanhGates
from core.network.fcn_linear import FullyConnectedLinear, FullyConnectedLinearGates

from core.learner.sgd import SGDLearner, SGDLearnerWithHesScale
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.learner.upgd import UPGDv1LearnerFOMax, UPGDv1LearnerSONormalized, UPGDv1LearnerFONormalized, UPGDv1LearnerSOMax, UPGDv2LearnerFOMax, UPGDv2LearnerSONormalized, UPGDv2LearnerFONormalized, UPGDv2LearnerSOMax
from core.learner.search import SearchLearnerNormalFONormalized, SearchLearnerNormalSONormalized, SearchLearnerAntiCorrFONormalized, SearchLearnerAntiCorrSONormalized, SearchLearnerNormalFOMax, SearchLearnerNormalSOMax, SearchLearnerAntiCorrFOMax, SearchLearnerAntiCorrSOMax
from core.utilities.fo_utility import FirstOrderUtility
from core.utilities.so_utility import SecondOrderUtility
from core.utilities.weight_utility import WeightUtility
from core.utilities.oracle_utility import OracleUtility
from core.utilities.fo_nvidia_utility import NvidiaUtilityFO
from core.utilities.so_nvidia_utility import NvidiaUtilitySO
from core.utilities.random_utility import RandomUtility
from core.utilities.fo_utility_normalized import FirstOrderUtilityNormalized
from core.utilities.so_utility_normalized import SecondOrderUtilityNormalized

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
    "small_fully_connected_leakyrelu": SmallFullyConnectedLeakyReLU,
    "fully_connected_leakyrelu_gates": FullyConnectedLeakyReLUGates,
    "fully_connected_relu": FullyConnectedReLU,
    "small_fully_connected_relu": SmallFullyConnectedReLU,
    "fully_connected_relu_gates": FullyConnectedReLUGates,
    "fully_connected_tanh": FullyConnectedTanh,
    "small_fully_connected_tanh": SmallFullyConnectedTanh,
    "fully_connected_tanh_gates": FullyConnectedTanhGates,
    "fully_connected_linear": FullyConnectedLinear,
    "fully_connected_linear_gates": FullyConnectedLinearGates,
}

learners = {
    "sgd": SGDLearner,
    "sgd_with_hesscale": SGDLearnerWithHesScale,
    "anti_pgd": AntiPGDLearner,
    "pgd": PGDLearner,
    "upgdv1_normalized_fo": UPGDv1LearnerFONormalized,
    "upgdv1_normalized_so": UPGDv1LearnerSONormalized,
    "upgdv2_normalized_fo": UPGDv2LearnerFONormalized,
    "upgdv2_normalized_so": UPGDv2LearnerSONormalized,
    "upgdv1_max_fo": UPGDv1LearnerFOMax,
    "upgdv1_max_so": UPGDv1LearnerSOMax,
    "upgdv2_max_fo": UPGDv2LearnerFOMax,
    "upgdv2_max_so": UPGDv2LearnerSOMax,
    "search_fo_normal_normalized": SearchLearnerNormalFONormalized,
    "search_so_normal_normalized": SearchLearnerNormalSONormalized,
    "search_fo_anticorr_normalized": SearchLearnerAntiCorrFONormalized,
    "search_so_anticorr_normalized": SearchLearnerAntiCorrSONormalized,
    "search_fo_normal_max": SearchLearnerNormalFOMax,
    "search_so_normal_max": SearchLearnerNormalSOMax,
    "search_fo_anticorr_max": SearchLearnerAntiCorrFOMax,
    "search_so_anticorr_max": SearchLearnerAntiCorrSOMax,
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
    "first_order_normalized": FirstOrderUtilityNormalized,
    "second_order_normalized": SecondOrderUtilityNormalized,
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
