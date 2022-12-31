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
from core.learner.feature.upgd import FeatureUPGDv2Learner
from core.utilities.weight.fo_utility import FirstOrderUtility
from core.utilities.weight.so_utility import SecondOrderUtility
from core.utilities.weight.weight_utility import WeightUtility
from core.utilities.weight.oracle_utility import OracleUtility
from core.utilities.weight.fo_nvidia_utility import NvidiaUtilityFO
from core.utilities.weight.so_nvidia_utility import NvidiaUtilitySO
from core.utilities.weight.random_utility import RandomUtility
from core.utilities.feature.fo_utility import FeatureFirstOrderUtility
from core.utilities.feature.oracle_utility import FeatureOracleUtility
from core.utilities.feature.random_utility import FeatureRandomUtility

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
    "feature_upgdv2": FeatureUPGDv2Learner,
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
    # "nvidia_fo": NvidiaUtilityFO,
    # "nvidia_so": NvidiaUtilitySO,
    "random": RandomUtility,
}

feature_utility_factory = {
    "first_order": FeatureFirstOrderUtility,
    "oracle": FeatureOracleUtility,
    "random": FeatureRandomUtility,
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
    
def compute_spearman_rank_coefficient_layerwise(approx_utility, oracle_utility):
    coeffs = []
    for fo, oracle in zip(approx_utility, oracle_utility):
        overall_count = len(list(oracle.ravel().numpy()))
        if overall_count == 1:
            continue
        oracle_list = np.argsort(list(oracle.ravel().numpy()))
        approx_list = np.argsort(list(fo.ravel().numpy()))
        difference = np.sum((approx_list - oracle_list) ** 2)
        coeff = 1 - 6.0 * difference / (overall_count * (overall_count**2-1))
        coeffs.append(coeff)
    coeff_average = np.mean(np.array(coeffs))
    return coeff_average
    
def compute_kandell_rank_coefficient(approx_utility, oracle_utility):
    approx_list = []
    oracle_list = []
    for fo, oracle in zip(approx_utility, oracle_utility):
        oracle_list += list(oracle.ravel().numpy())
        approx_list += list(fo.ravel().numpy())

    n = len(approx_list)

    ranked_x = np.argsort(np.asarray(approx_list))
    ranked_y = np.argsort(np.asarray(oracle_list))

    num_concordant_pairs = 0
    num_discordant_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            if ranked_x[i] < ranked_x[j] and ranked_y[i] > ranked_y[j]:
                num_discordant_pairs += 1
            elif ranked_x[i] > ranked_x[j] and ranked_y[i] < ranked_y[j]:
                num_discordant_pairs += 1
            else:
                num_concordant_pairs += 1
    return (num_concordant_pairs - num_discordant_pairs) / (n * (n-1) / 2)

def compute_kandell_rank_coefficient_layerwise(approx_utility, oracle_utility):
    coeffs = []
    for fo, oracle in zip(approx_utility, oracle_utility):
        oracle_list = list(oracle.ravel().numpy())
        approx_list = list(fo.ravel().numpy())

        n = len(approx_list)
        if n == 1:
            continue
        ranked_x = np.argsort(np.asarray(approx_list))
        ranked_y = np.argsort(np.asarray(oracle_list))

        num_concordant_pairs = 0
        num_discordant_pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                if ranked_x[i] < ranked_x[j] and ranked_y[i] > ranked_y[j]:
                    num_discordant_pairs += 1
                elif ranked_x[i] > ranked_x[j] and ranked_y[i] < ranked_y[j]:
                    num_discordant_pairs += 1
                else:
                    num_concordant_pairs += 1
        coeff =  (num_concordant_pairs - num_discordant_pairs) / (n * (n-1) / 2)
        coeffs.append(coeff)
    coeff_average = np.mean(np.array(coeffs))
    return coeff_average

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
