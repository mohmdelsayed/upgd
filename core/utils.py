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

from core.learner.weight.upgd import UPGDv2LearnerFOAntiCorrNormalized, UPGDv2LearnerSOAntiCorrNormalized, UPGDv1LearnerFOAntiCorrNormalized, UPGDv1LearnerSOAntiCorrNormalized, UPGDv2LearnerFOAntiCorrMax, UPGDv2LearnerSOAntiCorrMax, UPGDv1LearnerFOAntiCorrMax, UPGDv1LearnerSOAntiCorrMax, UPGDv2LearnerFONormalNormalized, UPGDv2LearnerSONormalNormalized, UPGDv1LearnerFONormalNormalized, UPGDv1LearnerSONormalNormalized, UPGDv2LearnerFONormalMax, UPGDv2LearnerSONormalMax, UPGDv1LearnerFONormalMax, UPGDv1LearnerSONormalMax
from core.learner.weight.search import SearchLearnerNormalFONormalized, SearchLearnerNormalSONormalized, SearchLearnerAntiCorrFONormalized, SearchLearnerAntiCorrSONormalized, SearchLearnerNormalFOMax, SearchLearnerNormalSOMax, SearchLearnerAntiCorrFOMax, SearchLearnerAntiCorrSOMax

from core.learner.feature.upgd import FeatureUPGDv1LearnerFOAntiCorrNormalized, FeatureUPGDv1LearnerFONormalNormalized, FeatureUPGDv2LearnerFOAntiCorrNormalized, FeatureUPGDv2LearnerFONormalNormalized, FeatureUPGDv2LearnerFOAntiCorrMax, FeatureUPGDv1LearnerFOAntiCorrMax, FeatureUPGDv2LearnerFONormalMax, FeatureUPGDv1LearnerFONormalMax
from core.learner.feature.search import FeatureSearchLearnerAntiCorrFONormalized, FeatureSearchLearnerNormalFONormalized, FeatureSearchLearnerAntiCorrFOMax, FeatureSearchLearnerNormalFOMax

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
    "ex1_weight_utils": SummerWithSignChange,
    "ex2_lop_summer_with_signals_change": SummerWithSignalsChange,
    "ex3_cat_forget_summer_with_sign_change": SummerWithSignChange,
    "ex4_cat_forget_lop_summer_with_sign_change": SummerWithSignChange,
    "ex5_label_permuted_mnist" : LabelPermutedMNIST,
    "ex6_static_mnist" : StaticMNIST,
    "ex7_feature_utils": UtilityTask,
    "ex8_feature_train": SummerWithSignChange,
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

    "upgd_v2_fo_anti_corr_normalized": UPGDv2LearnerFOAntiCorrNormalized,
    "upgd_v2_so_anti_corr_normalized": UPGDv2LearnerSOAntiCorrNormalized,
    "upgd_v1_fo_anti_corr_normalized": UPGDv1LearnerFOAntiCorrNormalized,
    "upgd_v1_so_anti_corr_normalized": UPGDv1LearnerSOAntiCorrNormalized,
    "upgd_v2_fo_anti_corr_max": UPGDv2LearnerFOAntiCorrMax,
    "upgd_v2_so_anti_corr_max": UPGDv2LearnerSOAntiCorrMax,
    "upgd_v1_fo_anti_corr_max": UPGDv1LearnerFOAntiCorrMax,
    "upgd_v1_so_anti_corr_max": UPGDv1LearnerSOAntiCorrMax,

    "upgd_v2_fo_normal_normalized": UPGDv2LearnerFONormalNormalized,
    "upgd_v2_so_normal_normalized": UPGDv2LearnerSONormalNormalized,
    "upgd_v1_fo_normal_normalized": UPGDv1LearnerFONormalNormalized,
    "upgd_v1_so_normal_normalized": UPGDv1LearnerSONormalNormalized,
    "upgd_v2_fo_normal_max": UPGDv2LearnerFONormalMax,
    "upgd_v2_so_normal_max": UPGDv2LearnerSONormalMax,
    "upgd_v1_fo_normal_max": UPGDv1LearnerFONormalMax,
    "upgd_v1_so_normal_max": UPGDv1LearnerSONormalMax,

    "search_fo_anti_corr_normalized": SearchLearnerAntiCorrFONormalized,
    "search_so_anti_corr_normalized": SearchLearnerAntiCorrSONormalized,
    "search_fo_anti_corr_max": SearchLearnerAntiCorrFOMax,
    "search_so_anti_corr_max": SearchLearnerAntiCorrSOMax,

    "search_fo_normal_normalized": SearchLearnerNormalFONormalized,
    "search_so_normal_normalized": SearchLearnerNormalSONormalized,
    "search_fo_normal_max": SearchLearnerNormalFOMax,
    "search_so_normal_max": SearchLearnerNormalSOMax,

    "feature_upgd_v2_fo_normal_normalized": FeatureUPGDv2LearnerFONormalNormalized,
    "feature_upgd_v1_fo_normal_normalized": FeatureUPGDv1LearnerFONormalNormalized,
    "feature_upgd_v2_fo_normal_max": FeatureUPGDv2LearnerFONormalMax,
    "feature_upgd_v1_fo_normal_max": FeatureUPGDv1LearnerFONormalMax,

    "feature_upgd_v2_fo_anti_corr_normalized": FeatureUPGDv2LearnerFOAntiCorrNormalized,
    "feature_upgd_v1_fo_anti_corr_normalized": FeatureUPGDv1LearnerFOAntiCorrNormalized,
    "feature_upgd_v2_fo_anti_corr_max": FeatureUPGDv2LearnerFOAntiCorrMax,
    "feature_upgd_v1_fo_anti_corr_max": FeatureUPGDv1LearnerFOAntiCorrMax,

    "feature_search_fo_anti_corr_normalized": FeatureSearchLearnerAntiCorrFONormalized,
    "feature_search_fo_anti_corr_max": FeatureSearchLearnerAntiCorrFOMax,

    "feature_search_fo_normal_normalized": FeatureSearchLearnerNormalFONormalized,
    "feature_search_fo_normal_max": FeatureSearchLearnerNormalFOMax,

}

criterions = {
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
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
