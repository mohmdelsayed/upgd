from core.task.stationary_mnist import StationaryMNIST
from core.task.label_permuted_mnist import LabelPermutedMNIST
from core.task.label_permuted_cifar100 import LabelPermutedCIFAR100
from core.task.input_permuted_mnist import InputPermutedMNIST
from core.task.changing_average import ChangingAverage
from core.task.permuted_average import PermutedAverage
from core.task.utility_task import UtilityTask

from core.network.fcn_leakyrelu import FullyConnectedLeakyReLU, SmallFullyConnectedLeakyReLU, FullyConnectedLeakyReLUGates, SmallFullyConnectedLeakyReLUGates
from core.network.fcn_relu import FullyConnectedReLU, SmallFullyConnectedReLU, FullyConnectedReLUGates, SmallFullyConnectedReLUGates, ConvolutionalNetworkReLU
from core.network.fcn_tanh import FullyConnectedTanh, SmallFullyConnectedTanh, FullyConnectedTanhGates, SmallFullyConnectedTanhGates
from core.network.fcn_linear import FullyConnectedLinear, FullyConnectedLinearGates, LinearLayer, SmallFullyConnectedLinear, SmallFullyConnectedLinearGates

from core.learner.sgd import SGDLearner, SGDLearnerWithHesScale
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.learner.adam import AdamLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.learner.adaptive_pgd import AdaptivePGDLearner
from core.learner.adaptive_anti_pgd import AdaptiveAntiPGDLearner

from core.learner.weight.upgd import UPGDv2LearnerFOAntiCorrNormalized, UPGDv2LearnerSOAntiCorrNormalized, UPGDv1LearnerFOAntiCorrNormalized, UPGDv1LearnerSOAntiCorrNormalized, UPGDv2LearnerFOAntiCorrMax, UPGDv2LearnerSOAntiCorrMax, UPGDv1LearnerFOAntiCorrMax, UPGDv1LearnerSOAntiCorrMax, UPGDv2LearnerFONormalNormalized, UPGDv2LearnerSONormalNormalized, UPGDv1LearnerFONormalNormalized, UPGDv1LearnerSONormalNormalized, UPGDv2LearnerFONormalMax, UPGDv2LearnerSONormalMax, UPGDv1LearnerFONormalMax, UPGDv1LearnerSONormalMax
from core.learner.weight.search import SearchLearnerNormalFONormalized, SearchLearnerNormalSONormalized, SearchLearnerAntiCorrFONormalized, SearchLearnerAntiCorrSONormalized, SearchLearnerNormalFOMax, SearchLearnerNormalSOMax, SearchLearnerAntiCorrFOMax, SearchLearnerAntiCorrSOMax
from core.learner.weight.random import RandomSearchLearnerNormal, RandomSearchLearnerAntiCorr
from core.learner.weight.adaptive_first_order import AdaptiveUPGDAntiCorrLayerwiseFOLearner, AdaptiveUPGDNormalLayerwiseFOLearner

from core.learner.feature.upgd import FeatureUPGDv1LearnerFOAntiCorrNormalized, FeatureUPGDv1LearnerFONormalNormalized, FeatureUPGDv2LearnerFOAntiCorrNormalized, FeatureUPGDv2LearnerFONormalNormalized, FeatureUPGDv2LearnerFOAntiCorrMax, FeatureUPGDv1LearnerFOAntiCorrMax, FeatureUPGDv2LearnerFONormalMax, FeatureUPGDv1LearnerFONormalMax, FeatureUPGDv1LearnerSOAntiCorrMax, FeatureUPGDv2LearnerSOAntiCorrMax, FeatureUPGDv1LearnerSONormalMax, FeatureUPGDv2LearnerSONormalMax, FeatureUPGDv1LearnerSOAntiCorrNormalized, FeatureUPGDv1LearnerSONormalNormalized, FeatureUPGDv2LearnerSOAntiCorrNormalized, FeatureUPGDv2LearnerSONormalNormalized
from core.learner.feature.search import FeatureSearchLearnerAntiCorrFONormalized, FeatureSearchLearnerNormalFONormalized, FeatureSearchLearnerAntiCorrFOMax, FeatureSearchLearnerNormalFOMax, FeatureSearchLearnerAntiCorrSONormalized, FeatureSearchLearnerNormalSONormalized, FeatureSearchLearnerAntiCorrSOMax, FeatureSearchLearnerNormalSOMax
from core.learner.feature.random import FeatureRandomSearchLearnerNormal, FeatureRandomSearchLearnerAntiCorr

from core.utilities.weight.fo_utility import FirstOrderUtility
from core.utilities.weight.so_utility import SecondOrderUtility
from core.utilities.weight.weight_utility import WeightUtility
from core.utilities.weight.oracle_utility import OracleUtility
from core.utilities.weight.fo_nvidia_utility import NvidiaUtilityFO
from core.utilities.weight.so_nvidia_utility import NvidiaUtilitySO
from core.utilities.weight.random_utility import RandomUtility
from core.utilities.feature.fo_utility import FeatureFirstOrderUtility
from core.utilities.feature.so_utility import FeatureSecondOrderUtility
from core.utilities.feature.oracle_utility import FeatureOracleUtility
from core.utilities.feature.random_utility import FeatureRandomUtility
from core.utilities.feature.fo_nvidia_utility import FeatureNvidiaUtilityFO
from core.utilities.feature.so_nvidia_utility import FeatureNvidiaUtilitySO

import torch
import numpy as np


tasks = {
    "ex1_weight_utils": UtilityTask,
    "ex2_feature_utils": UtilityTask,
    
    "ex3_permuted_average": PermutedAverage,
    "ex3_permuted_average_features": PermutedAverage,

    "ex4_changing_average": ChangingAverage,
    "ex4_changing_average_features": ChangingAverage,

    "ex5_stationary_mnist" : StationaryMNIST,
    "ex5_stationary_mnist_features" : StationaryMNIST,

    "ex6_input_permuted_mnist": InputPermutedMNIST,
    "ex6_input_permuted_mnist_features": InputPermutedMNIST,

    "ex9_label_permuted_mnist" : LabelPermutedMNIST,
    "ex9_label_permuted_mnist_features" : LabelPermutedMNIST,
    "ex9_label_permuted_cifar100" : LabelPermutedCIFAR100,

    "ex10_const_initialization": StationaryMNIST,
    "ex10_zero_initialization": StationaryMNIST,
}

networks = {
    "linear_layer": LinearLayer,
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
    "small_fully_connected_linear": SmallFullyConnectedLinear,
    "small_fully_connected_tanh_gates": SmallFullyConnectedTanhGates,
    "small_fully_connected_leakyrelu_gates": SmallFullyConnectedLeakyReLUGates,
    "small_fully_connected_relu_gates": SmallFullyConnectedReLUGates,
    "small_fully_connected_linear_gates": SmallFullyConnectedLinearGates,
    "convolutional_network_relu": ConvolutionalNetworkReLU,
}

learners = {
    "sgd": SGDLearner,
    "sgd_with_hesscale": SGDLearnerWithHesScale,
    "anti_pgd": AntiPGDLearner,
    "pgd": PGDLearner,
    "adam": AdamLearner,
    "shrink_and_perturb": ShrinkandPerturbLearner,
    "adaptive_pgd": AdaptivePGDLearner,
    "adaptive_anti_pgd": AdaptiveAntiPGDLearner,

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

    "feature_upgd_v2_so_normal_normalized": FeatureUPGDv2LearnerSONormalNormalized,
    "feature_upgd_v1_so_normal_normalized": FeatureUPGDv1LearnerSONormalNormalized,
    "feature_upgd_v2_so_normal_max": FeatureUPGDv2LearnerSONormalMax,
    "feature_upgd_v1_so_normal_max": FeatureUPGDv1LearnerSONormalMax,
    
    "feature_upgd_v2_so_anti_corr_normalized": FeatureUPGDv2LearnerSOAntiCorrNormalized,    
    "feature_upgd_v1_so_anti_corr_normalized": FeatureUPGDv1LearnerSOAntiCorrNormalized,
    "feature_upgd_v2_so_anti_corr_max": FeatureUPGDv2LearnerSOAntiCorrMax,
    "feature_upgd_v1_so_anti_corr_max": FeatureUPGDv1LearnerSOAntiCorrMax,

    "feature_search_so_anti_corr_normalized": FeatureSearchLearnerAntiCorrSONormalized,
    "feature_search_so_anti_corr_max": FeatureSearchLearnerAntiCorrSOMax,

    "feature_search_so_normal_normalized": FeatureSearchLearnerNormalSONormalized,
    "feature_search_so_normal_max": FeatureSearchLearnerNormalSOMax,

    "random_search_normal": RandomSearchLearnerNormal,
    "random_search_anti_corr": RandomSearchLearnerAntiCorr,

    "feature_random_search_normal": FeatureRandomSearchLearnerNormal,
    "feature_random_search_anti_corr": FeatureRandomSearchLearnerAntiCorr,

    "adaptive_upgd_v2_fo_anti_corr_layerwise": AdaptiveUPGDAntiCorrLayerwiseFOLearner,
    "adaptive_upgd_v2_fo_normal_layerwise": AdaptiveUPGDNormalLayerwiseFOLearner,
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
    "nvidia_fo": NvidiaUtilityFO,
    "nvidia_so": NvidiaUtilitySO,
    "random": RandomUtility,
}

feature_utility_factory = {
    "first_order": FeatureFirstOrderUtility,
    "second_order": FeatureSecondOrderUtility,
    "oracle": FeatureOracleUtility,
    "random": FeatureRandomUtility,
    "nvidia_fo": FeatureNvidiaUtilityFO,
    "nvidia_so": FeatureNvidiaUtilitySO,
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

def create_script_generator(path, exp_name):
    cmd=f'''#!/bin/bash
for f in *.txt
do
echo \"#!/bin/bash\" > ${{f%.*}}.sh
echo -e \"#SBATCH --signal=USR1@90\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --job-name=\"${{f%.*}}\"\\t\\t\\t# single job name for the array\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --mem=2G\\t\\t\\t# maximum memory 100M per job\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --time=01:00:00\\t\\t\\t# maximum wall time per job in d-hh:mm or hh:mm:ss\" >> ${{f%.*}}.sh
echo \"#SBATCH --array=1-240\" >> ${{f%.*}}.sh
echo -e \"#SBATCH --account=def-ashique\" >> ${{f%.*}}.sh

echo "cd \"../../\"" >> ${{f%.*}}.sh
echo \"FILE=\\"\$SCRATCH/GT-learners/generated_cmds/{exp_name}/${{f%.*}}.txt\\"\"  >> ${{f%.*}}.sh
echo \"SCRIPT=\$(sed -n \\"\${{SLURM_ARRAY_TASK_ID}}p\\" \$FILE)\"  >> ${{f%.*}}.sh
echo \"module load python/3.7.9\" >> ${{f%.*}}.sh
echo \"source \$SCRATCH/GT-learners/.gt-learners/bin/activate\" >> ${{f%.*}}.sh
echo \"srun \$SCRIPT\" >> ${{f%.*}}.sh
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
