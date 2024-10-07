from core.task.label_permuted_emnist import LabelPermutedEMNIST
from core.task.label_permuted_mnist_offline import LabelPermutedMNISTOffline
from core.task.label_permuted_cifar10 import LabelPermutedCIFAR10
from core.task.label_permuted_mini_imagenet import LabelPermutedMiniImageNet
from core.task.input_permuted_mnist import InputPermutedMNIST
from core.task.input_permuted_mnist_restarts import InputPermutedMNISTRestarts
from core.task.utility_task import UtilityTask

from core.network.fcn_leakyrelu import FullyConnectedLeakyReLU, SmallFullyConnectedLeakyReLU, FullyConnectedLeakyReLUGates, SmallFullyConnectedLeakyReLUGates
from core.network.fcn_relu import FullyConnectedReLU, SmallFullyConnectedReLU, FullyConnectedReLUGates, SmallFullyConnectedReLUGates, ConvolutionalNetworkReLU, FullyConnectedReLUWithHooks, ConvolutionalNetworkReLUWithHooks
from core.network.fcn_tanh import FullyConnectedTanh, SmallFullyConnectedTanh, FullyConnectedTanhGates, SmallFullyConnectedTanhGates
from core.network.fcn_linear import FullyConnectedLinear, FullyConnectedLinearGates, LinearLayer, SmallFullyConnectedLinear, SmallFullyConnectedLinearGates

from core.learner.sgd import SGDLearner, SGDLearnerWithHesScale
from core.learner.pgd import PGDLearner
from core.learner.adam import AdamLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.learner.ewc import EWCLearner
from core.learner.rwalk import RWalkLearner
from core.learner.mas import MASLearner
from core.learner.synaptic_intelligence import SynapticIntelligenceLearner

from core.learner.weight_upgd import FirstOrderLocalUPGDLearner, SecondOrderLocalUPGDLearner, FirstOrderNonprotectingLocalUPGDLearner, SecondOrderNonprotectingLocalUPGDLearner, FirstOrderGlobalUPGDLearner, SecondOrderGlobalUPGDLearner, FirstOrderNonprotectingGlobalUPGDLearner, SecondOrderNonprotectingGlobalUPGDLearner
from core.learner.feature_upgd import FeatureFirstOrderNonprotectingLocalUPGDLearner, FeatureFirstOrderLocalUPGDLearner, FeatureFirstOrderGlobalUPGDLearner, FeatureFirstOrderNonprotectingGlobalUPGDLearner, FeatureSecondOrderNonprotectingGlobalUPGDLearner, FeatureSecondOrderGlobalUPGDLearner, FeatureSecondOrderNonprotectingLocalUPGDLearner, FeatureSecondOrderLocalUPGDLearner

from core.utilities.weight.fo_utility import FirstOrderUtility
from core.utilities.weight.so_utility import SecondOrderUtility
from core.utilities.weight.weight_utility import WeightUtility
from core.utilities.weight.oracle_utility import OracleUtility
from core.utilities.weight.random_utility import RandomUtility
from core.utilities.weight.grad2_utility import SquaredGradUtility
from core.utilities.feature.fo_utility import FeatureFirstOrderUtility
from core.utilities.feature.so_utility import FeatureSecondOrderUtility
from core.utilities.feature.oracle_utility import FeatureOracleUtility
from core.utilities.feature.random_utility import FeatureRandomUtility
from core.utilities.feature.grad2_utility import FeatureSquaredGradUtility

import torch
import numpy as np


tasks = {
    "weight_utils": UtilityTask,
    "feature_utils": UtilityTask,
    
    "input_permuted_mnist": InputPermutedMNIST,
    "input_permuted_mnist_restarts": InputPermutedMNISTRestarts,

    "label_permuted_emnist" : LabelPermutedEMNIST,
    "label_permuted_emnist_offline" : LabelPermutedMNISTOffline,
    "label_permuted_cifar10" : LabelPermutedCIFAR10,
    "label_permuted_mini_imagenet" : LabelPermutedMiniImageNet,

    "input_permuted_mnist_stats": InputPermutedMNIST,
    "label_permuted_emnist_stats" : LabelPermutedEMNIST,
    "label_permuted_cifar10_stats" : LabelPermutedCIFAR10,
    "label_permuted_mini_imagenet_stats" : LabelPermutedMiniImageNet,

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
    "fully_connected_relu_with_hooks": FullyConnectedReLUWithHooks,
    "convolutional_network_relu_with_hooks": ConvolutionalNetworkReLUWithHooks,
}

learners = {
    "sgd": SGDLearner,
    "sgd_with_hesscale": SGDLearnerWithHesScale,
    "pgd": PGDLearner,
    "adam": AdamLearner,
    "shrink_and_perturb": ShrinkandPerturbLearner,
    "ewc": EWCLearner,
    "rwalk": RWalkLearner,
    "mas": MASLearner,
    "si": SynapticIntelligenceLearner,

    "upgd_fo_local": FirstOrderLocalUPGDLearner,
    "upgd_so_local": SecondOrderLocalUPGDLearner,
    "upgd_nonprotecting_fo_local": FirstOrderNonprotectingLocalUPGDLearner,
    "upgd_nonprotecting_so_local": SecondOrderNonprotectingLocalUPGDLearner,
    "upgd_fo_global": FirstOrderGlobalUPGDLearner,
    "upgd_so_global": SecondOrderGlobalUPGDLearner,
    "upgd_nonprotecting_fo_global": FirstOrderNonprotectingGlobalUPGDLearner,
    "upgd_nonprotecting_so_global": SecondOrderNonprotectingGlobalUPGDLearner,

    "feature_upgd_fo_local": FeatureFirstOrderLocalUPGDLearner,
    "feature_upgd_nonprotecting_fo_local": FeatureFirstOrderNonprotectingLocalUPGDLearner,
    "feature_upgd_fo_global": FeatureFirstOrderGlobalUPGDLearner,
    "feature_upgd_nonprotecting_fo_global": FeatureFirstOrderNonprotectingGlobalUPGDLearner,

    "feature_upgd_so_local": FeatureSecondOrderLocalUPGDLearner,
    "feature_upgd_nonprotecting_so_local": FeatureSecondOrderNonprotectingLocalUPGDLearner,
    "feature_upgd_so_global": FeatureSecondOrderGlobalUPGDLearner,
    "feature_upgd_nonprotecting_so_global": FeatureSecondOrderNonprotectingGlobalUPGDLearner,

}

criterions = {
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
}

utility_factory = {
    "first_order": FirstOrderUtility,
    "second_order": SecondOrderUtility,
    "random": RandomUtility,
    "weight": WeightUtility,
    "oracle": OracleUtility,
    "g2": SquaredGradUtility,
}

feature_utility_factory = {
    "first_order": FeatureFirstOrderUtility,
    "second_order": FeatureSecondOrderUtility,
    "oracle": FeatureOracleUtility,
    "random": FeatureRandomUtility,
    "g2": FeatureSquaredGradUtility,
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
echo \"FILE=\\"\$SCRATCH/upgd/generated_cmds/{exp_name}/${{f%.*}}.txt\\"\"  >> ${{f%.*}}.sh
echo \"SCRIPT=\$(sed -n \\"\${{SLURM_ARRAY_TASK_ID}}p\\" \$FILE)\"  >> ${{f%.*}}.sh
echo \"module load python/3.7.9\" >> ${{f%.*}}.sh
echo \"source \$SCRATCH/upgd/.upgd/bin/activate\" >> ${{f%.*}}.sh
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
