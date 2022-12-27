from core.task.label_permuted_mnist import LabelPermutedMNIST
from core.task.summer_with_sign_change import SummerWithSignChange
from core.task.summer_with_signals_change import SummerWithSignalsChange
from core.task.utility_task import UtilityTask

from core.network.fully_connected_leakyrelu import FullyConnectedLeakyReLU
from core.network.fully_connected_relu import FullyConnectedReLU
from core.network.fully_connected_tanh import FullyConnectedTanh

from core.learner.sgd import SGDLearner
from core.learner.upgd import UPGDv2NormalizedLearner
from core.utilities.fo_utility import FirstOrderUtility
from core.utilities.so_utility import SecondOrderUtility
from core.utilities.weight_utility import WeightUtility
from core.utilities.oracle_utility import OracleUtility
from core.utilities.fo_nvidia_utility import NvidiaUtilityFO
from core.utilities.so_nvidia_utility import NvidiaUtilitySO

import torch
import numpy as np

tasks = {
    "summer_with_sign_change": SummerWithSignChange,
    "summer_with_signals_change": SummerWithSignalsChange,
    "label_permuted_mnist": LabelPermutedMNIST,
    "utility_task": UtilityTask,
}

networks = {
    "fully_connected_leakyrelu": FullyConnectedLeakyReLU,
    "fully_connected_relu": FullyConnectedReLU,
    "fully_connected_tanh": FullyConnectedTanh,
}

learners = {
    "sgd": SGDLearner,
    "upgdv2_normalized": UPGDv2NormalizedLearner,
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

