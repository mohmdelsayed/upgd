from core.task.label_permuted_mnist import LabelPermutedMNIST
from core.task.summer_with_sign_change import SummerWithSignChange
from core.task.summer_with_signals_change import SummerWithSignalsChange

from core.network.fully_connected_leakyrelu import FullyConnectedLeakyReLU
from core.network.fully_connected_relu import FullyConnectedReLU
from core.network.fully_connected_tanh import FullyConnectedTanh

from core.learner.sgd import SGDLearner
from core.learner.upgd import UPGDv2NormalizedLearner

import torch

tasks = {
    "summer_with_sign_change": SummerWithSignChange,
    "summer_with_signals_change": SummerWithSignalsChange,
    "label_permuted_mnist": LabelPermutedMNIST,
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
}