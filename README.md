# Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

The official repo for reproducing the experiments. You can find the paper from [here](https://openreview.net/forum?id=sKPzAXoylB). Here we describe how to reproduce the results. If you only want the implementation of the UPGD algorithm you can find it here:

```python
import torch

class UPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.001, beta_utility=0.999, sigma=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma)
        super(UPGD, self).__init__(params, defaults)
    def step(self):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction_utility) / global_max_util)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1-scaled_utility),
                    alpha=-2.0*group["lr"],
                )
```

## Installation:
### 1. You need to have environemnt with python 3.7:
``` sh
git clone  --recursive git@github.com:mohmdelsayed/upgd.git
python3.7 -m venv .upgd
source .upgd/bin/activate
```
### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install HesScale/.
pip install .
```

### 3. Run experiment:
#### Weight utility experiment (Figure 2):
```sh
python experiments/weight_utility.py
```
This would generate a list of python cmds you need to run them. After they are done, the results would be saved in `logs/` in a JSON format. To plot, use the following:
```sh
python core/plot/plotter_utility.py
```

#### Input-permuted MNIST (Figure 3):
You first need to define the grid search of each method from here `experiments/input_permuted_mnist.py` then you generate then python cmds using:
```sh
python experiments/input_permuted_mnist.py
```
This would generate a list of python cmds you need to run them. After they are done, the results would be saved in `logs/` in a JSON format. To plot, use the following after choosing what to plot:
```sh
python core/plot/plotter.py
```

#### Label-permuted CIFAR-10/EMNIST/miniImageNet (Figure 6):
You first need to define the grid search of each method then you generate then python cmds using:
```sh
python experiments/label_permuted_emnist.py
python experiments/label_permuted_cifar10.py
python experiments/label_permuted_emnist.py
```
This would generate a list of python cmds you need to run them. After they are done, the results would be saved in `logs/` in a JSON format. To plot, use the following after choosing what to plot:
```sh
python core/plot/plotter.py
```

#### Input/Label-permuted Tasks Diagnostic Statistics (Figure 5):
You first need to choose the method and the hyperparameter setting you want to run the statistics on from:
```sh
python experiments/statistics_input_permuted_mnist.py
python experiments/statistics_output_permuted_cifar10.py
python experiments/statistics_output_permuted_emnist.py
python experiments/statistics_output_permuted_imagenet.py
```
This would generate a list of python cmds you need to run them. After they are done, the results would be saved in `logs/` in a JSON format.


#### Policy collapse experiment (Figure 8):
You need to choose the environment id and the seed number. In the paper, we averaged over 30 different seeds.
```sh
python core/run/rl/ppo_continuous_action_adam.py --seed 0 --env_id HalfCheetah-v4
python core/run/rl/ppo_continuous_action_upgd.py --seed 0 --env_id HalfCheetah-v4
```

## License

Distributed under the MIT License. See `LICENSE` for more information.


## How to cite

#### Bibtex:
```bibtex
@inproceedings{elsayed2024addressing,
    title={Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning},
    author={Mohamed Elsayed and A. Rupam Mahmood},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```

#### APA:
Elsayed, M., & Mahmood, A. R. (2023). Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning. <em>In Proceedings of the 12th International Conference on Learning Representations (ICLR)</em>.