# Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

The official repo for reproducing the experiments and UPGD implementation. You can find the paper from [here](https://openreview.net/forum?id=sKPzAXoylB). Here, we provide a minimal implementation of UPGD.

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

## Reproducing the results:
#### 1. You need to have an environment with python 3.7:
``` sh
git clone  --recursive git@github.com:mohmdelsayed/upgd.git
python3.7 -m venv .upgd
source .upgd/bin/activate
```
#### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install HesScale/.
pip install .
```

#### 3. Run experiment:
TBD

## How to cite

#### Bibtex:
```bibtex
@inproceedings{elsayed2024upgd,
  title={Addressing loss of plasticity and catastrophic forgetting in continual learning},
  author={Elsayed, Mohamed and Mahmood, A Rupam},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

#### APA:
Elsayed, M., Mahmood, A. R. (2024). Addressing loss of plasticity and catastrophic forgetting in continual learning. <em>In Proceedings of International Conference on Learning Representations</em>
