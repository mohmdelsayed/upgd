# Utility-based Perturbed Gradient Descent

Utility-based Perturbed Gradient Descent (UPGD) is an online learning algorithm well-suited for continual learning agents. UPGD protects useful weights or features from forgetting and perturbs less useful ones based on their utilities. Our empirical results show that UPGD helps reduce forgetting and maintain plasticity, enabling modern representation learning methods to work effectively in continual learning. You can find the paper from [here](https://arxiv.org/abs/2302.03281).

## Installation:
#### 1. You need to have environemnt with python 3.7:
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
```sh
python experiments/ex6_input_permuted_mnist.py
cat generated_cmds/ex6_input_permuted_mnist/*.txt | bash
```

#### 4. Remove log files:
```sh
./clean.sh
```

## How to cite
Elsayed, M., & Mahmood, A. R. (2023). Utility-based Perturbed Gradient Descent: An Optimizer for Continual Learning. <em>arXiv preprint arXiv:2302.03281</em>.