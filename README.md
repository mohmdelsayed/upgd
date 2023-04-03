# Utility-based Perturbed Gradient Descent
:warning: work in progess :warning:

Utility-based Perturbed Gradient Descent (UPGD) is an online representation-learning algorithm that is well-suited for continual learning agents with no knowledge about task boundaries. UPGD protects useful weights or features and perturbs less useful ones based on their utilities. Our empirical results show that our method alleviates catastrophic forgetting and decaying plasticity, enabling the use of modern representation learning methods to work in the continual learning setting.


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
python experiments/ex1_weight_utils.py
cat generated_cmds/ex1_weight_utils/*.txt | bash
```

#### 4. Remove log files:
```sh
./clean.sh
```
