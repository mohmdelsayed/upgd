# Utility-based Perturbed Gradient Descent: An Optimizer for Continual Learning

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

#### 3. Run experiments in the paper:
```sh
python experiments/ex1_weight_utils.py
cat generated_cmds/ex1_weight_utils/*.txt | bash

python experiments/ex2_feature_utils.py
cat generated_cmds/ex2_feature_utils/*.txt | bash

python experiments/ex3_permuted_average.py
cat generated_cmds/ex3_permuted_average/*.txt | bash

python experiments/ex4_changing_average.py
cat generated_cmds/ex4_changing_average/*.txt | bash

python experiments/ex5_stationary_mnist.py
cat generated_cmds/ex5_stationary_mnist/*.txt | bash

python experiments/ex6_input_permuted_mnist.py
cat generated_cmds/ex6_input_permuted_mnist/*.txt | bash

python experiments/ex7_binary_split_mnist.py
cat generated_cmds/ex7_binary_split_mnist/*.txt | bash

python experiments/ex8_two_label_change_mnist.py
cat generated_cmds/ex8_two_label_change_mnist/*.txt | bash

python experiments/ex9_label_permuted_mnist.py
cat generated_cmds/ex9_label_permuted_mnist/*.txt | bash
```

#### 4. Remove log files:
```sh
./clean.sh
```
