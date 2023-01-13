# Utility-based Perturbed Gradient Descent: An Optimizer for Continual Learning

Utility-based Perturbed Gradient Descent (UPGD) is an online representation-learning algorithm that is well-suited for continual learning agents with no knowledge about task boundaries. UPGD protects useful weights or features and perturbs less useful ones based on their utilities. Our empirical results show that our method alleviates catastrophic forgetting and decaying plasticity, enabling the use of modern representation learning methods to work in the continual learning setting.


## Installation:
#### 1. You need to have environemnt with python 3.7:
``` sh
git clone  --recursive git@github.com:mohmdelsayed/GT-learners.git
python3.7 -m venv .gt-learners
source .gt-learners/bin/activate
```
#### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install HesScale/.
pip install .
```

#### 3. Run experiments:
```sh
python experiments/ex1_utility_approximation.py
cat generated_cmds/ex1_weight_utils/*.txt | bash

python experiments/ex2_lop.py
cat generated_cmds/ex2_lop_summer_with_signals_change/*.txt | bash

python experiments/ex3_cat_forget.py
cat generated_cmds/ex3_cat_forget_summer_with_sign_change/*.txt | bash

python experiments/ex4_cat_forget_lop.py
cat generated_cmds/ex4_cat_forget_lop_summer_with_sign_change/*.txt | bash

python experiments/ex5_label_permuted_mnist.py
cat generated_cmds/ex5_label_permuted_mnist/*.txt | bash

python experiments/ex6_static_mnist.py
cat generated_cmds/ex6_static_mnist/*.txt | bash

python experiments/ex7_feature_utility.py
cat generated_cmds/ex7_feature_utils/*.txt | bash

python experiments/ex8_feature_optimizer.py
cat generated_cmds/ex8_feature_train/*.txt | bash
```

#### 4. Remove log files:
```sh
./clean.sh
```
