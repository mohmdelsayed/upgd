# GT-Learners

## Installation:
#### 1. You need to have environemnt with python 3.7:
``` sh
python3.7 -m venv .gt-learners
source .gt-learners/bin/activate
```
#### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install HesScale/.
```

#### 3. Run experiments:
```sh
python exp/ex1_utility_approximation.py
cat generated_cmds/utility_task/sgd.txt | bash

python exp/ex2_lop.py
cat generated_cmds/summer_with_signals_change/sgd.txt | bash
cat generated_cmds/summer_with_signals_change/upgdv2_normalized.txt | bash

python exp/ex3_cat_forget.py
cat generated_cmds/summer_with_sign_change/sgd.txt | bash
cat generated_cmds/summer_with_sign_change/upgdv2_normalized_fo.txt | bash
cat generated_cmds/summer_with_sign_change/upgdv2_normalized_so.txt | bash

python exp/ex4_cat_forget_lop.py
cat generated_cmds/summer_with_sign_change/sgd.txt | bash
cat generated_cmds/summer_with_sign_change/upgdv2_normalized.txt | bash

python exp/ex5_label_permuted_mnist.py
cat generated_cmds/label_permuted_mnist/sgd.txt | bash
cat generated_cmds/label_permuted_mnist/upgdv2_normalized.txt | bash
```

##### 4. Remove log files:
```sh
rm -rf generated_cmds/
rm -rf logs/
```