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
python experiments/ex1_utility_approximation.py
cat generated_cmds/utility_task/*.txt | bash

python experiments/ex2_lop.py
cat generated_cmds/summer_with_signals_change/*.txt | bash

python experiments/ex3_cat_forget.py
cat generated_cmds/summer_with_sign_change/*.txt | bash

python experiments/ex4_cat_forget_lop.py
cat generated_cmds/summer_with_sign_change/*.txt | bash

python experiments/ex5_label_permuted_mnist.py
cat generated_cmds/label_permuted_mnist/*.txt | bash

python experiments/ex6_static_mnist.py
cat generated_cmds/static_mnist/*.txt | bash
```

##### 4. Remove log files:
```sh
./clean.sh
```