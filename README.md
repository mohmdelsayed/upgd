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

#### 3. Experiment 1: Catastrohpic forgetting in Linear Nets:
```sh
python exp/ex1_cat_forgetting.py
cat generated_cmds/summer_with_sign_change/sgd.txt| bash
```