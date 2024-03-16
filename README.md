# Addressing Loss of Plasticity and Catastrophic Forgetting in Neural Networks

The official repo for reproducing the experiments. You can find the paper from [here](https://openreview.net/forum?id=sKPzAXoylB). If you want to check the implementation for UPGD go [here](https://github.com/mohmdelsayed/upgd/blob/main/core/optim/weight/upgd/first_order.py#L74).

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
TBD

## How to cite
Elsayed, M., & Mahmood, A. R. (2023). Addressing Loss of Plasticity and Catastrophic Forgetting in Neural Networks. <em>In Proceedings of the 12th International Conference on Learning Representations (ICLR)</em>.
