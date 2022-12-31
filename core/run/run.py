import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
import traceback
class Run:
    name = 'run'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, **kwargs):
        self.n_samples = int(n_samples)
        self.task = tasks[task]()
        self.task_name = task
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []

        self.learner.set_task(self.task)
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )

        for _ in range(self.n_samples):
            input, target = next(self.task)
            optimizer.zero_grad()
            output = self.learner.predict(input)
            loss = criterion(output, target)
            if self.learner.extend:
                with backpack(HesScale()):
                    loss.backward()
            else:
                loss.backward()
            optimizer.step(loss)
            losses_per_step_size.append(loss.item())

        self.logger.log(losses=losses_per_step_size,
                        task=self.task_name, 
                        learner=self.learner.name,
                        network=self.learner.network.name,
                        optimizer_hps=self.learner.optim_kwargs,
                        n_samples=self.n_samples,
                        seed=self.seed,
        )


if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = Run(**args)
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"python3 {' '.join(sys.argv)} \n")
    except BaseException as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"python3 {' '.join(sys.argv)} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"python3 {' '.join(sys.argv)} \n")
            f.write(f"{traceback.format_exc()} \n\n")