import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from core.run import Run
from core.utils import compute_spearman_rank_coefficient
from core.utils import utility_factory
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale

class RunUtility(Run):
    name = 'run_utility'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, **kwargs):
        self.n_samples = int(n_samples)
        self.task = tasks[task]()
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []

        self.learner.set_task(self.task)
        criterion = extend(criterions[self.task.criterion]())
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )

        oracle_factory = utility_factory['oracle'](self.learner.network, criterion)
        util_measure_corr = {}
        for utility_measure in utility_factory.keys():
            if utility_measure == 'oracle':
                continue
            util_measure_corr[utility_factory[utility_measure](self.learner.network, criterion)] = []
        for _ in range(self.n_samples):
            input, target = next(self.task)
            optimizer.zero_grad()
            output = self.learner.predict(input)
            loss = criterion(output, target)
            with backpack(HesScale()):
                loss.backward()         
            optimizer.step(loss)
            losses_per_step_size.append(loss.item())

            oracle_util = oracle_factory.compute_utility(loss, input, target) 
            for measure, _ in util_measure_corr.items():
                measure_util = measure.compute_utility()
                util_measure_corr[measure].append(compute_spearman_rank_coefficient(measure_util, oracle_util))

        self.logger.log(losses=losses_per_step_size,
                        task=self.task.name, 
                        learner=self.learner.name,
                        network=self.learner.network.name,
                        optimizer_hps=self.learner.optim_kwargs,
                        n_samples=self.n_samples,
                        seed=self.seed,
                        utilities={key.name:value for (key,value) in util_measure_corr.items()},
        )


if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunUtility(**args)
    run.start()