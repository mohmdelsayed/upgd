import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from core.run.run import Run
from core.utils import compute_spearman_rank_coefficient, compute_spearman_rank_coefficient_layerwise
from core.utils import utility_factory
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
import signal
import traceback
import time
from functools import partial

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class RunUtility(Run):
    name = 'run_utility'
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

        oracle_factory = utility_factory['oracle'](self.learner.network, criterion)
        util_measure_corr_global = {}
        util_measure_corr_layerwise = {}
        for utility_measure in utility_factory.keys():
            if utility_measure == 'oracle':
                continue
            util_measure_corr_global[utility_factory[utility_measure](self.learner.network, criterion)] = []
            util_measure_corr_layerwise[utility_factory[utility_measure](self.learner.network, criterion)] = []

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

            oracle_util = oracle_factory.compute_utility(loss, input, target) 
            for measure, _ in util_measure_corr_global.items():
                measure_util = measure.compute_utility()
                util_measure_corr_global[measure].append(compute_spearman_rank_coefficient(measure_util, oracle_util))

            for measure, _ in util_measure_corr_layerwise.items():
                measure_util = measure.compute_utility()
                util_measure_corr_layerwise[measure].append(compute_spearman_rank_coefficient_layerwise(measure_util, oracle_util))


        self.logger.log(losses=losses_per_step_size,
                        task=self.task_name, 
                        learner=self.learner.name,
                        network=self.learner.network.name,
                        optimizer_hps=self.learner.optim_kwargs,
                        n_samples=self.n_samples,
                        seed=self.seed,
                        global_correlations={key.name:value for (key,value) in util_measure_corr_global.items()},
                        layerwise_correlations={key.name:value for (key,value) in util_measure_corr_layerwise.items()},
        )


if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunUtility(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")