import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
from core.network.gate import GateLayer, GateLayerGrad
import signal
import traceback
import time
from functools import partial
import numpy as np

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class RunOffline:
    name = 'run_offline'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, **kwargs):
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task]()
        self.task_name = task
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []
        held_out_averages = []
        held_out_averages1 = []
        held_out_averages2 = []
        held_out_accuracies = []
        held_out_accuracies1 = []
        held_out_accuracies2 = []

        if self.task.criterion == 'cross_entropy':
            accuracy_per_step_size = []
        self.learner.set_task(self.task)
        if self.learner.extend:    
            extension = HesScale()
            extension.set_module_extension(GateLayer, GateLayerGrad())
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )

        for _ in range(self.n_samples):
            (input, target), context = next(self.task)
            input, target = input.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output1, output2 = self.learner.predict(input)
            output = output1 if context else output2
            loss = criterion(output, target)
            if self.learner.extend:
                with backpack(extension):
                    loss.backward()
            else:
                loss.backward()
            optimizer.step(loss)
            losses_per_step_size.append(loss.item())
            if self.task.criterion == 'cross_entropy':
                accuracy_per_step_size.append((output.argmax(dim=1) == target).float().mean().item())

            # evaluate on held out set with no grad
            held_out_losses = []
            held_out_losses1 = []
            held_out_losses2 = []
            held_out_accuracy = []
            held_out_accuracy1 = []
            held_out_accuracy2 = []
            for sample in range(100):
                sample = next(self.task.held_out)
                with torch.no_grad():
                    (input, target), context = sample
                    input, target = input.to(self.device), target.to(self.device)
                    output1, output2 = self.learner.predict(input)
                    output = output1 if context else output2
                    loss = criterion(output, target)
                if context:
                    held_out_losses1.append(loss.item())
                    if self.task.criterion == 'cross_entropy':
                        held_out_accuracy1.append((output.argmax(dim=1) == target).float().mean().item())
                else:
                    held_out_losses2.append(loss.item())
                    if self.task.criterion == 'cross_entropy':
                        held_out_accuracy2.append((output.argmax(dim=1) == target).float().mean().item())
                held_out_losses.append(loss.item())
                if self.task.criterion == 'cross_entropy':
                    held_out_accuracy.append((output.argmax(dim=1) == target).float().mean().item())
            # compute the average loss on the held out set using numpy
            held_out_averages.append(sum(held_out_losses) / len(held_out_losses))
            held_out_averages1.append(sum(held_out_losses1) / len(held_out_losses1))
            held_out_averages2.append(sum(held_out_losses2) / len(held_out_losses2))
            
            if self.task.criterion == 'cross_entropy':
                held_out_accuracies.append(sum(held_out_accuracy) / len(held_out_accuracy))
                held_out_accuracies1.append(sum(held_out_accuracy1) / len(held_out_accuracy1))
                held_out_accuracies2.append(sum(held_out_accuracy2) / len(held_out_accuracy2))

        if self.task.criterion == 'cross_entropy':
            self.logger.log(losses=losses_per_step_size,
                            accuracies=accuracy_per_step_size,
                            task=self.task_name, 
                            learner=self.learner.name,
                            network=self.learner.network.name,
                            optimizer_hps=self.learner.optim_kwargs,
                            n_samples=self.n_samples,
                            seed=self.seed,
                            held_out_averages=held_out_averages,
                            held_out_averages1=held_out_averages1,
                            held_out_averages2=held_out_averages2,
                            held_out_accuracies=held_out_accuracies,
                            held_out_accuracies1=held_out_accuracies1,
                            held_out_accuracies2=held_out_accuracies2,
            )
        else:
            self.logger.log(losses=losses_per_step_size,
                            task=self.task_name,
                            learner=self.learner.name,
                            network=self.learner.network.name,
                            optimizer_hps=self.learner.optim_kwargs,
                            n_samples=self.n_samples,
                            seed=self.seed,
                            held_out_averages=held_out_averages,
                            held_out_averages1=held_out_averages1,
                            held_out_averages2=held_out_averages2,
            )

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunOffline(**args)
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