# define the problem or the task that you want to the learner to solve
# this is the base class for all tasks


class Task:
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.iterator = self.generator()

    def __str__(self) -> str:
        return self.name

    def reset(self):
        self.iterator = self.generator()

    def generator(self):
        NotImplementedError("This method should be implemented by the subclass")

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)
