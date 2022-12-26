import itertools


class GridSearch:
    """
    GridSearch class to generate all possible combinations of hyperparameters
    """

    def __init__(self, **kwargs):
        keys, values = zip(*kwargs.items())
        self.permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def get_permutations(self):
        return self.permutations


if __name__ == "__main__":
    """
    This class takes a dictionary of hyperparameters and their values and generates all possible combinations. For example, in the above code, we are generating all possible combinations of learning rate, beta1, and beta2.
    """
    grid = GridSearch(
        lr=[2 ** -i for i in range(1, 3)],
        b1=[9 * 10 ** -i for i in range(1, 3)],
        b2=[9 * 10 ** -i for i in range(1, 3)],
    )
    print(grid.get_permutations())
