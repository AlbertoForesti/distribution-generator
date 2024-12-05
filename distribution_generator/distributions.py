from distribution_generator.evolution_lib import *
from scipy.stats import rv_discrete
from scipy.stats._multivariate import multi_rv_frozen

import itertools

class DistributionManager:
    def __init__(self):
        self.distribution_config = None
        self.evolution_task = None
        self.distribution = None

    def __call__(self, 
                 mutual_information: float,
                 dim_x: int,
                 dim_y: int,
                 seq_length_x: int = 1,
                 seq_length_y: int = 1,
                 scale: float = 1.0,
                 loc: float = 0.0,
                 mean: np.array = None,
                 cov: np.array = None,
                 strategy='comma',
                 mu=25,
                 population_size=50,
                 n_generations=100,
                 min_val=0,
                 force_retrain=False) -> None:

        if not force_retrain and self.distribution_config is not None and self.distribution_config == DistributionConfig(mutual_information, dim_x, dim_y, seq_length_x, seq_length_y, min_val):
            return self.rv
        self.distribution_config = DistributionConfig(mutual_information, dim_x, dim_y, seq_length_x, seq_length_y, min_val)
        self.train_tasks(scale, loc, mean, cov, strategy, mu, population_size, min_val, n_generations)
        self.rv = JointDiscrete(self.distribution, vocabulary_x=self.possible_x_sequences, vocabulary_y=self.possible_y_sequences)
        return self.rv

    def train_tasks(self, scale, loc, mean, cov, strategy, mu, population_size, min_val, n_generations):
        """
        Function to train many tasks to reach the desired mutual information by stacking random variables
        It only generates distributions with pairwise dependencies between tokens
        """

        seq_length_x = self.distribution_config.seq_length_x
        seq_length_y = self.distribution_config.seq_length_y
        mutual_information = self.distribution_config.mutual_information
        dim_x = self.distribution_config.dim_x
        dim_y = self.distribution_config.dim_y

        assert self.distribution_config.seq_length_x == self.distribution_config.seq_length_y, "Currently only supports same sequence length for x and y"

        relevant_dims = min(seq_length_x, seq_length_y)
        unit_mutinfo = mutual_information / relevant_dims
        self.evolution_tasks = []
        self.distributions = []
        for dim in range(relevant_dims):
            task = EvolutionTask(unit_mutinfo, dim_x, dim_y, scale, loc, mean, cov, strategy, mu, population_size, min_val)
            task.train(n_generations)
            self.distributions.append(task.best_agent.distribution)
        
        dist = np.zeros((dim_x**seq_length_x, dim_y**seq_length_y))

        self.possible_x_sequences = np.array(list(itertools.product(range(dim_x), repeat=relevant_dims)))
        self.possible_y_sequences = np.array(list(itertools.product(range(dim_y), repeat=relevant_dims)))
        
        for idx_x, x in enumerate(self.possible_x_sequences):
            for idx_y, y in enumerate(self.possible_y_sequences):
                joint_prob = 1
                for i in range(relevant_dims):
                    joint_prob *= self.distributions[i][x[i], y[i]]
                dist[idx_x, idx_y] = joint_prob
        self.distribution = dist

class DistributionConfig:
    def __init__(self, mutual_information: float, dim_x: int, dim_y: int, seq_length_x: int, seq_length_y: int, min_val: float):
        self.mutual_information = mutual_information
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.min_val = min_val
        self.seq_length_x = seq_length_x
        self.seq_length_y = seq_length_y
    
    def __eq__(self, value: object) -> bool:
        return self.mutual_information == value.mutual_information and\
               self.dim_x == value.dim_x and\
               self.dim_y == value.dim_y and\
               self.min_val == value.min_val and\
               self.seq_length_x == value.seq_length_x and\
               self.seq_length_y == value.seq_length_y

class JointDiscrete(multi_rv_frozen):

    def __init__(self, joint_dist, *args, vocabulary_x=None, vocabulary_y=None, **kwargs):
        self.joint_dist = joint_dist
        self.vocab_x = vocabulary_x
        self.vocab_y = vocabulary_y
        pmf = joint_dist.flatten()
        values = np.arange(len(pmf))
        self._hidden_univariate = rv_discrete(name="hidden_univariate", values=(values, pmf))
        super().__init__(*args, **kwargs)
    
    def rvs(self, *args, **kwargs):
        if len(args) > 1:
            raise NotImplementedError("Different sizes not implemented, pass keyworkd argument size instead")
        elif len(args) == 1:
            kwargs['size'] = args[0]
            args = []
        samples = self._hidden_univariate.rvs(*args, **kwargs)
        samples = np.unravel_index(samples, self.joint_dist.shape)
        samples = np.stack(samples, axis=1)
        
        X = samples[:,0].reshape(-1,1)
        if self.vocab_x is not None:
            X = self.vocab_x[X]
        
        Y = samples[:,1].reshape(-1,1)
        if self.vocab_y is not None:
            Y = self.vocab_y[Y]
        
        return X, Y
    
    @property
    def entropy(self):
        return -np.sum(self.joint_dist * np.log(self.joint_dist))
    
    @property
    def mutual_information(self):
        marginal_x = np.sum(self.joint_dist, axis=1)
        marginal_y = np.sum(self.joint_dist, axis=0)
        return np.sum(self.joint_dist * np.log(self.joint_dist / np.outer(marginal_x, marginal_y)))

distribution_manager = DistributionManager()

def get_rv(mutual_information: float,
                     dim_x: int,
                     dim_y: int,
                     seq_length_x: int = 1,
                     seq_length_y: int = 1,
                     scale: float = 1.0,
                     loc: float = 0.0,
                     mean: np.array = None,
                     cov: np.array = None,
                     strategy='comma',
                     mu=25,
                     population_size=50,
                     n_generations=100,
                     min_val=0,
                     force_retrain=False) -> None:
    assert mutual_information <= min(np.log(dim_x**seq_length_x), np.log(dim_y**seq_length_y)), f"Mutual information is too high for the given dimensions, max is {min(np.log(dim_x**seq_length_x), np.log(dim_y**seq_length_y))} nats"
    assert mutual_information >= 0, "Mutual information must be non-negative"
    custom_rv = distribution_manager(mutual_information,
                                dim_x,
                                dim_y,
                                seq_length_x,
                                seq_length_y,
                                scale,
                                loc,
                                mean,
                                cov,
                                strategy,
                                mu,
                                population_size,
                                n_generations,
                                min_val,
                                force_retrain)
    return custom_rv