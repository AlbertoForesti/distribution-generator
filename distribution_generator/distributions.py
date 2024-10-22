from distribution_generator.evolution_lib import *
from scipy.stats import rv_discrete
from scipy.stats._multivariate import multi_rv_frozen

class DistributionManager:
    def __init__(self):
        self.distribution_config = None
        self.evolution_task = None
        self.distribution = None

    def __call__(self, 
                 mutual_information: float,
                 dim_x: int,
                 dim_y: int,
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

        if not force_retrain and self.distribution_config is not None and self.distribution_config == DistributionConfig(mutual_information, dim_x, dim_y, min_val):
            return self.distribution
        self.distribution_config = DistributionConfig(mutual_information, dim_x, dim_y, min_val)
        self.evolution_task = EvolutionTask(mutual_information, dim_x, dim_y, scale, loc, mean, cov, strategy, mu, population_size, min_val)
        self.evolution_task.train(n_generations)
        self.distribution = self.evolution_task.best_agent.distribution
        return self.distribution

class DistributionConfig:
    def __init__(self, mutual_information: float, dim_x: int, dim_y: int, min_val: float):
        self.mutual_information = mutual_information
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.min_val = min_val
    
    def __eq__(self, value: object) -> bool:
        return self.mutual_information == value.mutual_information and self.dim_x == value.dim_x and self.dim_y == value.dim_y and self.min_val == value.min_val

class JointDiscrete(multi_rv_frozen):

    def __init__(self, joint_dist, *args, **kwargs):
        self.joint_dist = joint_dist
        pmf = joint_dist.flatten()
        values = np.arange(len(pmf))
        self._hidden_univariate = rv_discrete(name="hidden_univariate", values=(values, pmf))
        super().__init__(*args, **kwargs)
    
    def rvs(self, *args, **kwargs):
        if len(args) > 0:
            raise NotImplementedError("Different sizes not implemented, pass keyworkd argument size instead")
        samples = self._hidden_univariate.rvs(*args, **kwargs)
        samples = np.unravel_index(samples, self.joint_dist.shape)
        samples = np.stack(samples, axis=1)
        return samples[:,0], samples[:,1] # return X,Y

distribution_manager = DistributionManager()

def get_rv(mutual_information: float,
                     dim_x: int,
                     dim_y: int,
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
    assert mutual_information <= min(np.log(dim_x), np.log(dim_y)), f"Mutual information is too high for the given dimensions, max is {min(np.log(dim_x), np.log(dim_y))} nats"
    assert mutual_information >= 0, "Mutual information must be non-negative"
    dist = distribution_manager(mutual_information,
                                dim_x,
                                dim_y,
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
    custom_rv = JointDiscrete(dist)
    return custom_rv