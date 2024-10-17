from distribution_generator.evolution_lib import *
from scipy.stats import rv_discrete
from scipy.stats._multivariate import multi_rv_frozen

class DistributionManager:
    def __init__(self):
        self.distribution_config = None
        self.evolution_task = None
        self.distribution = None

    def __call__(self, 
                 target_mutinfo: float,
                 dim_x: int,
                 dim_y: int,
                 scale: float = 1.0,
                 loc: float = 0.0,
                 mean: np.array = None,
                 cov: np.array = None,
                 strategy='comma',
                 mu=25,
                 population_size=50,
                 force_retrain=False) -> None:

        if not force_retrain and self.distribution_config is not None and self.distribution_config == DistributionConfig(target_mutinfo, dim_x, dim_y):
            return self.distribution
        self.distribution_config = DistributionConfig(target_mutinfo, dim_x, dim_y)
        self.evolution_task = EvolutionTask(target_mutinfo, dim_x, dim_y, scale, loc, mean, cov, strategy, mu, population_size)
        self.evolution_task.train()
        self.distribution = self.evolution_task.best_agent.distribution
        return self.distribution

class DistributionConfig:
    def __init__(self, target_mutinfo: float, dim_x: int, dim_y: int):
        self.target_mutinfo = target_mutinfo
        self.dim_x = dim_x
        self.dim_y = dim_y
    
    def __eq__(self, value: object) -> bool:
        return self.target_mutinfo == value.target_mutinfo and self.dim_x == value.dim_x and self.dim_y == value.dim_y

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
        return samples

distribution_manager = DistributionManager()

def get_rv(target_mutinfo: float,
                     dim_x: int,
                     dim_y: int,
                     scale: float = 1.0,
                     loc: float = 0.0,
                     mean: np.array = None,
                     cov: np.array = None,
                     strategy='comma',
                     mu=25,
                     population_size=50,
                     force_retrain=False) -> None:
    assert target_mutinfo <= min(np.log(dim_x), np.log(dim_y)), f"Mutual information is too high for the given dimensions, max is {min(np.log(dim_x), np.log(dim_y))} nats"
    assert target_mutinfo >= 0, "Mutual information must be non-negative"
    dist = distribution_manager(target_mutinfo,
                                dim_x,
                                dim_y,
                                scale,
                                loc,
                                mean,
                                cov,
                                strategy,
                                mu,
                                population_size,
                                force_retrain)
    custom_rv = JointDiscrete(dist)
    return custom_rv