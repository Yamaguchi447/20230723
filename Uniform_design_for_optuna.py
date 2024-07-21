import itertools
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import warnings
import optuna
import numpy as np
from scipy.stats import qmc

from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.distributions import CategoricalDistribution
import optunahub
import pyunidoe as pydoe

_logger = get_logger(__name__)

class UniformDesignSampler(BaseSampler):
    def __init__(
        self, search_space: Mapping[str, BaseDistribution], discretization_level: int, seed: Optional[int] = 1234
    ) -> None:
        # check search_space type
        for param_name, distribution in search_space.items():
            assert isinstance(
                distribution,
                (
                    FloatDistribution,
                    IntDistribution,
                    CategoricalDistribution,
                ),
            ), '{} contains a value with the type of {}, which is not supported by UniformDesignSampler. Please make sure a value is `int`, `float` or `categorical` for persistent storage.'.format(param_name, type(distribution))

        # generate initial samples
        self._search_space = search_space
        self._param_names = sorted(search_space.keys())
        self._num_params = len(self._param_names)
        self._discretization_level = discretization_level

        base_ud = pydoe.gen_ud(n=self._discretization_level, s=self._num_params, q=self._discretization_level, crit="CD2", maxiter=100, random_state=seed)["final_design"]
        ud_space = np.repeat(np.linspace(1 / (2 * self._discretization_level), 1 - 1 / (2 * self._discretization_level), self._discretization_level).reshape([-1, 1]), self._num_params, axis=1)

        self._ud_space = np.zeros((self._discretization_level, self._num_params))
        for i in range(self._num_params):
            self._ud_space[:, i] = ud_space[base_ud[:, i] - 1, i]

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Instead of returning param values, UniformDesignSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        # When the trial is created by RetryFailedTrialCallback or enqueue_trial, we should not
        # assign a new grid_id.
        if "grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return

        if 0 <= trial.number and trial.number < len(self._ud_space):
            study._storage.set_trial_system_attr(
                trial._trial_id, "search_space", self._search_space
            )
            study._storage.set_trial_system_attr(trial._trial_id, "grid_id", trial.number)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # return self._rng.uniform(0, 1)
        if "grid_id" not in trial.system_attrs:
            message = "All parameters must be specified when using UniformDesignSampler with enqueue_trial."
            raise ValueError(message)

        if param_name not in self._search_space:
            message = "The parameter name, {}, is not found in the given grid.".format(param_name)
            raise ValueError(message)

        grid_id = trial.system_attrs["grid_id"]
        param_value = self._ud_space[grid_id][self._param_names.index(param_name)]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value

def after_trial(
    self,
    study: Study,
    trial: FrozenTrial,
    state: TrialState,
    values: Sequence[float] | None,
) -> None:
    # グリッドが使い果たされたかを確認
    if len(self._ud_space) == 0:

        # 新しいユニフォームデザインを追加サンプル
        stat = pydoe.gen_aud(
            xp=self._ud_space, 
            n=20,  # 追加サンプル数
            s=self._num_params, 
            q=self._discretization_level, 
            crit="CD2", 
            maxiter=100
        )
        x_f2 = stat["final_design"]

        # サンプル点を追加
        self._ud_space = np.vstack([self._ud_space, x_f2])


    
from matplotlib import pylab as plt

def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    obj = 2 * np.cos(10*  x) * np.sin(10 * y) + np.sin(10 * x * y)
    return  obj

def objective_show(parameters):
    x1 = parameters['x']
    x2 = parameters['y']
    obj = 2 * np.cos(10*  x1) * np.sin(10 * x2) + np.sin(10 * x1 * x2)
    return  obj

# Define the search space
search_space = {"x": FloatDistribution(0, 1), "y": FloatDistribution(0, 1)}

# Create the study
discretization_level = 20
sampler = UniformDesignSampler(search_space, discretization_level)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=40)

logs = study.trials_dataframe()
# print(logs)

def plot_trajectory(xlim, ylim, func, logs, title):
    grid_num = 25
    xlist = np.linspace(xlim[0], xlim[1], grid_num)
    ylist = np.linspace(ylim[0], ylim[1], grid_num)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((grid_num,grid_num))
    for i, x in enumerate(xlist):
        for j, y in enumerate(ylist):
            Z[j, i] = func({"x": x, "y": y})

    cp = plt.contourf(X, Y, Z)
    plt.scatter(logs.loc[:, ['params_x']],
                logs.loc[:, ['params_y']], color="red")
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.colorbar(cp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)

plot_trajectory([0, 1], [0, 1], objective_show, logs, "UD")