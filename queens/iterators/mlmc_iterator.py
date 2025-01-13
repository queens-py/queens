"""Multilevel Monte Carlo Iterator."""

import logging

import numpy as np

from queens.distributions.uniform_discrete import UniformDiscreteDistribution
from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class MLMCIterator(Iterator):
    """Multi Level Monte Carlo Iterator.

    The equations where taken from a review paper of Giles, Michael titled "Multilevel Monte Carlo
    methods" from 2018. This iterator can be used in two different modes by setting the truth
    value of the parameter use_optimal_n (defaults to False).

    Attributes:
        seed (int):                 Seed for random number generation.
        models (list(Model)):       Models of different fidelity to use for evaluation. The model
                                    fidely and model cost increases with increasing index.
        parameters (Parameters):    Parameters to use for model evaluation.
        num_samples (list(int)):    The number of samples to evaluate each level on. If
                                    use_optimal_n is set to False (defualt), then this respresents
                                    the actualammount of model evaluations on each level. If
                                    use_optimal_n is set to True, the values represent the inital
                                    number of model evaluations on each level needed to estimate
                                    the variance of each level, after which the optimal number of
                                    samples of each level is computed.
        result_description (dict):  Description of desired results.
        samples (list(np.array)):   List of samples arrays for each level.
        output (dict):              Dict with all iterator outputs, e.g. mean, standard deviation
                                    of estimator.
        cost_models (list(int), optional):  List with the relative cost of each model. This
                                            parameter is optional and defaults to None. However,
                                            if use_optimal_n is set to True, you need to provide a
                                            list. Otherwise, the iterator raises a ValueError.
        use_optimal_n (bool, optional): Sets the mode of the iterator to either use num_samples as
                                        the number of model evaluations on each level or use
                                        num_samples as initial samples to calculate the optimal
                                        number of samples from.
        bootstrap_samples (int, optional):  Number of resamples to use for bootstrap estimate of
                                            standard deviation of this estimator. Defualts to 0. If
                                            set to 0, the iterator won't compute a bootstrap
                                            estimate
    """

    @log_init_args
    def __init__(
        self,
        models,
        parameters,
        global_settings,
        seed,
        num_samples,
        cost_models=None,
        use_optimal_num_samples=False,
        result_description=None,
        bootstrap_samples=0,
    ):
        """Constructor MLMCIterator.

        Args:
            models (list(Model)):   Models of different fidelity to use for evaluation. The model
                                    fidely and model cost increases with increasing index.
            parameters (Parameters):    Pass Parameters to use to evalute the SimulationModels with.
            global_settings (GlobalSettings):   Global settings
            seed (int):             Seed to use for samples generation
            num_samples (np.array(int)):    Number of samples to evalute on each level or initial
                                            number of model evaluations, if use_optimal_n is set
                                            to True.
            cost_models (list(int), optional):  List with the relative cost of each model. This
                                            parameter is optional and defaults to None. However,
                                            if use_optimal_n is set to True, you need to provide a
                                            list. Otherwise, the iterator raises a ValueError.
            use_optimal_n (bool, optional): Sets the mode of the iterator to either use num_samples
                                        as the number of model evaluations on each level or use
                                        num_samples as initial samples to calculate the optimal
                                        number of samples from.
            bootstrap_samples (int, optional):  Number of resamples to use for bootstrap estimate of
                                            standard deviation of this estimator. Defualts to 0. If
                                            set to 0, the iterator won't compute a bootstrap
                                            estimate
            result_description (dict, optional):    Pass result description if needed.
                                                    Defaults to None.


        Raises:
            ValueError: If num_samples and models are not of same length,
                        or if models is not a list or num_samples is not a list.
        """
        if (
            not isinstance(models, list)
            or not isinstance(num_samples, list)
            or not len(num_samples) == len(models)
        ):
            raise ValueError("models and num_samples have to lists and need to be of same size")

        if use_optimal_num_samples is True:
            if cost_models is None:
                raise ValueError(
                    "cost_models needs to be specified to use optimal number of samples"
                )

            # The cost of level i is the sum of the cost of model i and model i-1,
            # since both have to be evaluated to compute the expecation of level i
            self.level_cost = [cost_models[0]] + [
                cost_models[i] + cost_models[i - 1] for i in range(1, len(cost_models))
            ]

        super().__init__(None, parameters, global_settings)  # init parent iterator with no model
        self.seed = seed
        self.models = models
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.cost_models = cost_models
        self.use_optimal_n = use_optimal_num_samples
        self.bootstrap_samples = bootstrap_samples
        models = []

    def __draw_samples(self, num_samples):
        """Draws samples from paramter space.

        Args:
            num_samples (list(int)): Number of samples to draw on each level.

        Returns:
            samples (list(np.array)): Drawn samples on each level.
        """
        samples = []

        for num in num_samples:
            samples.append(self.parameters.draw_samples(num))

        return samples

    def __compute_level_statistics(self, level_data):
        """Computes mean and variance for each level.

        Args:
            level_data (list(np.array)): list of np.array results for each level

        Returns:
            (level_mean (np.array), level_var (np.array)), mean, std): A tupel of
                four entries, the first one being the mean of each level and the
                second one being the variance of each level, the third one being
                the mean of the mlmc estimtor and the fourth one being the standard
                deviation of the mlmc estimator
        """
        level_mean = []
        level_var = []
        var = 0
        mean = 0

        for np_arr in level_data:
            # mean estimation of level via sample mean estimator
            level_mean.append(np_arr.mean())

            # variance estimation of level via sample variance estimator
            level_var.append(np_arr.var())

            # update mean
            mean += np_arr.mean()
            var += np_arr.var() / np_arr.size

        std = var**0.5

        return (np.array(level_mean), np.array(level_var), mean, std)

    def pre_run(self):
        """Generate samples for subsequent MLMC analysis and update model."""
        np.random.seed(self.seed)

        self.samples = self.__draw_samples(self.num_samples)

    def core_run(self):
        """Multi-level monte-carlo logic."""
        for i in range(1, len(self.num_samples)):
            if self.num_samples[i] - self.num_samples[i - 1] > 0:
                _logger.warning(
                    "WARNING: Number of samples not decreasing does not fulfil purpose of mlmc"
                )
                break

        # result list of computation on all levels
        result = []

        # level 0
        result.append(self.models[0].evaluate(self.samples[0])["result"])

        # levels 1,2,3 ... n
        for i, samples in enumerate(self.samples[1:], 1):
            result.append(
                self.models[i].evaluate(samples)["result"]
                - self.models[i - 1].evaluate(samples)["result"]
            )

        (level_mean, level_var, mean, std) = self.__compute_level_statistics(result)

        # Loop if use optimal is set to true.
        if self.use_optimal_n is True:

            # While standard error is higher than tolerance, calculate addional samples
            # needed to reach required tolerance

            ideal_num_samples = np.array(self.num_samples)

            # iterate over all levels expect for the last one in reversed order
            # to update the optimal number of samples on each level using
            # n_{l-1} = n_{l} * \sqrt{\frac{Var_{l-1}*Cost_{l}}{Var_{l}*Cost_{l-1}}}
            for i in reversed(range(len(self.num_samples) - 1)):

                # n_l = level_multiplyer * n_{l+1}
                level_multiplier = np.sqrt(
                    (level_var[i] * self.level_cost[i + 1])
                    / (level_var[i + 1] * self.level_cost[i])
                )

                ideal_num_samples[i] = int(ideal_num_samples[i + 1] * level_multiplier)

            # Calculate the difference between to current number of samples and the ideal
            # number of samples. These are the additional samples that have to be computed.
            # If this value is negative, choose 0
            num_samples_additional = np.maximum(
                np.array(ideal_num_samples - self.num_samples), np.zeros(len(self.num_samples))
            ).astype(int)

            self.samples = self.__draw_samples(num_samples_additional)

            # iteration on level 0
            result[0] = np.concatenate(
                (result[0], self.models[0].evaluate(self.samples[0])["result"])
            )

            # iteration on levels 1,2,3 ... n
            for i, samples in enumerate(self.samples[1:], 1):
                if num_samples_additional[i] == 0:
                    continue
                result[i] = np.concatenate(
                    (
                        result[i],
                        self.models[i].evaluate(samples)["result"]
                        - self.models[i - 1].evaluate(samples)["result"],
                    )
                )

            # Update num_samples with addional samples and update level_statistics
            self.num_samples = self.num_samples + num_samples_additional
            level_mean, level_var, mean, std = self.__compute_level_statistics(level_data=result)

        self.output = {
            "result": result,
            "level_mean": level_mean,
            "level_var": level_var,
            "mean": mean,
            "var": std**2,
            "std": std,
            "num_samples": self.num_samples,
        }

        if self.bootstrap_samples > 0:
            self.output["std_bootstrap"] = self.__bootstrap(result)

    def post_run(self):
        """Writes results to result file."""
        write_results(
            processed_results=self.output, file_path=self.global_settings.result_file(".pickle")
        )

    def __bootstrap(self, result):
        """Computation of bootstrap samples.

        Args:
            result (dict): Results of self.core_run()

        Returns:
            float: standard deviation of mlmc estimator using bootstrapping
        """
        var_estimate_bootstrap = 0
        for level in result:
            dist = UniformDiscreteDistribution(np.arange(stop=level.size, dtype=int).reshape(-1, 1))

            bootstrap_sample_mean = np.zeros(self.bootstrap_samples)
            for i in range(self.bootstrap_samples):
                bootstrap_sample_mean[i] = level[dist.draw(level.size)].mean()

            var_estimate_bootstrap += bootstrap_sample_mean.var()

        # return standard deviation
        return var_estimate_bootstrap**0.5
