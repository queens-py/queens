"""Uniform distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from pqueens.distributions.distributions import Distribution


class UniformDistribution(Distribution):
    """Uniform distribution class.

    Attributes:
        lower_bound (np.ndarray): Lower bound(s) of the distribution
        upper_bound (np.ndarray): Upper bound(s) of the distribution
        width (np.ndarray): Width(s) of the distribution
        pdf_const (float): Constant for the evaluation of the pdf
        logpdf_const (float): Constant for the evaluation of the log pdf
    """

    def __init__(
        self, lower_bound, upper_bound, width, pdf_const, logpdf_const, mean, covariance, dimension
    ):
        """Initialize uniform distribution.

        Args:
            lower_bound (np.ndarray): Lower bound(s) of the distribution
            upper_bound (np.ndarray): Upper bound(s) of the distribution
            width (np.ndarray): Width(s) of the distribution
            pdf_const (float): Constant for the evaluation of the pdf
            logpdf_const (float): Constant for the evaluation of the log pdf
            mean (np.ndarray): Mean of the distribution
            covariance (np.ndarray): Covariance of the distribution
            dimension (int): Dimensionality of the distribution
        """
        super().__init__(mean=mean, covariance=covariance, dimension=dimension)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width
        self.pdf_const = pdf_const
        self.logpdf_const = logpdf_const

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: UniformDistribution object
        """
        lower_bound = np.array(distribution_options['lower_bound']).reshape(-1)
        upper_bound = np.array(distribution_options['upper_bound']).reshape(-1)
        super().check_bounds(lower_bound, upper_bound)
        width = upper_bound - lower_bound

        mean = (lower_bound + upper_bound) / 2.0
        covariance = np.diag(width**2 / 12.0)
        dimension = mean.size

        pdf_const = 1.0 / np.prod(width)
        logpdf_const = np.log(pdf_const)

        return cls(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            width=width,
            pdf_const=pdf_const,
            logpdf_const=logpdf_const,
            mean=mean,
            covariance=covariance,
            dimension=dimension,
        )

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): CDF at evaluated positions
        """
        cdf = np.prod(
            np.clip(
                (x.reshape(-1, self.dimension) - self.lower_bound) / self.width,
                a_min=np.zeros(self.dimension),
                a_max=np.ones(self.dimension),
            ),
            axis=1,
        )
        return cdf

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        samples = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=num_draws)
        return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        x = x.reshape(-1, self.dimension)
        within_bounds = (x >= self.lower_bound).all(axis=1) * (x <= self.upper_bound).all(axis=1)
        logpdf = np.where(within_bounds, self.logpdf_const, -np.inf)
        return logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        x = x.reshape(-1, self.dimension)
        # Check if positions are within bounds of the uniform distribution
        within_bounds = (x >= self.lower_bound).all(axis=1) * (x <= self.upper_bound).all(axis=1)
        logpdf = within_bounds * self.pdf_const
        return logpdf

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = scipy.stats.uniform.ppf(q=q, loc=self.lower_bound, scale=self.width).reshape(-1)
        return ppf