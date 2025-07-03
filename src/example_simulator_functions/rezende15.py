#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Probabilistic models in [1].

[1]: Rezende, D. J., & Mohamed, S. (2016). Variational Inference with
Normalizing Flows. ArXiv:1505.      05770 [Cs, Stat].
http://arxiv.org/abs/1505.05770
"""

import numpy as np


def rezende15_potential1(x, theta=None, as_logpdf=False):
    r"""First potential in [1].

    The unnormalized probabilistic model used is proportional to
    :math:`p(\theta)\propto \exp(-U(\theta))`
    where :math:`U(\theta)` is a potential. Hence the `log_posterior_unnormalized` is given by
    :math:`-U(\theta)`.

    Args:
        x (np.ndarray): Samples at which to evaluate the potential (2 :math:`\times` n_samples)
        theta (float): Angle in radiants in which to rotate the potential
        as_logpdf (bool,optional): *True* if :math:`-U` is to be returned

    Returns:
        np.ndarray: Potential or unnormalized logpdf
    """
    if theta:
        cos, sin = np.cos(theta), np.sin(theta)
        r = np.array(((cos, -sin), (sin, cos)))
        x = np.dot(x, r.T)

    # Construct z
    z_1, z_2 = x[:, 0], x[:, 1]
    norm = np.sqrt(z_1**2 + z_2**2)

    # First term
    outer_term_1 = 0.5 * ((norm - 2) / 0.4) ** 2

    # Second term
    inner_term_1 = np.exp((-0.5 * ((z_1 - 2) / 0.6) ** 2))
    inner_term_2 = np.exp((-0.5 * ((z_1 + 2) / 0.6) ** 2))
    outer_term_2 = np.log(inner_term_1 + inner_term_2 + 1e-7)

    # Potential
    potential = outer_term_1 - outer_term_2

    if as_logpdf:
        return -potential

    return potential
