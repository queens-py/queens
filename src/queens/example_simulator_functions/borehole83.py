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
"""Borehole function."""

import numpy as np


def borehole83_lofi(rw, r, tu, hu, tl, hl, l, kw):
    r"""Low-fidelity version of Borehole benchmark function.

    Very simple and quick to evaluate eight dimensional function that models
    water flow through a borehole. Frequently used function for testing a wide
    variety of methods in computer experiments.

    The low-fidelity version is defined as in [1] as:

    :math:`f_{lofi}({\bf x}) = \frac{5 T_u (H_u-H_l)}{\ln(\frac{r}{r_w})(1.5)
    + \frac{2 L T_u}{\ln(\frac{r}{r_w})r_w^2 K_w} + \frac{T_u}{T_l}}`

    For the purposes of uncertainty quantification, the distributions of the
    input random variables are often chosen as:

    | rw  ~ N(0.10, 0.0161812)
    | r   ~ Lognormal(7.71, 1.0056)
    | tu  ~ Uniform[63070, 115600]
    | hu  ~ Uniform[990, 1110]
    | tl  ~ Uniform[63.1, 116]
    | hl  ~ Uniform[700, 820]
    | l   ~ Uniform[1120, 1680]
    | kw  ~ Uniform[9855, 12045]

    Args:
        rw (float): Radius of borehole :math:`(m)` [0.05, 0.15]
        r  (float): Radius of influence :math:`(m)` [100, 50000]
        tu (float): Transmissivity of upper aquifer :math:`(\frac{m^2}{yr})` [63070, 115600]
        hu (float): Potentiometric head of upper aquifer :math:`(m)` [990, 1110]
        tl (float): Transmissivity of lower aquifer :math:`(\frac{m^2}{yr})` [63.1, 116]
        hl (float): Potentiometric head of lower aquifer :math:`(m)` [700, 820]
        l  (float): Length of borehole :math:`(m)` [1120, 1680]
        kw (float): Hydraulic conductivity of borehole :math:`(\frac{m}{yr})` [9855, 12045]

    Returns:
        float: The response is water flow rate, in :math:`(\frac{m^3}{yr})`

    References:
        [1] Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and
            analysis of high-accuracy and low-accuracy computer codes.
            Technometrics, 55(1), 37-46.
    """
    frac1 = 5 * tu * (hu - hl)

    frac2a = 2 * l * tu / (np.log(r / rw) * rw**2 * kw)
    frac2b = tu / tl
    frac2 = np.log(r / rw) * (1.5 + frac2a + frac2b)

    y = frac1 / frac2
    return y


def borehole83_hifi(rw, r, tu, hu, tl, hl, l, kw):
    r"""High-fidelity version of Borehole benchmark function.

    Very simple and quick to evaluate eight dimensional function, that models
    water flow through a borehole. Frequently used function for testing a wide
    variety of methods in computer experiments, see, e.g., [1]-[10].

    The high-fidelity version is defined as:

    :math:`f_{hifi}({\\bf x}) = \frac{2 \pi T_u(H_u-H_l)}{\ln(\frac{r}{r_w})
    \left(1 + \frac{2LT_u}{\ln(\frac{r}{r_w})r_w^2K_w} \right)+ \frac{T_u}{T_l}}`

    For the purpose of multi-fidelity simulation, Xiong et al. (2013) [8] use
    the following function for the lower fidelity code:

    :math:`f_{lofi}({\bf x}) = \frac{5 T_u (H_u-H_l)}{\ln(\frac{r}{r_w})(1.5)
    + \frac{2 L T_u}{\ln(\frac{r}{r_w})r_w^2 K_w} + \frac{T_u}{T_l}}`

    For the purposes of uncertainty quantification, the distributions of the
    input random variables are often chosen as:

    | rw  ~ N(0.10,0.0161812)
    | r   ~ Lognormal(7.71,1.0056)
    | tu  ~ Uniform[63070, 115600]
    | hu  ~ Uniform[990, 1110]
    | tl  ~ Uniform[63.1, 116]
    | hl  ~ Uniform[700, 820]
    | l   ~ Uniform[1120, 1680]
    | kw  ~ Uniform[9855, 12045]

    Args:
        rw (float): Radius of borehole :math:`(m)` [0.05, 0.15]
        r  (float): Radius of influence :math:`(m)` [100, 50000]
        tu (float): Transmissivity of upper aquifer :math:`(\frac{m^2}{yr})` [63070, 115600]
        hu (float): Potentiometric head of upper aquifer :math:`(m)`  [990, 1110]
        tl (float): Transmissivity of lower aquifer :math:`(\frac{m^2}{yr})` [63.1, 116]
        hl (float): Potentiometric head of lower aquifer :math:`(m)` [700, 820]
        l  (float): Length of borehole :math:`(m)` [1120, 1680]
        kw (float): Hydraulic conductivity of borehole :math:`(\frac{m}{yr})`  [9855, 12045]

    Returns:
        float: The response is water flow rate, in :math:`\frac{m^3}{yr}`

    References:
        [1] An, J., & Owen, A. (2001). Quasi-regression. Journal of Complexity,
            17(4), 588-607.

        [2] Gramacy, R. B., & Lian, H. (2012). Gaussian process single-index models
            as emulators for computer experiments. Technometrics, 54(1), 30-41.

        [3] Harper, W. V., & Gupta, S. K. (1983). Sensitivity/uncertainty analysis
            of a borehole scenario comparing Latin Hypercube Sampling and
            deterministic sensitivity approaches (No. BMI/ONWI-516).
            Battelle Memorial Inst., Columbus, OH (USA). Office of Nuclear
            Waste Isolation.

        [4] Joseph, V. R., Hung, Y., & Sudjianto, A. (2008). Blind kriging:
            A new method for developing metamodels. Journal of mechanical design,
            130, 031102.

        [5] Moon, H. (2010). Design and Analysis of Computer Experiments for
            Screening Input Variables (Doctoral dissertation, Ohio State University).

        [6] Moon, H., Dean, A. M., & Santner, T. J. (2012). Two-stage
            sensitivity-based group screening in computer experiments.
            Technometrics, 54(4), 376-387.

        [7] Morris, M. D., Mitchell, T. J., & Ylvisaker, D. (1993).
            Bayesian design and analysis of computer experiments: use of derivatives
            in surface prediction. Technometrics, 35(3), 243-255.

        [8] Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and
            analysis of high-accuracy and low-accuracy computer codes.
            Technometrics, 55(1), 37-46.

        [9] Worley, B. A. (1987). Deterministic uncertainty analysis
            (No. CONF-871101-30). Oak Ridge National Lab., TN (USA).

        [10] Zhou, Q., Qian, P. Z., & Zhou, S. (2011). A simple approach to
             emulation for computer models with qualitative and quantitative
             factors. Technometrics, 53(3).

    For further information, see also http://www.sfu.ca/~ssurjano/borehole.html
    """
    frac1 = 2 * np.pi * tu * (hu - hl)

    frac2a = 2 * l * tu / (np.log(r / rw) * rw**2 * kw)
    frac2b = tu / tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)

    y = frac1 / frac2
    return y
