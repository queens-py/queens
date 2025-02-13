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
"""Levenberg Marquardt iterator."""

import logging

import numpy as np
import pandas as pd
import plotly.express as px

from queens.iterators.iterator import Iterator
from queens.utils.fd_jacobian import fd_jacobian
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class LevenbergMarquardt(Iterator):
    """Iterator for Levenberg-Marquardt deterministic optimization problems.

    Implements the Levenberg-Marquardt algorithm for optimization, adapted from the
    4C `gen_inv_analysis` approach. This class focuses on a simplified yet controlled
    optimization process.

    Attributes:
        initial_guess (np.ndarray): Initial guess for the optimization parameters.
        bounds (tuple): Tuple specifying the lower and upper bounds for parameters. If None, no
                        bounds are applied.
        havebounds (bool): Flag indicating if bounds are provided.
        param_current (np.ndarray): Current parameter values being optimized.
        jac_rel_step (float): Relative step size used for finite difference approximation of the
                              Jacobian.
        max_feval (int): Maximum number of allowed function evaluations.
        result_description (dict): Configuration for result file handling and plotting.
        jac_abs_step (float): Absolute step size used for finite difference approximation of the
                              Jacobian.
        reg_param (float): Regularization parameter for the Levenberg-Marquardt algorithm.
        init_reg (float): Initial value for the regularization parameter.
        update_reg (str): Strategy for updating the regularization parameter ("res" for
                          residual-based, "grad" for gradient-based).
        tolerance (float): Convergence tolerance for the optimization process.
        verbose_output (bool): If True, provides detailed output during optimization.
        iter_opt (int): The iteration number at which the lowest error was achieved.
        lowesterror (float or None): The minimum error encountered during the optimization process.
        param_opt (np.ndarray): Parameter values corresponding to the minimum error.
        solution (np.ndarray): The final optimized parameter values.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        initial_guess=None,
        bounds=None,
        jac_rel_step=1e-4,
        jac_abs_step=0.0,
        init_reg=1.0,
        update_reg="grad",
        convergence_tolerance=1e-6,
        max_feval=1,
        verbose_output=False,
    ):
        """Initializes the Levenberg-Marquardt iterator.

        Args:
            model (Model): Model to be optimized.
            parameters (Parameters): Object containing parameter settings.
            global_settings (GlobalSettings): Configuration settings for the QUEENS experiment.
            result_description (dict): Description and configuration for result handling and
                                       plotting.
            initial_guess (np.ndarray, optional): Initial guess for optimization parameters.
            bounds (tuple, optional): Tuple of lower and upper bounds for parameters. If None, no
                                      bounds are used.
            jac_rel_step (float, optional): Relative step size for finite difference approximation.
            jac_abs_step (float, optional): Absolute step size for finite difference approximation.
            init_reg (float, optional): Initial regularization parameter.
            update_reg (str, optional): Strategy for updating the regularization parameter ("res" or
                                        "grad").
            convergence_tolerance (float, optional): Convergence tolerance for the optimization.
            max_feval (int, optional): Maximum number of function evaluations allowed.
            verbose_output (bool, optional): If True, enables verbose output during optimization.
        """
        super().__init__(model, parameters, global_settings)

        self.havebounds = True
        if bounds is None:
            self.havebounds = False

        self.initial_guess = np.array(initial_guess, dtype=float)
        self.bounds = bounds
        self.param_current = self.initial_guess
        self.jac_rel_step = jac_rel_step
        self.max_feval = max_feval
        self.result_description = result_description

        self.jac_abs_step = jac_abs_step
        self.reg_param = init_reg
        self.init_reg = init_reg
        self.update_reg = update_reg
        self.tolerance = convergence_tolerance

        self.verbose_output = verbose_output
        self.iter_opt = 0
        self.lowesterror = None
        self.param_opt = None
        self.solution = None

    def jacobian_and_residual(self, x0):
        """Evaluate Jacobian and residual of objective function at *x0*.

        For LM we can restrict to "2-point".

        Args:
            x0 (numpy.ndarray): Vector with current parameters

        Returns:
            jacobian_matrix (numpy.ndarray): Jacobian Matrix approximation from finite differences
            f0 (numpy.ndarray): Residual of objective function at `x0`
        """
        positions, delta_positions = self.get_positions_raw_2pointperturb(x0)

        self.model.evaluate(positions)

        f = self.model.response["result"]
        f_batch = f[-(len(positions)) :]

        f0 = f_batch[0]  # first entry corresponds to f(x0)
        f_perturbed = np.delete(f_batch, 0, 0)  # delete the first entry
        jacobian_matrix = fd_jacobian(f0, f_perturbed, delta_positions, True, "2-point")
        # sanity checks:
        # in the case of LSQ, the number of residuals needs to be
        # greater or equal to the number of parameters to be fitted
        num_res, num_par = jacobian_matrix.shape
        if num_res < num_par:
            raise ValueError(
                f"Number of residuals (={num_res}) has to be greater or equal to"
                f" number of parameters (={num_par})."
                f" You have {num_res}<{num_par}."
            )

        return jacobian_matrix, f0

    def pre_run(self):
        """Initialize run.

        Print console output and optionally open .csv file for results
        and write header.
        """
        _logger.info("Initialize Levenberg-Marquardt run.")

        # produce .csv file and write header
        if self.result_description:
            if self.result_description["write_results"]:
                df = pd.DataFrame(
                    columns=["iter", "resnorm", "gradnorm", "params", "delta_params", "mu"],
                )
                csv_file = self.global_settings.result_file(".csv")
                df.to_csv(csv_file, mode="w", sep="\t", index=None)

    def core_run(self):
        """Core run of Levenberg Marquardt iterator."""
        resnorm = np.inf
        gradnorm = np.inf

        converged = False

        i = 0

        # Levenberg Marquardt iterations
        while not converged:
            if i > self.max_feval:
                converged = True
                _logger.info("Maximum number of steps max_feval= %d reached.", self.max_feval)
                break

            _logger.info(
                "iteration: %d reg_param: %s current_parameters: %s",
                i,
                self.reg_param,
                self.param_current,
            )
            jacobian, residual = self.jacobian_and_residual(self.param_current)

            # store terms for repeated usage
            jacobian_inner_product = (jacobian.T).dot(jacobian)
            jacobian_transpose_residual_product = (jacobian.T).dot(residual)

            resnorm_o = resnorm
            resnorm = np.linalg.norm(residual) / np.sqrt(residual.size)

            gradnorm_o = gradnorm
            gradnorm = np.linalg.norm(jacobian_transpose_residual_product)

            if i == 0:
                resnorm_o = resnorm
                gradnorm_o = gradnorm
                self.lowesterror = resnorm
                self.param_opt = self.param_current

            # save most optimal step in resnorm
            if resnorm < self.lowesterror:
                self.lowesterror = resnorm
                self.param_opt = self.param_current
                self.iter_opt = i

            # compute update step. Not necessary for last iteration! Should usually be low
            # dimension matrix inversion. We need J and r anyway for convergence check.
            param_delta = np.linalg.solve(
                jacobian_inner_product + self.reg_param * np.diag(np.diag(jacobian_inner_product)),
                -(jacobian_transpose_residual_product),
            )

            # output for iteration. Before update including next step
            self.printstep(i, resnorm, gradnorm, param_delta)

            # evaluate bounds
            if self.havebounds and self.checkbounds(param_delta, i):
                if self.reg_param > 1.0e6 * self.init_reg:
                    _logger.info(
                        "WARNING: STEP #%d IS OUT OF BOUNDS and reg_param is unreasonably "
                        "high. Ending iterations!",
                        i,
                    )
                    break
                continue

            # update param for next step
            self.param_current = self.param_current + param_delta

            # update reg_param and check for tolerance
            if self.update_reg == "res":
                if resnorm < self.tolerance:
                    converged = True
                    break
                self.reg_param = self.reg_param * resnorm / resnorm_o
            elif self.update_reg == "grad":
                if gradnorm < self.tolerance:
                    converged = True
                    break
                if resnorm < resnorm_o and gradnorm < gradnorm_o:
                    self.reg_param = self.reg_param * gradnorm / gradnorm_o
            else:
                raise ValueError("update_reg unknown")
            i += 1

        # store set of parameters which leads to the lowest residual as solution
        self.solution = self.param_opt

    def post_run(self):
        """Post run.

        Write solution to the console and optionally create .html plot
        from result file.
        """
        _logger.info("The optimum:\t%s occurred in iteration #%s.", self.solution, self.iter_opt)
        if self.result_description:
            if self.result_description["plot_results"] and self.result_description["write_results"]:
                data = pd.read_csv(
                    self.global_settings.result_file(".csv"),
                    sep="\t",
                )
                xydata = data["params"]
                xydata = xydata.str.extractall(r"([+-]?\d+\.\d*e?[+-]?\d*)")
                xydata = xydata.unstack()
                data = data.drop(columns="params")
                i = 0
                for column in xydata:
                    data[self.parameters.names[i]] = xydata[column].astype(float)
                    i = i + 1

                if i > 2:
                    _logger.warning(
                        "write_results for more than 2 parameters not implemented, "
                        "because we are limited to 3 dimensions. "
                        "You have: %d. Plotting is skipped.",
                        i,
                    )
                    return
                if i == 2:
                    fig = px.line_3d(
                        data,
                        x=self.parameters.names[0],
                        y=self.parameters.names[1],
                        z="resnorm",
                        hover_data=[
                            "iter",
                            "resnorm",
                            "gradnorm",
                            "delta_params",
                            "mu",
                            self.parameters.names[0],
                            self.parameters.names[1],
                        ],
                    )
                    fig.update_traces(mode="lines+markers", marker={"size": 2}, line={"width": 4})
                elif i == 1:
                    fig = px.line(
                        data,
                        x=self.parameters.names[0],
                        y="resnorm",
                        hover_data=[
                            "iter",
                            "resnorm",
                            "gradnorm",
                            "delta_params",
                            "mu",
                            self.parameters.names[0],
                        ],
                    )
                    fig.update_traces(mode="lines+markers", marker={"size": 7}, line={"width": 3})
                else:
                    raise ValueError("You shouldn't be here without parameters.")

                fig.write_html(self.global_settings.result_file(".html"))

    def get_positions_raw_2pointperturb(self, x0):
        """Get parameter sets for objective function evaluations.

        Args:
            x0 (numpy.ndarray): Vector with current parameters

        Returns:
            positions (numpy.ndarray): Parameter batch for function evaluation
            delta_positions (numpy.ndarray): Parameter perturbations for finite difference
              scheme
        """
        delta_positions = self.jac_abs_step + self.jac_rel_step * np.abs(x0)
        # if bounds do not allow finite difference step, use opposite direction
        if self.havebounds:
            perturbed = x0 + delta_positions
            for i, current_p in enumerate(perturbed):
                if (current_p < self.bounds[0][i]) or (current_p > self.bounds[1][i]):
                    delta_positions[i] = -delta_positions[i]

        positions = np.zeros((x0.size + 1, x0.size))
        positions[0] = x0

        for i in range(x0.size):
            positions[i + 1] = x0
            positions[i + 1][i] = positions[i + 1][i] + delta_positions[i]

        delta_positions.shape = (delta_positions.size, 1)

        return positions, delta_positions

    def printstep(self, i, resnorm, gradnorm, param_delta):
        """Print iteration data to console and optionally to file.

        Opens file in append mode, so that file is updated frequently.

        Args:
            i (int): Iteration number
            resnorm (float): Residual norm
            gradnorm (float): Gradient norm
            param_delta (numpy.ndarray): Parameter step
        """
        # write iteration to file
        if self.result_description:
            if self.result_description["write_results"]:
                df = pd.DataFrame(
                    {
                        "iter": i,
                        "resnorm": np.format_float_scientific(resnorm, precision=8),
                        "gradnorm": np.format_float_scientific(gradnorm, precision=8),
                        "params": [np.array2string(self.param_current, precision=8)],
                        "delta_params": [np.array2string(param_delta, precision=8)],
                        "mu": np.format_float_scientific(self.reg_param, precision=8),
                    }
                )
                csv_file = self.global_settings.result_file(".csv")
                df.to_csv(
                    csv_file, mode="a", sep="\t", header=None, index=None, float_format="%.8f"
                )

    def checkbounds(self, param_delta, i):
        """Check if proposed step is in bounds.

        Otherwise double regularization and compute new step.

        Args:
            param_delta (numpy.ndarray): Parameter step
            i (int): Iteration number

        Returns:
            stepisoutside (bool): Flag if proposed step is out of bounds
        """
        stepisoutside = False
        nextstep = self.param_current + param_delta
        if np.any(nextstep < self.bounds[0]) or np.any(nextstep > self.bounds[1]):
            stepisoutside = True
            _logger.warning(
                "WARNING: STEP #%d IS OUT OF BOUNDS; double reg_param and compute new iteration."
                "\n declined step was: %s",
                i,
                nextstep,
            )

            self.reg_param *= 2

        return stepisoutside
