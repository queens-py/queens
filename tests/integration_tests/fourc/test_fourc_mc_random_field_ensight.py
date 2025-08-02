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
"""Test 4C with RF materials."""

import logging

import numpy as np
import pytest

from queens.data_processors.ensight_file import EnsightFile
from queens.drivers.fourc import Fourc
from queens.external_geometries.fourc_dat import FourcDat
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.parameters.random_fields.karhunen_loeve import KarhunenLoeve
from queens.schedulers.local import Local
from queens.utils.config_directories import experiment_directory
from queens.utils.io import load_result, read_file

_logger = logging.getLogger(__name__)


# Skip this test as it is no longer compatible with 4C, will be fixed with #176
pytestmark = pytest.mark.skip


class DummyKLField(KarhunenLoeve):
    """Dummy Karhunen-Loeve random field."""

    def expanded_representation(self, samples):
        """Dummy method for expansion."""
        return self.mean + self.std**2 * np.linalg.norm(self.coords["coords"], axis=1) * samples[0]


def test_write_random_material_to_dat(
    tmp_path,
    third_party_inputs,
    fourc_link_paths,
    expected_mean,
    expected_var,
    global_settings,
):
    """Test 4C with random field for material parameters."""
    dat_template = third_party_inputs / "fourc" / "coarse_plate_dirichlet_template.dat"

    dat_file_preprocessed = tmp_path / "coarse_plate_dirichlet_template.dat"

    fourc_executable, post_ensight, _ = fourc_link_paths

    fourc_input = dat_template
    fourc_input_preprocessed = dat_file_preprocessed

    # Parameters
    random_field_preprocessor = FourcDat(
        list_geometric_sets=["DSURFACE 1"],
        associated_material_numbers_geometric_set=[[10, 11]],
        random_fields=[
            {
                "name": "mat_param",
                "type": "material",
                "external_instance": "DSURFACE 1",
            }
        ],
        input_template=fourc_input,
        input_template_preprocessed=fourc_input_preprocessed,
    )
    random_field_preprocessor.main_run()
    random_field_preprocessor.write_random_fields_to_dat()
    mat_param = DummyKLField(
        corr_length=5.0,
        std=0.03,
        mean=0.25,
        explained_variance=0.95,
        coords=random_field_preprocessor.coords_dict["mat_param"],
    )
    parameters = Parameters(mat_param=mat_param)

    # Setup iterator
    external_geometry = FourcDat(
        list_geometric_sets=["DSURFACE 1"],
        input_template=fourc_input_preprocessed,
    )
    data_processor = EnsightFile(
        file_name_identifier="*_structure.case",
        file_options_dict={
            "delete_field_data": False,
            "geometric_target": ["geometric_set", "DSURFACE 1"],
            "physical_field_dict": {
                "vtk_field_type": "structure",
                "vtk_array_type": "point_array",
                "vtk_field_label": "displacement",
                "field_components": [0, 1],
            },
            "target_time_lst": ["last"],
        },
        external_geometry=external_geometry,
    )
    scheduler = Local(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    driver = Fourc(
        parameters=parameters,
        input_templates=fourc_input_preprocessed,
        executable=fourc_executable,
        post_processor=post_ensight,
        data_processor=data_processor,
    )
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = MonteCarlo(
        seed=1,
        num_samples=3,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    try:
        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))

        # Check if we got the expected results
        np.testing.assert_array_almost_equal(results["mean"], expected_mean, decimal=8)
        np.testing.assert_array_almost_equal(results["var"], expected_var, decimal=8)
    except Exception as error:
        experiment_dir = experiment_directory(global_settings.experiment_name)
        job_dir = experiment_dir / "0"
        _logger.info(list(job_dir.iterdir()))
        output_dir = job_dir / "output"
        _logger.info(list(output_dir.iterdir()))
        _logger.info(read_file(output_dir / "test_write_random_material_to_dat_0.log"))
        raise error


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Reference samples mean."""
    result = np.array(
        [
            [2.5, 2.5],
            [2.2498064041137695, 2.5],
            [2.2498133977254233, 2.2498133977254233],
            [2.5, 2.2498064041137695],
            [1.9997715950012207, 2.5],
            [1.9997826019922893, 2.2498329480489097],
            [1.7498218218485515, 2.5],
            [1.7498336633046467, 2.249856392542521],
            [1.4999066193898518, 2.5],
            [1.499914526939392, 2.2498757044474282],
            [1.25, 2.5],
            [1.25, 2.249883015950521],
            [1.0000933607419331, 2.5],
            [1.0000854929288228, 2.2498757044474282],
            [0.7501781582832336, 2.5],
            [0.7501662969589233, 2.249856392542521],
            [0.5002284049987793, 2.5],
            [0.5002173682053884, 2.2498329480489097],
            [0.2501935462156932, 2.5],
            [0.25018665194511414, 2.2498133977254233],
            [0.0, 2.5],
            [0.0, 2.2498064041137695],
            [2.2498329480489097, 1.9997826019922893],
            [2.5, 1.9997715950012207],
            [1.9998154640197754, 1.9998154640197754],
            [1.7498701413472493, 1.999857783317566],
            [1.4999394814173381, 1.9998946984608967],
            [1.25, 1.9999097188313801],
            [1.000060498714447, 1.9998946984608967],
            [0.7501298586527506, 1.999857783317566],
            [0.5001845061779022, 1.9998154640197754],
            [0.25016703208287555, 1.9997826019922893],
            [0.0, 1.9997715950012207],
            [2.249856392542521, 1.7498336633046467],
            [2.5, 1.7498218218485515],
            [1.999857783317566, 1.7498701413472493],
            [1.7499215205510457, 1.7499215205510457],
            [1.4999781052271526, 1.7499723037083943],
            [1.25, 1.7499951521555583],
            [1.0000218947728474, 1.7499723037083943],
            [0.7500784595807394, 1.7499215205510457],
            [0.5001422166824341, 1.7498701413472493],
            [0.25014352798461914, 1.7498336633046467],
            [0.0, 1.7498218218485515],
            [2.2498757044474282, 1.499914526939392],
            [2.5, 1.4999066193898518],
            [1.9998946984608967, 1.4999394814173381],
            [1.7499723037083943, 1.4999781052271526],
            [1.5000266234079997, 1.5000266234079997],
            [1.25, 1.5000547568003337],
            [0.9999733964602152, 1.5000266234079997],
            [0.7500277161598206, 1.4999781052271526],
            [0.5001053313414255, 1.4999394814173381],
            [0.25012436012427014, 1.499914526939392],
            [0.0, 1.4999066193898518],
            [2.249883015950521, 1.25],
            [2.5, 1.25],
            [1.9999097188313801, 1.25],
            [1.7499951521555583, 1.25],
            [1.5000547568003337, 1.25],
            [1.25, 1.25],
            [0.9999452233314514, 1.25],
            [0.7500048677126566, 1.25],
            [0.5000902911027273, 1.25],
            [0.25011690457661945, 1.25],
            [0.0, 1.25],
            [2.2498757044474282, 1.0000854929288228],
            [2.5, 1.0000933607419331],
            [1.9998946984608967, 1.000060498714447],
            [1.7499723037083943, 1.0000218947728474],
            [1.5000266234079997, 0.9999733964602152],
            [1.25, 0.9999452233314514],
            [0.9999733964602152, 0.9999733964602152],
            [0.7500277161598206, 1.0000218947728474],
            [0.5001053313414255, 1.000060498714447],
            [0.25012436012427014, 1.0000854929288228],
            [0.0, 1.0000933607419331],
            [2.249856392542521, 0.7501662969589233],
            [2.5, 0.7501781582832336],
            [1.999857783317566, 0.7501298586527506],
            [1.7499215205510457, 0.7500784595807394],
            [1.4999781052271526, 0.7500277161598206],
            [1.25, 0.7500048677126566],
            [1.0000218947728474, 0.7500277161598206],
            [0.7500784595807394, 0.7500784595807394],
            [0.5001422166824341, 0.7501298586527506],
            [0.25014352798461914, 0.7501662969589233],
            [0.0, 0.7501781582832336],
            [2.2498329480489097, 0.5002173682053884],
            [2.5, 0.5002284049987793],
            [1.9998154640197754, 0.5001845061779022],
            [1.7498701413472493, 0.5001422166824341],
            [1.4999394814173381, 0.5001053313414255],
            [1.25, 0.5000902911027273],
            [1.000060498714447, 0.5001053313414255],
            [0.7501298586527506, 0.5001422166824341],
            [0.5001845061779022, 0.5001845061779022],
            [0.25016703208287555, 0.5002173682053884],
            [0.0, 0.5002284049987793],
            [2.2498133977254233, 0.25018665194511414],
            [2.5, 0.2501935462156932],
            [1.9997826019922893, 0.25016703208287555],
            [1.7498336633046467, 0.25014352798461914],
            [1.499914526939392, 0.25012436012427014],
            [1.25, 0.25011690457661945],
            [1.0000854929288228, 0.25012436012427014],
            [0.7501662969589233, 0.25014352798461914],
            [0.5002173682053884, 0.25016703208287555],
            [0.25018665194511414, 0.25018665194511414],
            [0.0, 0.2501935462156932],
            [2.2498064041137695, 0.0],
            [2.5, 0.0],
            [1.9997715950012207, 0.0],
            [1.7498218218485515, 0.0],
            [1.4999066193898518, 0.0],
            [1.25, 0.0],
            [1.0000933607419331, 0.0],
            [0.7501781582832336, 0.0],
            [0.5002284049987793, 0.0],
            [0.2501935462156932, 0.0],
            [0.0, 0.0],
        ]
    )
    return result


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """Reference samples var."""
    result = np.array(
        [
            [0.0, 0.0],
            [4.526312369534935e-05, 0.0],
            [4.8868636061646e-05, 4.8868636061646e-05],
            [0.0, 4.526312369534935e-05],
            [9.116194934222221e-05, 0.0],
            [9.855069840133031e-05, 5.8283404617516986e-05],
            [7.674825360955614e-05, 0.0],
            [8.343726076039577e-05, 7.036472205375806e-05],
            [2.6622029527819297e-05, 0.0],
            [2.910567816627463e-05, 8.094860620152151e-05],
            [0.0, 0.0],
            [0.0, 8.523252804100898e-05],
            [2.662238345981412e-05, 0.0],
            [2.910530808127495e-05, 8.094860620152151e-05],
            [7.67479102812274e-05, 0.0],
            [8.343663418131086e-05, 7.036472205375806e-05],
            [9.116101417205869e-05, 0.0],
            [9.8551817331168e-05, 5.8283404617516986e-05],
            [4.526391435562734e-05, 0.0],
            [4.886853357799481e-05, 4.8868636061646e-05],
            [0.0, 0.0],
            [0.0, 4.526312369534935e-05],
            [5.8283404617516986e-05, 9.855069840133031e-05],
            [0.0, 9.116194934222221e-05],
            [0.0001188671531053842, 0.0001188671531053842],
            [0.00010267079992350622, 0.00014688445149602103],
            [3.653376410757876e-05, 0.00017303674629450447],
            [0.0, 0.0001840597232150761],
            [3.653417877913512e-05, 0.00017303674629450447],
            [0.0001026707026253367, 0.00014688445149602103],
            [0.00011886827494134167, 0.0001188671531053842],
            [5.828318029562259e-05, 9.855069840133031e-05],
            [0.0, 9.116194934222221e-05],
            [7.036472205375806e-05, 8.343726076039577e-05],
            [0.0, 7.674825360955614e-05],
            [0.00014688445149602103, 0.00010267079992350622],
            [0.0001320656893758117, 0.0001320656893758117],
            [4.9085635145236964e-05, 0.0001632383106529763],
            [0.0, 0.00017756617530058824],
            [4.9085635145236964e-05, 0.0001632383106529763],
            [0.0001320641124825291, 0.0001320656893758117],
            [0.00014688575668841963, 0.00010267079992350622],
            [7.036398242821207e-05, 8.343726076039577e-05],
            [0.0, 7.674825360955614e-05],
            [8.094860620152151e-05, 2.910567816627463e-05],
            [0.0, 2.6622029527819297e-05],
            [0.00017303674629450447, 3.653376410757876e-05],
            [0.0001632383106529763, 4.9085635145236964e-05],
            [6.656995740191482e-05, 6.656995740191482e-05],
            [0.0, 7.683283800948479e-05],
            [6.656939753331888e-05, 6.656995740191482e-05],
            [0.00016323880867119556, 4.9085635145236964e-05],
            [0.00017303810002318917, 3.653376410757876e-05],
            [8.094884984773584e-05, 2.910567816627463e-05],
            [0.0, 2.6622029527819297e-05],
            [8.523252804100898e-05, 0.0],
            [0.0, 0.0],
            [0.0001840597232150761, 0.0],
            [0.00017756617530058824, 0.0],
            [7.683283800948479e-05, 0.0],
            [0.0, 0.0],
            [7.68334395182535e-05, 0.0],
            [0.00017756669444087453, 0.0],
            [0.00018405925781224872, 0.0],
            [8.523275283274975e-05, 0.0],
            [0.0, 0.0],
            [8.094860620152151e-05, 2.910530808127495e-05],
            [0.0, 2.662238345981412e-05],
            [0.00017303674629450447, 3.653417877913512e-05],
            [0.0001632383106529763, 4.9085635145236964e-05],
            [6.656995740191482e-05, 6.656939753331888e-05],
            [0.0, 7.68334395182535e-05],
            [6.656939753331888e-05, 6.656939753331888e-05],
            [0.00016323880867119556, 4.9085635145236964e-05],
            [0.00017303810002318917, 3.653417877913512e-05],
            [8.094884984773584e-05, 2.910530808127495e-05],
            [0.0, 2.662238345981412e-05],
            [7.036472205375806e-05, 8.343663418131086e-05],
            [0.0, 7.67479102812274e-05],
            [0.00014688445149602103, 0.0001026707026253367],
            [0.0001320656893758117, 0.0001320641124825291],
            [4.9085635145236964e-05, 0.00016323880867119556],
            [0.0, 0.00017756669444087453],
            [4.9085635145236964e-05, 0.00016323880867119556],
            [0.0001320641124825291, 0.0001320641124825291],
            [0.00014688575668841963, 0.0001026707026253367],
            [7.036398242821207e-05, 8.343663418131086e-05],
            [0.0, 7.67479102812274e-05],
            [5.8283404617516986e-05, 9.8551817331168e-05],
            [0.0, 9.116101417205869e-05],
            [0.0001188671531053842, 0.00011886827494134167],
            [0.00010267079992350622, 0.00014688575668841963],
            [3.653376410757876e-05, 0.00017303810002318917],
            [0.0, 0.00018405925781224872],
            [3.653417877913512e-05, 0.00017303810002318917],
            [0.0001026707026253367, 0.00014688575668841963],
            [0.00011886827494134167, 0.00011886827494134167],
            [5.828318029562259e-05, 9.8551817331168e-05],
            [0.0, 9.116101417205869e-05],
            [4.8868636061646e-05, 4.886853357799481e-05],
            [0.0, 4.526391435562734e-05],
            [9.855069840133031e-05, 5.828318029562259e-05],
            [8.343726076039577e-05, 7.036398242821207e-05],
            [2.910567816627463e-05, 8.094884984773584e-05],
            [0.0, 8.523275283274975e-05],
            [2.910530808127495e-05, 8.094884984773584e-05],
            [8.343663418131086e-05, 7.036398242821207e-05],
            [9.8551817331168e-05, 5.828318029562259e-05],
            [4.886853357799481e-05, 4.886853357799481e-05],
            [0.0, 4.526391435562734e-05],
            [4.526312369534935e-05, 0.0],
            [0.0, 0.0],
            [9.116194934222221e-05, 0.0],
            [7.674825360955614e-05, 0.0],
            [2.6622029527819297e-05, 0.0],
            [0.0, 0.0],
            [2.662238345981412e-05, 0.0],
            [7.67479102812274e-05, 0.0],
            [9.116101417205869e-05, 0.0],
            [4.526391435562734e-05, 0.0],
            [0.0, 0.0],
        ]
    )
    return result
