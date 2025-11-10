from queens.models import Simulation
from queens.schedulers import Cluster

first_model = Simulation.scheduler_function_from_simulation_code(
    parameters_1, driver_1, data_processor_1
)


def sequential_execution(
    sample: np.ndarray,
    job_id: int,
    job_dir: Path,
    num_procs: int,
    experiment_dir: Path,
    experiment_name: str,
):

    metadata = SimulationMetadata(job_id, sample, job_dir, file_name="job_metadata")

    # Make job dir
    job_dir.mkdir(exist_ok=True, parents=True)

    # Run first model
    with metadata.time_code("run_first_model"):
        results_model_1 = first_model(
            sample,
            job_id,
            job_dir / "model_1",
            num_procs,
            experiment_dir,
            experiment_name,
        )

    # Run second model
    with metadata.time_code("run_second_model"):
        driver_2.run(
            results_model_1,
            job_id,
            job_dir / "model_2",
            num_procs=1,
            experiment_dir=experiment_dir,
            experiment_name=experiment_dir,
        )

    # Extract the data if needed
    data = {
        "model_1": results_model_1,
        "results_model_2": data_processor_model_2(job_dir / "model_2"),
    }
    metadata.outputs = data

    return data


sequential_model = Simulation(scheduler, sequential_execution)
