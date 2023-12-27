import os
import json
import hydra
import mlflow
import tempfile
from omegaconf import DictConfig

_ALL_STEP = [
    'data_cleaning'
]


@hydra.main(config_name='hydra_config')
def run(config: DictConfig):

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _ALL_STEP

    with tempfile.TemporaryDirectory() as _:

        root_path = hydra.utils.get_original_cwd()

        if "data_cleaning" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, "src/data_clean"),
                "main",
                parameters={
                    "input_data_path": config["data_path"]["input_path"],
                    "output_data_path": config['data_path']["output_path"]
                },
            )


if __name__ == "__main__":
    run()
