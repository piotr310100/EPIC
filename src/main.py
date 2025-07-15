import hydra
from hydra.utils import call
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    call(config.exp.run_func, config)


if __name__ == "__main__":
    main()
