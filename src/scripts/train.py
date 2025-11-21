import logging
import hydra
from omegaconf import DictConfig
from src.models.trainer import ModelTrainer
from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="training", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for training using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing model, training, and LoRA parameters
    """
    trainer = ModelTrainer(cfg)
    trainer.train()

    logger.info("Training script completed")


if __name__ == "__main__":
    main()
