import logging
import hydra
from omegaconf import DictConfig
from src.models import load_model, TextGenerator
from src.utils.logging import setup_logging, format_text_box

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../config", config_name="text_generation", version_base=None
)
def main(cfg: DictConfig) -> None:
    """
    Main function for text generation using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing model and generation parameters
    """
    model, tokenizer = load_model(cfg)
    generator = TextGenerator(model, tokenizer)

    generated_text = generator.generate(
        prompt=cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
    )

    logger.info("Generation complete")
    logger.info("")
    logger.info(format_text_box(generated_text, title="Generated Text"))
    logger.info("")


if __name__ == "__main__":
    main()
