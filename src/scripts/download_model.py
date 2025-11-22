import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def download_model(model_name: str, folder_name: str) -> None:
    """
    Download a model from Hugging Face Hub to a specified folder.

    Args:
        model_name: The name or identifier of the model on Hugging Face Hub
        folder_name: The local folder path where the model should be downloaded
    """
    folder_path = Path(folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model '{model_name}' to '{folder_path.absolute()}'")

    snapshot_download(
        repo_id=model_name,
        local_dir=str(folder_path),
        local_dir_use_symlinks=False,
    )

    logger.info(
        f"Successfully downloaded model '{model_name}' to '{folder_path.absolute()}'"
    )


def main() -> None:
    """
    Main function to parse arguments and download the model.
    """
    parser = argparse.ArgumentParser(
        description="Download a model from Hugging Face Hub to a local folder"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The name or identifier of the model on Hugging Face Hub (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "folder_name",
        type=str,
        help="The local folder path where the model should be downloaded",
    )

    args = parser.parse_args()
    download_model(args.model_name, args.folder_name)


if __name__ == "__main__":
    main()
