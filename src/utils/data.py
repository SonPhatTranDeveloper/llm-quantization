import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """
    Preprocesses text data for causal language modeling.

    Handles loading text from file, tokenization, chunking, and
    creating train/validation splits.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        block_size: int = 512,
        train_split: float = 0.9,
    ) -> None:
        """
        Initialize the DatasetPreprocessor.

        Args:
            tokenizer: The tokenizer to use for tokenization
            block_size: Maximum sequence length for chunks
            train_split: Fraction of data to use for training (rest for validation)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train_split = train_split

    def load_text(self, data_path: str) -> str:
        """
        Load text data from a file.

        Args:
            data_path: Path to the text file

        Returns:
            The text content as a string
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info(f"Loaded {len(text):,} characters from {data_path}")
        return text

    def tokenize_and_chunk(self, text: str) -> list[dict[str, list[int]]]:
        """
        Tokenize text and split into chunks of block_size.

        Args:
            text: The input text to tokenize and chunk

        Returns:
            List of tokenized chunks, each containing input_ids
        """
        tokenizer = self.tokenizer

        # Tokenize the entire text
        tokenized = tokenizer(text, return_overflowing_tokens=False)
        input_ids = tokenized["input_ids"]

        # Split into chunks of block_size
        chunks = []
        for i in range(0, len(input_ids), self.block_size):
            chunk = input_ids[i : i + self.block_size]
            if len(chunk) == self.block_size:
                chunks.append({"input_ids": chunk})

        logger.info(
            f"Tokenized text into {len(chunks)} chunks of size {self.block_size}"
        )
        return chunks

    def prepare_dataset(self, data_path: str) -> tuple[Dataset, Dataset]:
        """
        Load text, tokenize, chunk, and create train/validation splits.

        Args:
            data_path: Path to the text file

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        text = self.load_text(data_path)
        chunks = self.tokenize_and_chunk(text)

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(chunks)

        # Split into train and validation
        split_dataset = dataset.train_test_split(train_size=self.train_split, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logger.info(
            f"Created train dataset with {len(train_dataset)} samples "
            f"and eval dataset with {len(eval_dataset)} samples"
        )

        return train_dataset, eval_dataset
