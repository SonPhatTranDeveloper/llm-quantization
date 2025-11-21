import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import print_text_box

logger = logging.getLogger(__name__)


def log_generation_stats(
    num_tokens: int, generation_time: float, tokens_per_second: float
) -> None:
    """
    Log generation statistics in a boxed format.

    Args:
        num_tokens: Number of tokens generated
        generation_time: Time taken for generation in seconds
        tokens_per_second: Tokens per second throughput
    """
    stats_text = f"{num_tokens:,} tokens │ {generation_time:.3f}s │ {tokens_per_second:.2f} tokens/s"
    logger.info("")
    print_text_box(stats_text, width=64, title="Generation Statistics")
    logger.info("")


class TextGenerator:
    """
    A class for generating text using a pre-trained language model and tokenizer.

    This class handles device placement automatically and follows HuggingFace
    best practices for text generation.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        """
        Initialize the TextGenerator with a model and tokenizer.

        Args:
            model: The pre-trained causal language model for text generation
            tokenizer: The tokenizer associated with the model
        """
        self.model = model
        self.tokenizer = tokenizer

        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """
        Generate text from a given prompt.

        Args:
            prompt: The input text prompt to generate from
            max_new_tokens: Maximum number of new tokens to generate (default: 50)
            temperature: Sampling temperature for controlling randomness.
                         Higher values make output more random (default: 1.0)
            top_p: Nucleus sampling threshold. Tokens with cumulative probability
                   above this threshold are considered (default: 1.0)
            top_k: Top-k sampling. Only consider the top k most likely tokens (default: 50)

        Returns:
            The newly generated text (without the input prompt) as a string
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Detect model device and move inputs to the same device
        # Try to get device from model attribute first, then from parameters
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate text with sampling parameters
        # Use do_sample=True when using temperature, top_p, or top_k
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generation_time = time.perf_counter() - start_time

        # Extract only the newly generated tokens (excluding the input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        # Calculate tokens per second
        num_tokens = len(generated_tokens)
        tokens_per_second = num_tokens / generation_time
        log_generation_stats(num_tokens, generation_time, tokens_per_second)

        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text
