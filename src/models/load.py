from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def create_bits_and_bytes_config(quantization: int) -> BitsAndBytesConfig:
    """
    Create the bits and bytes configuration for the model
    """
    if quantization == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )
    raise ValueError(f"Invalid quantization bits: {quantization}")


def load_model(config: DictConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model from the configuration
    The configuration is expected to be a dictionary with the following keys:
    - model_name: The name of the model to load
    - quantization: The quantization bits to use (4 or 8)

    Args:
        config: The configuration for the model

    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    model_name = config.model_name
    quantization = config.quantization

    # Load model
    bits_and_bytes_config = create_bits_and_bytes_config(quantization)

    if quantization is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bits_and_bytes_config, device_map="auto"
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
