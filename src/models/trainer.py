import logging
from typing import Optional

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.models.load import load_model
from src.utils.data import DatasetPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training of language models with LoRA/QLoRA support.

    Supports both full fine-tuning and parameter-efficient LoRA training
    with quantized models (QLoRA).
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the ModelTrainer.

        Args:
            config: Configuration dictionary containing model, training, and LoRA settings
        """
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trainer: Optional[Trainer] = None

    def load_model_for_training(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer for training.

        For training, we need to enable gradient computation even with quantization.

        Returns:
            Tuple of (model, tokenizer)
        """
        model, tokenizer = load_model(self.config)

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Apply LoRA adapters to the model.

        Args:
            model: The base model to apply LoRA to

        Returns:
            Model with LoRA adapters applied
        """
        lora_config_dict = self.config.lora

        # Determine target modules based on model architecture
        # Convert ListConfig to regular list for JSON serialization
        target_modules = OmegaConf.to_container(
            lora_config_dict.target_modules, resolve=True
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config_dict.r,
            lora_alpha=lora_config_dict.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_config_dict.lora_dropout,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Set model to training mode to ensure gradients are enabled
        model.train()

        logger.info("Applied LoRA adapters to model")
        return model

    def prepare_training(
        self,
    ) -> tuple[Dataset, Dataset, DataCollatorForLanguageModeling]:
        """
        Prepare datasets and data collator for training.

        Returns:
            Tuple of (train_dataset, eval_dataset, data_collator)
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError(
                "Tokenizer not loaded. Call load_model_for_training first."
            )

        preprocessor = DatasetPreprocessor(
            tokenizer=tokenizer,
            block_size=self.config.block_size,
            train_split=self.config.train_split,
        )

        train_dataset, eval_dataset = preprocessor.prepare_dataset(self.config.data)

        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return train_dataset, eval_dataset, data_collator

    def create_training_arguments(self) -> TrainingArguments:
        """
        Create TrainingArguments from configuration.

        Returns:
            TrainingArguments object
        """
        training_config = self.config.training

        training_args = TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            lr_scheduler_type=training_config.lr_scheduler_type,
            optim=training_config.optim,
            max_grad_norm=training_config.max_grad_norm,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            save_total_limit=training_config.save_total_limit,
            eval_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
        )

        return training_args

    def train(self) -> None:
        """
        Execute the training process.

        Loads model, applies LoRA if enabled, prepares datasets,
        and runs training.
        """
        logger.info("Starting training process")

        # Load model and tokenizer
        model, _ = self.load_model_for_training()

        # Apply LoRA
        self.model = self.apply_lora(model)

        # Prepare datasets
        train_dataset, eval_dataset, data_collator = self.prepare_training()

        # Create training arguments
        training_args = self.create_training_arguments()

        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Beginning training")
        self.trainer.train()

        # Save full model
        self.save_model(training_args.output_dir)
        logger.info("Training completed successfully")

    def save_model(self, output_dir: str) -> None:
        """
        Save the full model by merging LoRA adapters into the base model.

        This method merges the trained LoRA adapters with the base model
        and saves the complete merged model. This is useful when you want
        to save a standalone model without needing to load the base model
        and adapters separately.

        Args:
            output_dir: Directory to save the full model.
        """
        # Merge LoRA adapters into base model
        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()

        # Save model
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Full model saved successfully")
