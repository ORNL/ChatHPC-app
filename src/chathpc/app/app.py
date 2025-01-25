"""App Module: used to construct an app for training ChatHPC LLMs."""

from __future__ import annotations

import atexit
import logging
import os
import readline
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict
from pytz import timezone
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from chathpc.app.utils.common_utils import evaluate_fstring

logger = logging.getLogger(__name__)

DEFAULT_APP_CONFIG_FILE = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "config/default_app_settings.json"))
)


class AppConfig(BaseSettings):
    """Configuration settings for the application.

    This class inherits from [Pydantic Settings - BaseSettings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
    and defines the configuration parameters for the ChatHPC application.

    Attributes:
        data_file (str): Path to the JSON file containing training data for model fine-tuning.
        base_model_path (str): Path to the pre-trained base LLM model directory.
        finetuned_model_path (str): Path where fine-tuned model layers will be saved.
        merged_model_path (str): Path where the complete merged model will be saved.
        max_response_tokens (int): Maximum number of tokens to generate in model responses.

    Configuration:
        The settings can be loaded from:
        - Environment variables with prefix 'CHATKOKKOS_'
        - .env file
        - JSON configuration file (default: config/default_app_settings.json)
        - Direct initialization

    Example:
        ```python
        config = AppConfig()
        print(config.base_model_path)
        ```

    Note:
        Settings priority follows Pydantic's source order: explicite values in constructor > env vars > .env file >
        config file > defaults
    """

    data_file: Path = Field(..., description="Path to the JSON file containing training data for model fine-tuning.")
    base_model_path: Path = Field("/auto/projects/ChatHPC/models/cache/meta-llama/CodeLlama-7b-hf", description= "Path to the pre-trained base LLM model directory.")
    finetuned_model_path: Path = Field("./peft_adapter", description="Path where fine-tuned model layers will be saved.")
    merged_model_path: Path = Field("./merged_adapters", description="Path where the complete merged model will be saved.")
    training_output_dir: Path = Field("./training_checkpoints", description="Path where training output will be saved.")
    max_response_tokens: int = Field(600, gt=0, description="Maximum number of tokens to generate in model responses.")
    prompt_history_file: Path = Field("~/.chathpc_history", description="Path to the file containing interactive prompt history.")
    training_prompt: str = Field(..., description="Prompt template to use for training.")
    inference_prompt: str = Field(..., description="Prompt template to use for inference.")
    use_wandb: bool = Field(False, description="Whether to use Weights & Biases for logging.")

    model_config = SettingsConfigDict(
        # cli_parse_args=True,
        env_prefix="CHATHPC_",
        env_file=".env",
        env_file_encoding="utf-8",
        # json_file=DEFAULT_APP_CONFIG_FILE,
        json_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            dotenv_settings,
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


class App:
    """Main application class for ChatHPC Application.

    This class handles the initialization, loading, and management of models,
    datasets, and training processes for the ChatHPC application. It provides
    methods for loading different types of models, evaluating prompts, and
    fine-tuning the model.

    Attributes:
        preferences (AppConfig): Configuration settings for the application.
        tokenizer: Tokenizer for processing input text.
        model: The language model used for text generation and fine-tuning.
        train_dataset: Dataset used for training.
        eval_dataset: Dataset used for evaluation.
    """

    def __init__(self, app_config: AppConfig=None):
        """Initialize the Application object."""
        if app_config is None:
            app_config = AppConfig()

        self.config = app_config

    def load_base_model(self) -> None:
        """Load and initialize the base Large Language Model.

        This method initializes both the tokenizer and model from the base model path
        specified in the application preferences. The model is loaded with specific
        configurations for optimal performance.

        Requires:
            - preferences.base_model_path must be set to a valid model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized AutoModelForCausalLM in float16 precision

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/model"
            >>> app.load_base_model()
            ```

        Note:
            The model is loaded with float16 precision and automatic device mapping
            for optimal performance on available hardware.
        """

        logger.info("Loading the base model from %s", self.config.base_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            # device_map={'':torch.cuda.current_device()}
        )

    def load_finetuned_model(self) -> None:
        """Load and initialize the finetuned Large Language Model.

        This method loads a finetuned model by first initializing the base model and tokenizer,
        then loading the finetuned layers on top of it using PeftModel.

        Requires:
            - preferences.base_model_path must be set to a valid base model path
            - preferences.finetuned_model_path must be set to a valid finetuned model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized PeftModel with finetuned layers

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/base/model"
            >>> app.preferences.finetuned_model_path = "path/to/finetuned/model"
            >>> app.load_finetuned_model()
            ```

        Note:
            This method first calls load_base_model() to initialize the foundation model
            before applying the finetuned layers.
        """

        logger.info("Loading the finetuned model from %s", self.config.finetuned_model_path)

        self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, self.config.finetuned_model_path)

    def load_merged_model(self) -> None:
        """Load and initialize the merged Large Language Model.

        This method loads a complete merged model that combines the base model with
        finetuned layers into a single model file. The tokenizer is initialized from
        the base model path while the full model is loaded from the merged model path.

        Requires:
            - preferences.base_model_path must be set to a valid base model path for tokenizer
            - preferences.merged_model_path must be set to a valid merged model path

        Sets:
            - self.tokenizer: Initialized AutoTokenizer for text processing
            - self.model: Initialized AutoModelForCausalLM with merged weights

        Example:
            ```python
            >>> app = App()
            >>> app.preferences.base_model_path = "path/to/base/model"
            >>> app.preferences.merged_model_path = "path/to/merged/model"
            >>> app.load_merged_model()
            ```

        Note:
            The model is loaded with float16 precision and automatic device mapping
            for optimal performance on available hardware.
        """

        logger.info("Loading the merged model from %s", self.config.merged_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.merged_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            # device_map={'':torch.cuda.current_device()}
        )

    def load_datasets(self) -> None:
        """Load training and evaluation datasets from a JSON file.

        This method loads datasets from the JSON file specified in the application preferences.
        The datasets are loaded using the Hugging Face datasets library and split into
        training and evaluation sets.

        Config:
            preferences.data_file (str): Path to the JSON file containing the datasets.

        Sets:
            self.train_dataset: Dataset object for training
            self.eval_dataset: Dataset object for evaluation

        Requires:
            - The data file must be in JSON format
            - The data file path must be set in preferences.data_file
        """
        logger.info("Loading the dataset from %s", self.config.data_file)

        from datasets import load_dataset

        self.train_dataset = load_dataset("json", data_files=self.config.data_file.as_posix(), split="train")
        self.eval_dataset = load_dataset("json", data_files=self.config.data_file.as_posix(), split="train")

    def evaluate_model(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Evaluate the model on a given prompt and generate a response.

        Args:
            prompt (str): The input text prompt to be evaluated by the model.
            max_new_tokens (int|None, optional): Maximum number of tokens to generate in the response.
                If None, uses the value from config.max_response_tokens.

        Returns:
            str: The generated text response from the model, decoded from output tokens.

        Requires:
            - A model must be loaded via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - The tokenizer must be initialized

        Example:
            ```python
            >>> app = App()
            >>> app.load_base_model()
            >>> response = app.evaluate_model(
            ...     "What is Kokkos?",
            ...     max_new_tokens=100
            ... )
            >>> print(response)
            "Kokkos is a programming model..."
            ```

        Note:
            The model is automatically put into evaluation mode and uses
            torch.no_grad() for inference. The input is processed on CUDA
            if available.
        """
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        if max_new_tokens is None:
            max_new_tokens = self.config.max_response_tokens

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                **model_input, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
            )[0]
            return self.tokenizer.decode(output)

    def chat_prompt(self, question: str, context: str) -> str:
        """Create a formatted prompt for Kokkos-related questions.

        This method generates a structured prompt that includes the question and context
        for the Kokkos programming model queries. The prompt follows a specific format
        that instructs the model about its role and expected output.

        Args:
            question (str): The question about Kokkos to be answered.
            context (str): Additional context or information related to the question.

        Returns:
            str: A formatted prompt string containing the question and context with
                 appropriate instructions for the model.

        Example:
            ```python
            >>> app.chat_prompt("How do I use Views?", "Views are memory spaces in Kokkos...")
            "You are a powerful LLM model for Kokkos..."
            ```
        """
        return evaluate_fstring(self.config.inference_prompt, question=question, context=context)

    def chat_evaluate(self, question: str, context: str, **kwargs: dict[str, Any]) -> str:
        """Evaluate a Kokkos-related question with provided context.

        This method processes a Kokkos-related question by combining it with context
        into a formatted prompt and generating a response using the loaded model.

        Args:
            question (str): The question about Kokkos programming model to be answered.
            context (str): Supporting context or documentation related to the question.
            **kwargs (dict[str, Any]): Additional keyword arguments passed to evaluate_model(),
                such as max_new_tokens.

        Returns:
            respose (str): The model-generated response addressing the Kokkos question.

        Requires:
            - A model must be loaded via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - The tokenizer must be initialized

        Example:
            ```
            >>> app = App()
            >>> app.load_base_model()
            >>> response = app.chat_evaluate(
            ...     "How do I create a 2D View?",
            ...     "Kokkos::View is a multidimensional array class"
            ... )
            >>> print(response)
            "To create a 2D Kokkos View..."
            ```
        """
        prompt = self.chat_prompt(question, context)
        return self.evaluate_model(prompt, **kwargs)

    def tokenize_training_set(self) -> None:
        """Tokenize the training and validation datasets.

        This method processes the loaded datasets by tokenizing the text data using the model's
        tokenizer. It creates formatted prompts combining questions, context, and answers, then
        tokenizes them for model training.

        Requires:
            - The datasets must be loaded first through App.load_datasets()
            - A tokenizer must be initialized through loading a model

        Sets:
            - self.tokenized_train_dataset: Tokenized dataset for training
            - self.tokenized_val_dataset: Tokenized dataset for validation

        Note:
            This method also handles padding token configuration and adds/removes EOS tokens
            as needed for the tokenization process.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        def tokenize(prompt):
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None,
            )

            # "self-supervised learning" means the labels are also the inputs:
            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point):
            full_prompt = evaluate_fstring(self.config.training_prompt, **data_point)
            return tokenize(full_prompt)

        self.tokenizer.add_eos_token = True

        self.tokenized_train_dataset = self.train_dataset.map(generate_and_tokenize_prompt)
        self.tokenized_val_dataset = self.eval_dataset.map(generate_and_tokenize_prompt)

        self.tokenizer.add_eos_token = False

    def train(self):
        """Train the model using fine-tuning layers.

        This method performs fine-tuning of the base model using LoRA (Low-Rank Adaptation)
        configuration. It prepares the model for training, sets up training arguments,
        and executes the training process.

        Requires:
            - App.load_datasets() must be called first to load training data
            - App.load_base_model() must be called first to load the base model
            - Tokenizer and model must be properly initialized

        Sets:
            - self.peft_config: LoRA configuration for fine-tuning
            - self.training_args: Training arguments for the Trainer
            - self.model: Updated model after training

        Saves:
            - Finetuned model layers to preferences.finetuned_model_path
            - Complete merged model to preferences.merged_model_path

        Note:
            This method uses Hugging Face's Trainer for the training process and
            supports multi-GPU training when available. It also integrates with
            Weights & Biases (wandb) for experiment tracking.
        """

        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )
        self.model.train()  # put model back into training mode
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

        batch_size = 128
        per_device_train_batch_size = 32
        gradient_accumulation_steps = batch_size // per_device_train_batch_size
        output_dir = self.config.training_output_dir

        # resume_from_checkpoint = os.path.join(base_model_path, "pytorch_model-00001-of-00003.bin")

        # if resume_from_checkpoint:
        #     if os.path.exists(resume_from_checkpoint):
        #         print(f"Restarting from {resume_from_checkpoint}")
        #         adapters_weights = torch.load(resume_from_checkpoint)
        #         set_peft_model_state_dict(self.model, adapters_weights)
        #     else:
        #         print(f"Checkpoint {resume_from_checkpoint} not found")

        wandb_project = "ChatHPC"
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project

        if torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            print("multiple gpus detected!")
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        self.training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            max_steps=400,
            # max_steps=20,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps",  # if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=20,
            save_steps=20,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=False,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=True,  # group sequences of roughly the same length together to speed up training
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=f"codellama-{datetime.now(tz=timezone('EST')).strftime('%Y-%m-%d-%H-%M')}",  # if use_wandb else None,
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

        self.model.config.use_cache = False

        # old_state_dict = model.state_dit
        # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        #     model, type(model)
        # )

        if torch.__version__ >= "2" and sys.platform != "win32":
            print("compiling the model")
            self.model = torch.compile(self.model)

        # model.to('cuda')

        trainer.train()

        trainer.model.save_pretrained(self.config.finetuned_model_path)
        self.model = trainer.model.merge_and_unload()
        self.tokenizer.save_pretrained(self.config.merged_model_path)
        self.model.save_pretrained(self.config.merged_model_path)

    def interactive(self, prompt="chathpc") -> None:
        """Start an interactive chat session with the model.

        This method provides a command-line interface for interacting with the model.
        It maintains a command history and supports context setting for conversations.

        Commands:
            /bye: Exit the interactive session
            /context: Set a new context for subsequent questions

        Args:
            prompt (str, optional): The prompt prefix to display. Defaults to "chathpc_app".

        Requires:
            - A model must be loaded via one of:
                - load_base_model()
                - load_finetuned_model()
                - load_merged_model()
            - The tokenizer must be initialized

        Example:
            ```python
            >>> app = App()
            >>> app.load_merged_model()
            >>> app.interactive()
            chathpc_app ()> What is Kokkos?
        """
        history_file = Path(self.config.prompt_history_file).expanduser()
        try:
            readline.read_history_file(history_file)
            h_len = readline.get_current_history_length()
        except FileNotFoundError:
            open(history_file, "wb+").close()
            readline.add_history("/context")
            readline.add_history("/bye")
            h_len = readline.get_current_history_length()

        def save_history(prev_h_len, histfile):
            new_h_len = readline.get_current_history_length()
            readline.set_history_length(1000)
            readline.append_history_file(new_h_len - prev_h_len, histfile)

        atexit.register(save_history, h_len, history_file)

        context = ""
        print("Use '/bye' to exit.\nUse '/context' to set context.")
        while True:
            user_input = input(f"{prompt} ({context})> ")
            if user_input == "/bye":
                break
            if user_input == "/context":
                context = input("Context: ")
                continue
            print(self.chat_evaluate(user_input, context))

    def print_config(self) -> None:
        """Print the current configurations of the application in a formatted table.

        This method displays all configuration settings from self.preferences in a
        formatted table using the tabulate library. The output includes paths for:
        - Data files
        - Base model
        - Finetuned model layers
        - Merged model

        Example:
            >>> app = App()
            >>> app.print_config()
            =====================  ==========================
            Setting               Value
            =====================  ==========================
            data_file             /path/to/data.json
            base_model_path       /path/to/base/model
            finetuned_model_path  /path/to/finetuned/model
            merged_model_path     /path/to/merged/model
            =====================  ==========================
        """
        # Get configuration as dict, excluding internal pydantic fields
        config_dict = self.config.model_dump()

        # Format as table rows
        table_data = [[setting, value] for setting, value in config_dict.items()]

        # Define table headers
        headers = ["Setting", "Value"]

        # Print formatted table
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
