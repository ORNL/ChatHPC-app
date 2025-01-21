"""App Module: used to construct an app for training ChatHPC LLMs."""
from __future__ import annotations

import logging
import os
import sys
import textwrap
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
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict
from pytz import timezone
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

logger = logging.getLogger(__name__)

DEFAULT_APP_CONFIG_FILE = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "config/default_app_settings.json")))
class AppConfig(BaseSettings):
    """Configuration settings for the application.

    This class inherits from [Pydantic Settings - BaseSettings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) and defines the configuration
    parameters for the ChatKokkos application.

    Attributes:
        data_file (str): Path to the training data file.
        base_model_path (str): Path to the base LLM model.
        finetuned_model_path (str): Path to the finetuned LLM layers.
        merged_model_path (str): Path to the merged LLM model.
    """
    data_file: str = "init"
    base_model_path: str = "init"
    finetuned_model_path: str = "init"
    merged_model_path: str = "init"

    model_config = SettingsConfigDict(
        cli_parse_args=False,
        env_prefix="CHATKOKKOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        json_file=DEFAULT_APP_CONFIG_FILE,
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
        return (env_settings, dotenv_settings, init_settings, JsonConfigSettingsSource(settings_cls), file_secret_settings)


class App:
    """Main application class for ChatKokkos.

    This class handles the initialization, loading, and management of models,
    datasets, and training processes for the ChatKokkos application. It provides
    methods for loading different types of models, evaluating prompts, and
    fine-tuning the model.

    Attributes:
        preferences (AppConfig): Configuration settings for the application.
        tokenizer: Tokenizer for processing input text.
        model: The language model used for text generation and fine-tuning.
        train_dataset: Dataset used for training.
        eval_dataset: Dataset used for evaluation.
    """
    def __init__(self):
        """Initialize the Application object."""
        self.preferences = AppConfig()

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

        logger.info("Loading the base model from %s", self.preferences.base_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.preferences.base_model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.preferences.base_model_path,
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

        logger.info("Loading the finetuned model from %s", self.preferences.finetuned_model_path)

        self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, self.preferences.finetuned_model_path)

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

        logger.info("Loading the merged model from %s", self.preferences.merged_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.preferences.base_model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.preferences.merged_model_path,
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
        logger.info("Loading the dataset from %s", self.preferences.data_file)

        from datasets import load_dataset
        self.train_dataset = load_dataset(
            "json", data_files=self.preferences.data_file, split="train"
        )
        self.eval_dataset = load_dataset(
            "json", data_files=self.preferences.data_file, split="train"
        )

    def evaluate_model(self, prompt:str, max_new_tokens:int=800) -> str:
        """Evaluate the model on a given prompt and generate a response.

        Args:
            prompt (str): The input text prompt to be evaluated by the model.
            max_new_tokens (int, optional): Maximum number of tokens to generate in the response.
                Defaults to 800.

        Returns:
            str: The generated text response from the model, decoded from output tokens.

        Note:
            This method requires the model and tokenizer to be loaded first through either
            load_base_model(), load_finetuned_model(), or load_merged_model().
        """
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**model_input, max_new_tokens=max_new_tokens)[0]
            return self.tokenizer.decode(output)

    @staticmethod
    def chatkokkos_prompt(question:str, context:str) -> str:
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
            >>> app.chatkokkos_prompt("How do I use Views?", "Views are memory spaces in Kokkos...")
            "You are a powerful LLM model for Kokkos..."
            ```
        """
        return textwrap.dedent(f"""\
            You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

            You must output the Kokkos question that answers the question.

            ### Input:
            {question}

            ### Context:
            {context}

            ### Response:
            """)

    def chatkokkos_evaluate(self, question: str, context: str, **kwargs: dict[str, Any]) -> str:
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
            >>> response = app.chatkokkos_evaluate(
            ...     "How do I create a 2D View?",
            ...     "Kokkos::View is a multidimensional array class"
            ... )
            >>> print(response)
            "To create a 2D Kokkos View..."
            ```
        """
        prompt = self.chatkokkos_prompt(question, context)
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
            full_prompt = textwrap.dedent(f"""\
                You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

                You must output the Kokkos question that answers the question.

                ### Input:
                {data_point["question"]}

                ### Context:
                {data_point["context"]}

                ### Response:
                {data_point["answer"]}
                """)
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
        output_dir = "kokkos-code-llama"

        # resume_from_checkpoint = os.path.join(base_model_path, "pytorch_model-00001-of-00003.bin")

        # if resume_from_checkpoint:
        #     if os.path.exists(resume_from_checkpoint):
        #         print(f"Restarting from {resume_from_checkpoint}")
        #         adapters_weights = torch.load(resume_from_checkpoint)
        #         set_peft_model_state_dict(self.model, adapters_weights)
        #     else:
        #         print(f"Checkpoint {resume_from_checkpoint} not found")


        wandb_project = "ChatKokkos"
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
            report_to="wandb",  # if use_wandb else "none",
            run_name=f"codellama-{datetime.now(tz=timezone('EST')).strftime('%Y-%m-%d-%H-%M')}",  # if use_wandb else None,
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
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

        trainer.model.save_pretrained(self.preferences.finetuned_model_path)
        self.model = trainer.model.merge_and_unload()
        self.model.save_pretrained(self.preferences.merged_model_path)


    def print_preferences(self) -> None:
        """Print the current preferences of the application.

        This method displays all the preferences stored in the self.preferences
        attribute, which typically includes configuration settings for the
        application such as model paths, dataset locations, and other parameters.

        Note:
            The output format depends on the __str__ implementation of the
            Preferences class.
        """
        print(self.preferences)

