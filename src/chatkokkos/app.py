import os
import sys
from datetime import datetime

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

import logging
logger = logging.getLogger(__name__)

from chatkokkos.utils import common_utils
from peft import LoraConfig, PeftModel

from pytz import timezone

import textwrap


class AppPreferences:
    def __init__(self, path=None):
        if path is None:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config/default_preferences.json"))
        self._config = common_utils.load_json_arg(path)

    @property
    def data_file(self) -> str:
        return self._config["data_file"]

    @data_file.setter
    def data_file(self, value) -> str:
        self._config["data_file"] = value

    @property
    def base_model_path(self) -> str:
        return self._config["base_model_path"]

    @base_model_path.setter
    def base_model_path(self, value) -> str:
        self._config["base_model_path"] = value

    @property
    def finetuned_model_path(self) -> str:
        return self._config["finetuned_model_path"]

    @finetuned_model_path.setter
    def finetuned_model_path(self, value) -> str:
        self._config["finetuned_model_path"] = value

    @property
    def merged_model_path(self) -> str:
        return self._config["merged_model_path"]

    @merged_model_path.setter
    def merged_model_path(self, value) -> str:
        self._config["merged_model_path"] = value

    def update(self, config):
        self._config.update(config)

    def update_with_json_file(self, json_path):
        self._config.update(common_utils.load_json_arg(json_path))

    def __str__(self):
        return str(self._config)


class App:
    def __init__(self, config_file=None):
        """Initialize the Application object."""
        self.preferences = AppPreferences(config_file)

    def load_base_model(self) -> None:
        """Load model from base path."""

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
        """Load model from finetuned path."""

        logger.info("Loading the finetuned model from %s", self.preferences.finetuned_model_path)

        self.load_base_model()

        self.model = PeftModel.from_pretrained(self.model, self.preferences.finetuned_model_path)

    def load_merged_model(self) -> None:
        """Load model from merged path."""

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
        """Load Datasets from memory"""
        logger.info("Loading the dataset from %s", self.preferences.data_file)

        from datasets import load_dataset
        self.train_dataset = load_dataset(
            "json", data_files=self.preferences.data_file, split="train"
        )
        self.eval_dataset = load_dataset(
            "json", data_files=self.preferences.data_file, split="train"
        )

    def evaluate_model(self, prompt:str, max_new_tokens:int=800) -> str:
        """Evaluate the model on a prompt.

        Args:
            prompt (str): _description_
            max_new_tokens (int, optional): _description_. Defaults to 800.

        Returns:
            _type_: _description_
        """
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**model_input, max_new_tokens=max_new_tokens)[0]
            return self.tokenizer.decode(output)

    @staticmethod
    def chatkokkos_prompt(question:str, context:str) -> str:
        """Create a prompt from the input."""
        return textwrap.dedent(f"""\
            You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

            You must output the Kokkos question that answers the question.

            ### Input:
            {question}

            ### Context:
            {context}

            ### Response:
            """)

    def chatkokkos_evaluate(self, question, context, **kwargs):
        prompt = self.chatkokkos_prompt(question, context)
        return self.evaluate_model(prompt, **kwargs)

    def tokenize_training_set(self):
        """Tokenize the Training Set.

        Requires:
            Need to have ran App.load_datasets() first.

        Returns:
            self.tokenized_train_dataset
            self.tokenized_val_dataset
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
        """Train the finetunning layers

        Requires:
            Need to have ran App.load_datasets() and App.load_base_model() first.

        Returns:
            self.model
            Saves finetuned model to preferences.finetuned_model_path
            Saves merged model to preferences.merged_model_path
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
        print(self.preferences)

