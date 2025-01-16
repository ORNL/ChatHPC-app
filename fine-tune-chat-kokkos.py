# %% [markdown]
# # Fine-tunning ChatKokkos

# %% [markdown]
# These are the steps taken to fine-tune ChatKokkos. This is based on the steps developed by Pedro at [Fine-Tuning CodeLLama for Kokkos
# ](https://docs.google.com/document/d/1u_r9PKUYYV_n5vte4oHDeZiPjUa_hnCS-pqdoB8YmF4/edit?tab=t.0) and on the [Hugging Face PEFT Adaptor Training Guide](https://huggingface.co/docs/transformers/en/peft).

# %%
# Save package state
# !pip freeze > requirements-lock.txt

# %% [markdown]
# ## Load Libraries

# %%
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
from datetime import datetime

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# %% [markdown]
# ## Load Dataset

# %%
from datasets import load_dataset

# data_files = "/auto/projects/ChatHPC/datasets/ornl/kokkos-data/kokkos_create_context.json"
# data_files = "/auto/projects/ChatHPC/.zfs/snapshot/zrepl_20241218_105321_000/datasets/ornl/kokkos-data/kokkos_create_context.json"
# data_files = "/home/7ry/Data/ellora/kokkos-data-2024-12-18/kokkos_create_context.json"
data_files = "/home/7ry/Data/ellora/kokkos-data/kokkos_create_context.json"

train_dataset = load_dataset(
    "json", data_files=data_files, split="train"
)
eval_dataset = load_dataset(
    "json", data_files=data_files, split="train"
)

# %% [markdown]
# ## Load Model

# %%
# Load model directly
from transformers import BitsAndBytesConfig

# base_model_path = "meta-llama/CodeLlama-7b-hf"
# base_model_path = "codellama/CodeLlama-7b-hf"
# base_model_path = "/home/7ry/Data/ellora/models/meta-llama/CodeLlama-7b-hf"
base_model_path = "/auto/projects/ChatHPC/models/cache/meta-llama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    # device_map={'':torch.cuda.current_device()}
)

# %% [markdown]
# ## Test base model

# %%
# eval_prompt = """You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

# You must output the Kokkos question that answers the question.
# ### Input:
# Which kind of Kokkos views are?

# ### Context:
# Introduction to Kokkos programming model

# ### Response:
# """

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# model.eval()
# with torch.no_grad():
#     output = model.generate(**model_input, max_new_tokens=700)[0]
#     stop = tokenizer.eos_token_id
#     if stop in output:
#         print("stop found")
#     print(tokenizer.decode(output))

# %%
# eval_prompt = """You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

# You must output the Kokkos question that answers the question.
# ### Input:
# Which compilers can I use to compile Kokkos codes?

# ### Context:
# Kokkos installation

# ### Response:
# """
# # {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0]))

# %%
# eval_prompt = """You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

# You must output the Kokkos question that answers the question.
# ### Input:
# Can you give me an example of Kokkos parallel_reduce?

# ### Context:
# Introduction to Kokkos programming model

# ### Response:
# """

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=400)[0], skip_special_tokens=True))

# %% [markdown]
# ## Tokenization

# %%
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

def tokenize(prompt):
    result = tokenizer(
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
    full_prompt = f"""You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

You must output the Kokkos question that answers the question.


### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)


tokenizer.add_eos_token = True

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

tokenizer.add_eos_token = False

# %% [markdown]
# ## Setup Lora and training arguments

# %%
from pytz import timezone

peft_config = LoraConfig(
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
model.train()  # put model back into training mode
# model = prepare_model_for_int8_training(model)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
# model.add_adapter(peft_config)
model.print_trainable_parameters()

# self.model = DataParallel(self.model)

batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "kokkos-code-llama"

# resume_from_checkpoint = os.path.join(base_model_path, "pytorch_model-00001-of-00003.bin")

# if resume_from_checkpoint:
#     if os.path.exists(resume_from_checkpoint):
#         print(f"Restarting from {resume_from_checkpoint}")
#         adapters_weights = torch.load(resume_from_checkpoint)
#         set_peft_model_state_dict(model, adapters_weights)
#     else:
#         print(f"Checkpoint {resume_from_checkpoint} not found")


wandb_project = "ChatKokkos"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    print("multiple gpus detected!")
    model.is_parallelizable = True
    model.model_parallel = True

training_args = TrainingArguments(
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
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

model.config.use_cache = False

# old_state_dict = model.state_dit
# model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
#     model, type(model)
# )

if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)

# model.to('cuda')

# %% [markdown]
# ## Train

# %%
trainer.train()

# %% [markdown]
# ## Save Results

# %%
save_dir = "./peft_adapter"
save_dir_tokenize = "./tokenizer"
save_dir_embedding_layers = "./embedding_layers"
tokenizer.save_pretrained(save_dir_tokenize)
trainer.model.save_pretrained(save_dir)


# %% [markdown]
# ## Load back trained model

# %%
# Load model directly
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# base_model_path = "meta-llama/CodeLlama-7b-hf"
# base_model_path = "codellama/CodeLlama-7b-hf"
# base_model_path = "/home/7ry/Data/ellora/models/meta-llama/CodeLlama-7b-hf"
base_model_path = "/auto/projects/ChatHPC/models/cache/meta-llama/CodeLlama-7b-hf"
save_dir = "./peft_adapter"
save_dir_tokenize = "./tokenizer"
save_dir_embedding_layers = "./embedding_layers"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=False,
    torch_dtype=torch.float,
    device_map="auto",
    # use_safe_serialization=False
    # device_map={'':torch.cuda.current_device()}
)

model = PeftModel.from_pretrained(model, save_dir)

model = model.merge_and_unload()
model.save_pretrained("merged_adapters")
tokenizer.save_pretrained("merged_adapters")

# model.to("cuda");

# %% [markdown]
# ## Evaluate Trained Model

# %%
eval_prompt = """You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

You must output the Kokkos question that answers the question.
### Input:
Which kind of Kokkos views are?

### Context:
Introduction to Kokkos programming model

### Response:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0]))

# %%
eval_prompt = """You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

You must output the Kokkos question that answers the question.
### Input:
Which compilers can I use to compile Kokkos codes?

### Context:
Kokkos installation

### Response:
"""
# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=500)[0]))

# %%
eval_prompt = """You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.

You must output the Kokkos question that answers the question.
### Input:
Can you give me an example of Kokkos parallel_reduce?

### Context:
Introduction to Kokkos programming model

### Response:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=700)[0]))

# %% [markdown]
# Exit kernel to free up resources when done running.

# %%
import sys
sys.exit()


