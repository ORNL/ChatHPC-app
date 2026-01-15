# ChatHPC Application

The ChatHPC Application is part of the larger ChatHPC ecosystem and Ellora Project. See the [ChatHPC Homepage](https://ornl.github.io/ChatHPC/).

## How to create a new ChatHPC/ChatX application repo

These steps go over how to create a new ChatX application where ChatX is a place holder for your application name. Replace X with your custom name. For an example ChatX application, take a look at [ChatHPC for Kokkos](https://code.ornl.gov/ChatHPC/ChatHPCforKokkos) or [ChatASMKernel](https://code.ornl.gov/ChatHPC/ChatASMKernel).

!!! caution
    Due to trademark and licensing requirements. It is now recommended and potentially required to avoid using the "ChatX" convention. The recommendation now is to use the convention "ChatHPC for X". See [ChatHPC for Kokkos](https://code.ornl.gov/ChatHPC/ChatHPCforKokkos) as an example of the latest recommended convention.

1. Create new repostories to hold custom app and dataset files.
    1. Create a training data repository for ChatX with the naming convention "ChatX Data" and slug "ChatX-data.
    2. Create a script and configuration repositry for ChatX with the nameing convention "ChatX" and slug "ChatX". If this application will create its own python library or programs, fork the [Python Project Template](https://code.ornl.gov/7ry/python-project-template/-/forks/new), otherwise start with a blank project and add the minimal files covered below.
2. ChatX Data Repo Setup: Add the fine-tunning, verification, and testings datasets to the ChatX Data repository with a README.md containing information and meta-data about the dataset. See [ChatHPC for Kokkos Data](https://code.ornl.gov/ChatHPC/ChatHPCforKokkos-data) for an example.
3. ChatX Repo Setup: Add the minimal components to the ChatX repository. See [Minimal Components for ChatX Application Repo](#minimal-components-for-chatx-application-repo) below.
    1. `requirements.txt`
    2. `README.md`
    3. `config.json`
    4. `prompt_template.txt`
    5. Modelfile (optional).
    6. Training scripts (optional).
    7. Evaluation scripts (optional).

### Minimal Components for ChatX Application Repo.

`requirements.txt`:
```txt
chathpc-app @ git+ssh://git@github.com/ORNL/ChatHPC-app.git@main
```
Used to install the Python pip dependences. The only required package is the [ChatHPC-app](https://github.com/ORNL/ChatHPC-app) which pulls in all the required LLM packages. To pin a specific version replace `main` with a tagged version.

`README.md`:
```markdown
# ChatX

## Quick Start

Very quick start, follow:

    python3 -m venv --upgrade-deps --prompt $(basename $PWD) .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    chathpc --config config.json train
    chathpc --config config.json verify
    chathpc --config config.json run

## How to Export Trained Assistant to Ollama

After installing the requirments and running training, you can use the commands below to export the model to Ollama. Each of the training scripts also export the GGUF model needed for ollama uploading.

    module load ollama
    /home/7ry/Data/ellora/llama.cpp/convert_hf_to_gguf.py merged_adapters/
    ollama create <appname>
```
Include information about the custom assistant and the steps for training and validating it.

`config.json`
```json
{
    "data_file": "../ChatX-data/training-data.json",
    "base_model_path": "/auto/projects/ChatHPC/models/cache/meta-llama/CodeLlama-7b-hf",
    "prompt_template_file": "prompt_template.txt"
}
```
Used to customize the settings for ChatHPC applications. All the unique values for the app need to be set here. Only the required fields without a default must be specified, along with any of the fields which you want to change from the default. See [ChatHPC.app.AppConfig documentation](https://chathpc-app.readthedocs.io/en/latest/api/#chathpc.app.AppConfig) for details on the configuration fields.

`prompt_template.txt`:
```txt
You are a powerful LLM model for writing high-performance kernels called ChatX created by ORNL. Your job is to write kernels for the operations you are asked to. You are given a question and context regarding the application.

You must output the answer the question.
{% if context %}
### Context:
{{ context }}
{% endif %}
### Question:
{{ question }}

### Answer:
{{ answer }}


```
Used to set the prompt template for both training and inference. Use this same basic structure which fills values from the training set into the prompt, and customize the first sentance, which provides the assistant with infromation about itself. This template file is written using the [Jinja](https://jinja.palletsprojects.com/) templating language.

`Modelfile`:
```txt
FROM merged_adapters/Merged_Adapters-6.7B-F16.gguf

TEMPLATE """You are a powerful LLM model for writing high-performance kernels called ChatX created by ORNL. Your job is to write kernels for the operations you are asked to. You are given a question and context regarding the application.


You must output the answer the question.
{{ if .System }}
### Context:
{{ .System }}
{{ end }}
### Question:
{{ .Prompt }}

### Answer:
{{ .Response }}

"""

PARAMETER stop "### Context:"
PARAMETER stop "### Answer:"
PARAMETER stop "### Question:"

```
This model file is used to package the model for Ollama. See [How to Export Trained Assistant to Ollama](#how-to-export-trained-assistant-to-ollama) and [Ollama Documentation](https://github.com/ollama/ollama?tab=readme-ov-file#import-from-gguf). This template is written using the [Go template syntax](https://pkg.go.dev/text/template). It should be easy enough to copy your custom template from Jinja to Go template language by following the patterns in the example. At somepoint I might automate the creation of the Modelfile. This file also defines the location of the gguf file and additional stop parameters. These additional stop paramters help the assistant to stop replying after answering the question.

## How to Export Trained Assistant to Ollama

After completing the training of the assistant, you can use the commands below to export the model to [Ollama on ExCL](https://docs.excl.ornl.gov/quick-start-guides/ollama). Make sure you also have a `Modulefile` setup in the repo.

```bash
module load ollama
/home/7ry/Data/ellora/llama.cpp/convert_hf_to_gguf.py merged_adapters/
ollama create <appname>
```

After the assistant is uploaded to Ollama, you can also access the assistant using [Open WebUI](https://docs.excl.ornl.gov/quick-start-guides/open-webui) for a nice user interface.

Note: If you have already uploaded the assistant to Ollama, you will need to remove the prior model with `ollama rm <appname>` before you can upload it again.

## How to view the Training Data as Markdown

The [ChatHPC-json-to-md](https://github.com/ORNL/ChatHPC-app#chathpc-json-to-md) tool which is part of ChatHPC App can be used to convert intput and output JSON data files into markdown to make them easier to read. This is useful for both verifing the correct formating of the input and for reviewing and rating the reponses (output) from the LLM assistants.

## Tools Used

- [Hatch](https://hatch.pypa.io/) --- Python Build System.
- [MkDocs](https://www.mkdocs.org/) --- Documentation Generator.
    - [Material Theme](https://squidfunk.github.io/mkdocs-material/) --- Theme for documentation.
    - [mkdocstrings](https://mkdocstrings.github.io/) --- Automatic documentation generation from sources.
    - [DevDocs](https://docs.excl.ornl.gov/quick-start-guides/devdocs) --- Internal to ORNL document website hosting.
- [GitLab CI](https://docs.gitlab.com/ee/ci/) --- Continuous Integration.
    - [Example Pipeline](https://code.ornl.gov/ChatHPC/ChatHPC-app/-/pipelines)
    - [Example Pipeline Source](https://code.ornl.gov/ChatHPC/ChatHPC-app/-/blob/main/.gitlab-ci.yml?ref_type=heads)
- [Ruff](https://docs.astral.sh/ruff/) --- Python linter and code formatter.
    - [Ruff Rules](https://docs.astral.sh/ruff/rules/) --- Rules used by Ruff.
- [EditorConfig](https://editorconfig.org/) --- Maintain consistent coding styles between different editors and IDEs.
- [Markdown Lint Tool](https://github.com/markdownlint/markdownlint) --- Markdown linting tool.
- [Pre-Commit](https://pre-commit.com/) --- Git precommit hooks.
    - [Built-in Hooks](https://github.com/pre-commit/pre-commit-hooks)
    - [Ruff Pre-Commit Hooks](https://github.com/astral-sh/ruff-pre-commit)
    - [Editor Config Pre-Commit Hooks](https://github.com/editorconfig-checker/editorconfig-checker.python)
    - [Markdown Lint Pre-Commit Hooks](https://github.com/markdownlint/markdownlint)
