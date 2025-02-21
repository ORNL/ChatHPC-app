#!/usr/bin/env python3

"""Uses the output from scripts/test_training.sh to verify the the models trained properly."""

import argparse
import contextlib
import os
import subprocess
import sys
import traceback
from subprocess import check_output

from datastore.datastore import read_or_new_json
from tqdm import tqdm

GIT_ROOT = check_output("git rev-parse --show-toplevel", shell=True).decode().strip()  # noqa S602
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def run(command, verbose=True, noop=False, directory=None):
    """Print command then run command"""
    return_val = ""

    if directory is not None:
        with pushd(directory):
            return run(command, verbose, noop)

    if verbose:
        print(command)
    if not noop:
        try:
            return_val = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE).decode()  # noqa: S602
        except subprocess.CalledProcessError as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}\n\n{e.returncode}\n\n{e.stdout.decode()}\n\n{e.stderr.decode()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise
        except Exception as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise
        if verbose and return_val:
            print(return_val)

    return return_val


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    import os
    import subprocess

    pipe = subprocess.Popen(f"bash -c 'source {script} > /dev/null; env'", stdout=subprocess.PIPE, shell=True)  # noqa: S602
    output = pipe.communicate()[0].decode()
    env = dict(line.split("=", 1) for line in output.splitlines())
    os.environ.update(env)


def run_notebook():
    from chathpc.app import App as ChatApp

    experiment = "jupyter"
    os.environ["CHATHPC_FINETUNED_MODEL_PATH"] = "./peft_adapter"
    os.environ["CHATHPC_MERGED_MODEL_PATH"] = "./merged_adapters"
    os.environ["CHATHPC_TRAINING_OUTPUT_DIR"] = "./kokkos-code-llama"
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            finetune.append(datapoint)
        return finetune

    finetune = read_or_new_json(f"{experiment}_finetune_out", get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            merged.append(datapoint)
        return merged

    merged = read_or_new_json(f"{experiment}_merged_out", get_merged)

    return (finetune, merged)


def run_notebook_app():
    from chathpc.app import App as ChatApp

    experiment = "jupyter_app"
    os.environ["CHATHPC_FINETUNED_MODEL_PATH"] = "./jupyter_app/peft_adapter"
    os.environ["CHATHPC_MERGED_MODEL_PATH"] = "./jupyter_app/merged_adapters"
    os.environ["CHATHPC_TRAINING_OUTPUT_DIR"] = "./jupyter_app/kokkos-code-llama"
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            finetune.append(datapoint)
        return finetune

    finetune = read_or_new_json(f"{experiment}_finetune_out", get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            merged.append(datapoint)
        return merged

    merged = read_or_new_json(f"{experiment}_merged_out", get_merged)

    return (finetune, merged)


def run_app():
    from chathpc.app import App as ChatApp

    experiment = "app"
    chat_app = ChatApp.from_json(
        {
            "prompt_template": "You are a powerful LLM model for Kokkos called ChatKokkos created by ORNL. Your job is to answer questions about the Kokkos programming model. You are given a question and context regarding the Kokkos programming model.\n\nYou must output the answer the question.\n\n### Context:\n{{ context }}\n\n### Question:\n{{ question }}\n\n### Answer:\n{{ answer }}\n\n",
            "finetuned_model_path": "./app/peft_adapter",
            "merged_model_path": "./app/merged_adapters",
            "training_output_dir": "./app/kokkos-code-llama",
        }
    )

    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            finetune.append(datapoint)
        return finetune

    finetune = read_or_new_json(f"{experiment}_finetune_out", get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            merged.append(datapoint)
        return merged

    merged = read_or_new_json(f"{experiment}_merged_out", get_merged)

    return (finetune, merged)


def run_app_old():
    from chathpc.app import App as ChatApp

    experiment = "app_old"
    chat_app = ChatApp.from_json(
        {
            "prompt_template": "You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{{ question }}\n\n### Context:\n{{ context }}\n\n### Response:\n{{ answer }}\n",
            "finetuned_model_path": "./app_old/peft_adapter",
            "merged_model_path": "./app_old/merged_adapters",
            "training_output_dir": "./app_old/kokkos-code-llama",
        }
    )

    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            finetune.append(datapoint)
        return finetune

    finetune = read_or_new_json(f"{experiment}_finetune_out", get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chat_evaluate_extract(**item)
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response,
            }
            merged.append(datapoint)
        return merged

    merged = read_or_new_json(f"{experiment}_merged_out", get_merged)

    return (finetune, merged)


def ignore_minor(string: str):
    s = string.strip()
    line = s.splitlines()
    line = [x.strip() for x in line]
    return "\n".join(line)


def run_ollama():
    from ollama import GenerateResponse, generate

    from chathpc.app import App as ChatApp

    experiment = "ollama"
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_ol():
        ol = []
        for item in tqdm(chat_app.train_dataset, "Run ol"):
            prompt = chat_app.chat_prompt(**item)
            response: GenerateResponse = generate(model="ChatKokkos", prompt=prompt, options={"temperature": 0.0})
            datapoint = {
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "response": response.response,
            }
            ol.append(datapoint)
        return ol

    return read_or_new_json(f"{experiment}_ol_out", get_ol)


def verify_ollama(expected_reponse_errors=0):
    (finetuned, merged) = run_app()
    ol = run_ollama()

    response_errors = 0
    ol_errors = 0

    for i, (fine, o) in tqdm(enumerate(zip(merged, ol)), "Compare"):  # type: ignore
        if fine["answer"] != o["answer"]:
            print("Error: answer mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['answer']}")
            print(f"Ollama:\n{o['answer']}")
            print("**********************************************************")
            print()
            raise RuntimeError("Answer Mismatch")
        if ignore_minor(o["answer"]) != ignore_minor(o["response"]):
            response_errors += 1
            # print("Error: response mismatch")
            # print(f"Sample {i}")
            # print(f"Answer:\n{o['answer']}")
            # print(f"Response:\n{o['response']}")
            # print(f"**********************************************************")
            # print()
        if ignore_minor(fine["response"]) != ignore_minor(o["response"]):
            ol_errors += 1
            print("Error: ollama mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['response']}")
            print(f"Ollama:\n{o['response']}")
            print("**********************************************************")
            print()

    if response_errors != expected_reponse_errors:
        print(f"Error: Response Errors do not match expected: {response_errors} != {expected_reponse_errors}")
        ol_errors += 1

    print(f"Ollama Errors: {ol_errors}")
    return ol_errors


def verify_app(runner):
    (finetuned, merged) = runner()

    response_errors = 0
    merge_errors = 0

    for i, (fine, merge) in tqdm(enumerate(zip(finetuned, merged)), "Compare"):
        if fine["answer"] != merge["answer"]:
            print("Error: answer mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['answer']}")
            print(f"Merged:\n{merge['answer']}")
            print("**********************************************************")
            print()
            raise RuntimeError("Answer Mismatch")
        if ignore_minor(fine["answer"]) != ignore_minor(fine["response"]):
            response_errors += 1
            print("Error: response mismatch")
            print(f"Sample {i}")
            print(f"Answer:\n{fine['answer']}")
            print(f"Response:\n{fine['response']}")
            print("**********************************************************")
            print()
        if ignore_minor(fine["response"]) != ignore_minor(merge["response"]):
            merge_errors += 1
            print("Error: merge mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['response']}")
            print(f"Merged:\n{merge['response']}")
            print("**********************************************************")
            print()

    print(f"Response Errors: {response_errors}, Merge Errors: {merge_errors}")
    return response_errors, merge_errors


def init_parser(parser):
    # parser.add_argument('-d', '--dir', type=str, default=OUTPUT_DIR)
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")
    parser.add_argument("files", metavar="p", type=str, nargs="*")


def main(raw_args=None):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Extract example output from jupyter notebook.""")
    init_parser(parser)
    args = parser.parse_args(raw_args)

    if args.debug:
        import debugpy  # noqa: T100

        debugpy.listen(5678)  # noqa: T100
        print("Attach debugger to continue.")
        debugpy.wait_for_client()  # noqa: T100

    os.environ["CHATHPC_DATA_FILE"] = "/home/7ry/Data/ellora/ChatKokkos-data/kokkos_dataset_before_reinforcement.json"
    # os.environ["CHATHPC_FINETUNED_MODEL_PATH"] = "./peft_adapter"
    # os.environ["CHATHPC_MERGED_MODEL_PATH"] = "./merged_adapters"
    # os.environ["CHATHPC_TRAINING_OUTPUT_DIR"] = "./kokkos-code-llama"

    # print("** Running Notebook **")
    # print("** Running Notebook **", file=sys.stderr)
    # notebook_errors = verify_app(run_notebook)
    # print("Response Errors: {}, Merge Errors: {}".format(*notebook_errors))
    # print("Response Errors: {}, Merge Errors: {}".format(*notebook_errors), file=sys.stderr)

    # print("\n\n** Running Notebook App **")
    # print("\n\n** Running Notebook App **", file=sys.stderr)
    # notebook_app_errors = verify_app(run_notebook_app)
    # print("Response Errors: {}, Merge Errors: {}".format(*notebook_app_errors))
    # print("Response Errors: {}, Merge Errors: {}".format(*notebook_app_errors), file=sys.stderr)

    print("\n\n** Running App **")
    print("\n\n** Running App **", file=sys.stderr)
    app_errors = verify_app(run_app)
    print("Response Errors: {}, Merge Errors: {}".format(*app_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*app_errors), file=sys.stderr)

    print("\n\n** Running App Old **")
    print("\n\n** Running App Old **", file=sys.stderr)
    app_old_errors = verify_app(run_app_old)
    print("Response Errors: {}, Merge Errors: {}".format(*app_old_errors))
    print("Response Errors: {}, Merge Errors: {}".format(*app_old_errors), file=sys.stderr)

    # print("\n\n** Running Ollama **")
    # print("\n\n** Running Ollama **", file=sys.stderr)
    # ol_errors = verify_ollama(expected_reponse_errors=app_errors[0])
    # print(f"Ollama Errors: {ol_errors}")
    # print(f"Ollama Errors: {ol_errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
