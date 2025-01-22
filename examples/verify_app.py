#!/usr/bin/env python3

"""Uses the output from scripts/test_training.sh to verify the the models trained properly."""

import argparse
import sys
import contextlib
import os
import datetime
import subprocess
import time
import functools
import random
import re
import json
import traceback
from multiprocessing import Pool
from subprocess import check_output
from tqdm import tqdm
from datastore.datastore import read_or_new_json

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

def extract_answer(response: str):
    matchstr = '### Response:\n'
    index = response.find(matchstr)
    if index == -1:
        return None
    return response[index + len(matchstr):]\
        .replace('<s>', '')\
        .replace('</s>', '')


def run_notebook():
    from chatkokkos.app import App as ChatApp
    experiment = 'jupyter'
    os.environ["CHATKOKKOS_FINETUNED_MODEL_PATH"] = "./peft_adapter"
    os.environ["CHATKOKKOS_MERGED_MODEL_PATH"] = "./merged_adapters"
    os.environ["CHATKOKKOS_TRAINING_OUTPUT_DIR"] = "./kokkos-code-llama"
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chatkokkos_evaluate(item['question'], item['context'])
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                "response": extract_answer(response),
            }
            finetune.append(datapoint)
        return finetune
    finetune = read_or_new_json(f'{experiment}_finetune_out', get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chatkokkos_evaluate(item['question'], item['context'])
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                "response": extract_answer(response),
            }
            merged.append(datapoint)
        return merged
    merged = read_or_new_json(f'{experiment}_merged_out', get_merged)

    return (finetune, merged)

def run_notebook_app():
    from chatkokkos.app import App as ChatApp
    experiment = 'jupyter_app'
    os.environ["CHATKOKKOS_FINETUNED_MODEL_PATH"] = "./jupyter_app/peft_adapter"
    os.environ["CHATKOKKOS_MERGED_MODEL_PATH"] = "./jupyter_app/merged_adapters"
    os.environ["CHATKOKKOS_TRAINING_OUTPUT_DIR"] = "./jupyter_app/kokkos-code-llama"
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chatkokkos_evaluate(item['question'], item['context'])
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                "response": extract_answer(response),
            }
            finetune.append(datapoint)
        return finetune
    finetune = read_or_new_json(f'{experiment}_finetune_out', get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chatkokkos_evaluate(item['question'], item['context'])
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                "response": extract_answer(response),
            }
            merged.append(datapoint)
        return merged
    merged = read_or_new_json(f'{experiment}_merged_out', get_merged)

    return (finetune, merged)

def run_app():
    from chatkokkos.app import App as ChatApp
    experiment = 'app'
    os.environ["CHATKOKKOS_FINETUNED_MODEL_PATH"] = "./app/peft_adapter"
    os.environ["CHATKOKKOS_MERGED_MODEL_PATH"] = "./app/merged_adapters"
    os.environ["CHATKOKKOS_TRAINING_OUTPUT_DIR"] = "./app/kokkos-code-llama"
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_finetuned():
        chat_app.load_finetuned_model()
        finetune = []
        for item in tqdm(chat_app.train_dataset, "Run Finetune"):
            response = chat_app.chatkokkos_evaluate(item['question'], item['context'])
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                "response": extract_answer(response),
            }
            finetune.append(datapoint)
        return finetune
    finetune = read_or_new_json(f'{experiment}_finetune_out', get_finetuned)

    def get_merged():
        chat_app.load_merged_model()
        merged = []
        for item in tqdm(chat_app.train_dataset, "Run Merged"):
            response = chat_app.chatkokkos_evaluate(item['question'], item['context'])
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                "response": extract_answer(response),
            }
            merged.append(datapoint)
        return merged
    merged = read_or_new_json(f'{experiment}_merged_out', get_merged)

    return (finetune, merged)

def ignore_minor(string:str):
    s = string.strip()
    l = s.splitlines()
    l = [x.strip() for x in l]
    s = "\n".join(l)
    return s

def run_ollama():
    from ollama import GenerateResponse, generate

    from chatkokkos.app import App as ChatApp
    experiment = 'ollama'
    chat_app = ChatApp()
    chat_app.load_datasets()

    def get_ol():
        ol = []
        for item in tqdm(chat_app.train_dataset, "Run ol"):
            prompt = chat_app.chatkokkos_prompt(item['question'], item['context'])
            response: GenerateResponse = generate(model="ChatKokkos", prompt=prompt, options={"temperature": 0.0})
            datapoint = {
                "question": item['question'],
                "context": item['context'],
                "answer": item['answer'],
                # "response": extract_answer(response.response),
                "response": response.response,
            }
            ol.append(datapoint)
        return ol
    ol = read_or_new_json(f'{experiment}_ol_out', get_ol)

    return ol


def verify_ollama():
    (finetuned, merged) = run_app()
    ol = run_ollama()

    ol_errors = 0

    for i, (fine, o) in tqdm(enumerate(zip(merged, ol)), "Compare"):
        if fine['answer'] != o['answer']:
            print("Error: answer mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['answer']}")
            print(f"Ollama:\n{o['answer']}")
            print(f"**********************************************************")
            print()
            assert False, "Answer Mismatch"
        if ignore_minor(fine['response']) != ignore_minor(o['response']):
            ol_errors += 1
            print("Error: ollama mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['response']}")
            print(f"Ollama:\n{o['response']}")
            print(f"**********************************************************")
            print()

    print(f'Ollama Errors: {ol_errors}')
    return ol_errors


def verify_app(runner):
    (finetuned, merged) = runner()

    response_errors = 0
    merge_errors = 0

    for i, (fine, merge) in tqdm(enumerate(zip(finetuned, merged)), "Compare"):
        if fine['answer'] != merge['answer']:
            print("Error: answer mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['answer']}")
            print(f"Merged:\n{merge['answer']}")
            print(f"**********************************************************")
            print()
            assert False, "Answer Mismatch"
        if ignore_minor(fine['answer']) != ignore_minor(fine['response']):
            response_errors += 1
            print("Error: response mismatch")
            print(f"Sample {i}")
            print(f"Answer:\n{fine['answer']}")
            print(f"Response:\n{fine['response']}")
            print(f"**********************************************************")
            print()
        if ignore_minor(fine['response']) != ignore_minor(merge['response']):
            merge_errors += 1
            print("Error: merge mismatch")
            print(f"Sample {i}")
            print(f"Finetuned:\n{fine['response']}")
            print(f"Merged:\n{merge['response']}")
            print(f"**********************************************************")
            print()

    print(f'Response Errors: {response_errors}, Merge Errors: {merge_errors}')
    return response_errors, merge_errors


def init_parser(parser):
    # parser.add_argument('-d', '--dir', type=str, default=OUTPUT_DIR)
    parser.add_argument("--debug", action="store_true", help="Open debug port (5678).")
    parser.add_argument('files', metavar='p', type=str, nargs='*')


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

    print("** Running Notebook **")
    print("** Running Notebook **", file=sys.stderr)
    notebook_errors = verify_app(run_notebook)
    print('Response Errors: {}, Merge Errors: {}'.format(*notebook_errors))
    print('Response Errors: {}, Merge Errors: {}'.format(*notebook_errors), file=sys.stderr)

    print("\n\n** Running Notebook App **")
    print("\n\n** Running Notebook App **", file=sys.stderr)
    notebook_app_errors = verify_app(run_notebook_app)
    print('Response Errors: {}, Merge Errors: {}'.format(*notebook_app_errors))
    print('Response Errors: {}, Merge Errors: {}'.format(*notebook_app_errors), file=sys.stderr)

    print("\n\n** Running App **")
    print("\n\n** Running App **", file=sys.stderr)
    app_errors = verify_app(run_app)
    print('Response Errors: {}, Merge Errors: {}'.format(*app_errors))
    print('Response Errors: {}, Merge Errors: {}'.format(*app_errors), file=sys.stderr)

    print("\n\n** Running Ollama **")
    print("\n\n** Running Ollama **", file=sys.stderr)
    ol_errors = verify_ollama()
    print(f'Ollama Errors: {ol_errors}')
    print(f'Ollama Errors: {ol_errors}', file=sys.stderr)

if __name__ == "__main__":
    main()

