from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import traceback


def load_json_arg(str_or_fn):
    """Load and parse JSON data from either a string or file.

    Args:
        str_or_fn (str): Either a JSON string starting with '{' or a path to a JSON file.
            If None, returns an empty dict.

    Returns:
        dict: Parsed JSON data as a dictionary. Returns empty dict if input is None.

    Examples:
        >>> load_json_arg('{"key": "value"}')
        {'key': 'value'}
        >>> load_json_arg("path/to/file.json")
        {'contents': 'from file'}
        >>> load_json_arg(None)
        {}
    """
    if str_or_fn is None:
        return {}
    if isinstance(str_or_fn, dict):
        return str_or_fn
    if isinstance(str_or_fn, str) and str_or_fn[0] == "{":
        params = json.loads(str_or_fn)
    else:
        with open(str_or_fn) as f:
            params = json.loads(f.read())
            params["filename"] = str_or_fn
            f.close()
    return params


def evaluate_fstring(fstring, **kwargs):
    """Evaluate a string as an f-string with provided keyword arguments.

    Args:
        fstring (str): The string to be evaluated as an f-string. Can contain
            Python expressions inside curly braces {}.
        **kwargs: Keyword arguments that will be used to format the f-string.

    Returns:
        str: The evaluated f-string with all expressions replaced with their values.

    Examples:
        >>> evaluate_fstring("Hello {name}!", name="World")
        'Hello World!'
        >>> evaluate_fstring("The sum is {x + y}", x=1, y=2)
        'The sum is 3'
    """
    return eval(f"f'''{fstring}'''", {}, kwargs)  # noqa: S307


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
