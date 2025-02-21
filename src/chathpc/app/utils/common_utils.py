from __future__ import annotations

import json


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


def extract_answer(response: str, prompt: str, stop: str | None = None):
    answer = response
    answer = answer.replace("<s> ", "").replace("</s>", "")

    if answer.startswith(prompt):
        answer = answer[len(prompt) :]

    if stop is not None and answer.endswith(stop):
        answer = answer[: -len(stop)]

    return answer
