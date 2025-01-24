import json


def load_json_arg(str_or_fn):
    if str_or_fn is None:
        return {}
    if str_or_fn is str and str_or_fn[0] == "{":
        params = json.loads(str_or_fn)
    else:
        with open(str_or_fn) as f:
            params = json.loads(f.read())
            f.close()
    return params
