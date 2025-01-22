#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -x

source $GIT_ROOT/.venv/bin/activate

pip install papermill

for f in $GIT_ROOT/examples/*.ipynb
do
    echo papermill "$f" "${f%.*}_output.ipynb"
    papermill "$f" "${f%.*}_output.ipynb"
    echo jupyter nbconvert --to html "${f%.*}_output.ipynb"
    jupyter nbconvert --to html "${f%.*}_output.ipynb"
done

chatkokkos train
