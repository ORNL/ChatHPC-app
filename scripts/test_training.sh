#!/usr/bin/env bash
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -x

source $GIT_ROOT/.venv/bin/activate

pip install papermill

pushd $GIT_ROOT/examples

for f in $GIT_ROOT/examples/*.ipynb
do
    echo papermill "$f" "${f%.*}_output.ipynb"
    papermill "$f" "${f%.*}_output.ipynb"
    echo jupyter nbconvert --to html "${f%.*}_output.ipynb"
    jupyter nbconvert --to html "${f%.*}_output.ipynb"
    $GIT_ROOT/scripts/extract_responses.py "${f%.*}_output.ipynb" > "${f%.*}_output.txt"
done

CHATKOKKOS_FINETUNED_MODEL_PATH="./app/peft_adapter" CHATKOKKOS_MERGED_MODEL_PATH="./app/merged_adapters" CHATKOKKOS_TRAINING_OUTPUT_DIR="./app/kokkos-code-llama" chatkokkos train

popd
