#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:${PWD}"
export DATA_PATH="${PWD}/data"

(. .venv/bin/activate \
    && pushd ${1} &> /dev/null \
    && shift \
    && python3 ${@} \
    && popd &> /dev/null)
