#!/bin/bash

set -a
. "${PWD}/common.env"
set +a

"${PWD}/cmd/utility/run_in_venv.sh" $@
