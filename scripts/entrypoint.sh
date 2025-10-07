#!/usr/bin/env bash
set -euo pipefail

# default to interactive bash if no args
if [ $# -eq 0 ]; then
  exec bash
fi

# forward the command into python -m or shell
exec "$@"
