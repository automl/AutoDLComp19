#!/bin/bash

if [ -d ".miniconda/envs/autodl" ]; then
    source .miniconda/bin/activate autodl
    python -m black --line-length 90 --target-version py35 src

    echo "Sorting imports"
    isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=90 -rc src

    echo ""
    echo "Unused imports:"
    importchecker src
else
   echo "Unsucessfull: No environment setup."
fi
