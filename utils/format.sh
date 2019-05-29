#!/bin/bash

SRC_FOLDER=src/image

if [ -d ".miniconda/envs/autodl" ]; then
    source .miniconda/bin/activate
    python -m black --line-length 90 --target-version py35 $SRC_FOLDER

    echo "Sorting imports"
    isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=90 -rc $SRC_FOLDER

    echo ""
    echo "Unused imports:"
    importchecker $SRC_FOLDER
else
   echo "Unsucessfull: No environment setup."
fi
