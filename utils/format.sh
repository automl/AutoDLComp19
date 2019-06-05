if [ -f ".miniconda/bin/activate" ]; then
    # Use local miniconda installation
    source .miniconda/bin/activate
fi
conda activate autodl
pre-commit run --all-files
