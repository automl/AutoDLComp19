if [ -f ".miniconda/bin/activate" ]; then
    # Use local miniconda installation
    source .miniconda/bin/activate
fi
eval "$(conda shell.bash hook)"
conda activate autodl
python src/image/download_pretrained_models.py
