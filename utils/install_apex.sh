if [ -f ".miniconda/bin/activate" ]; then
    # Use local miniconda installation
    source .miniconda/bin/activate
fi
eval "$(conda shell.bash hook)"
conda activate autodl
git clone https://github.com/NVIDIA/apex
if [ -z "$(which nvcc)" ]; then
    srun -p meta_gpu-x pip install --no-cache-dir --global-option="--cuda_ext" --global-option="--cpp_ext" ./apex
else
    pip install --no-cache-dir --global-option="--cuda_ext" --global-option="--cpp_ext" ./apex
fi
rm -rf apex