#!/bin/bash

set -e

# Conda setup and environment creation
eval "$(conda shell.bash hook)"

conda create --name 3dvg python=3.10 --yes
conda activate 3dvg
echo "The conda environment was successfully created"

# Install PyTorch and related libraries
# NOTE: use "nvcc -V" to find a cuda version satisfied with your systerm, change the command below following "https://pytorch.org/get-started/previous-versions/"
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

echo "Pytorch installation is complete."

# # Install common Python dependencies
pip install hydra-core omegaconf
pip install freetype-py shapely svgutils cairosvg plyfile open3d
pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
pip install triton numba
pip install numpy scipy scikit-fmm einops timm fairscale
pip install accelerate==0.33.0 transformers huggingface_hub==0.24.2 safetensors datasets
pip install easydict scikit-learn webdataset
pip install cssutils open3d
echo "The basic dependency library is installed."

# Additional utility libraries
pip install ftfy regex tqdm
pip install git+https://github.com/jonbarron/robust_loss_pytorch
pip install git+https://github.com/openai/CLIP.git
echo "Additional utility installation is complete."

# Install diffusers
pip install diffusers==0.20.2
echo "Diffusers installation is complete. version: 0.20.2"

# Install xformers (should match torch version, eg. torch 1.13.1 - xformers 0.0.16)
pip install xformers==0.0.16

# Clone and set up DiffVG, handling dependencies on Ubuntu
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive

# for Ampere sm_80
sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode=arch=compute_80,code=sm_80")/' CMakeLists.txt

# Install system dependencies for Ubuntu (to avoid potential issues)
echo "Installing system dependencies for DiffVG..."
sudo apt update
sudo apt install -y cmake ffmpeg build-essential libjpeg-dev libpng-dev libtiff-dev

pip install svgwrite svgpathtools cssutils torch-tools

# Install DiffVG
python setup.py install
echo "DiffVG installation is complete."
cd ..

# 3DGS
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/

# point-e
pip install git+https://github.com/openai/point-e.git

# Final confirmation
echo "The running environment has been successfully installed!!!"