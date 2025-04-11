#!/bin/bash

echo "Make sure you have mounted Google Drive by executing the following code in Colab Notebook："
echo "from google.colab import drive; drive.mount('/content/drive')"
echo ""

if [ ! -d "/content/StyleTransfer" ]; then
  echo "Cloning StyleTransfer repository..."
  git clone https://github.com/JescYip/StyleTransfer.git
fi

cd /content/StyleTransfer

echo "Installing PyTorch and dependency libraries..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4
pip install configargparse kornia imageio matplotlib tqdm scikit-image plyfile opencv-python
