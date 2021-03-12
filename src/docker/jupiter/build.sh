#!/usr/bin/env bash

echo "Installing Miniconda3, if you do not want this to happen, modify the file."
read -r

wget 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.shy'

bash ./Miniconda3-latest-Linux-x86_64.shy

conda --version

conda init bash
source ~/.bashrc

# conda create --name stes python=3.8
# conda activate stes

conda install anaconda-client
conda install conda-build
conda install -c conda-forge pytorch
conda install numpy
conda install pandas
conda install -c anaconda ipykernel

# python -m ipykernel install --user --name=stes

chmod +x /app/run.sh
ls -lah /app