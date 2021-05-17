#!/usr/bin/env bash

conda --version

conda init bash
source ~/.bashrc

# conda create --name stes python=3.8
# conda activate stes

conda install anaconda-client
conda install conda-build
conda install -c conda-forge pytorch
conda install numpy==1.19.5
conda install pandas
conda install -c anaconda ipykernel
conda install -c conda-forge/label/cf202003 ciso8601
conda install pip
conda init bash
conda install sktime
pip install influxdb-client

# python -m ipykernel install --user --name=stes

chmod +x /app/run.sh
ls -lah /app