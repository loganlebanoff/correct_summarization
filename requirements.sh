#!/usr/bin/env bash

set -x

wget "https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh"
chmod +x Anaconda3-2018.12-Linux-x86_64.sh
./Anaconda3-2018.12-Linux-x86_64.sh
source ~/.bashrc

conda install numpy -y
conda install cudatoolkit==9.0 -y
conda install tensorflow-gpu==1.11 -y
pip install sumy -y
pip install pyrouge -y
conda install spacy -y
python -m spacy download en_core_web_sm