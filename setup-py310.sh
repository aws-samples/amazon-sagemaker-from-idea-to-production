#!/bin/bash
set -e

conda create -n py310 python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate py310
pip install -q ipykernel
python -m ipykernel install --name py310 --display-name "Python 3.10" --prefix=/opt/conda
pip install pickleshare ipywidgets