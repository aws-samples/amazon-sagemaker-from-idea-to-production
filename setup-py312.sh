#!/bin/bash
set -e

conda create -n py312 python=3.12 pip -y
source activate py312
pip install -q ipykernel
python -m ipykernel install --name py312 --display-name "Python 3.12" --prefix=/opt/conda
pip install pickleshare ipywidgets
