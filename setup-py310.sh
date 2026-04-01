#!/bin/bash
set -e

conda create -n py310 python=3.10 -y
source activate py310
pip install -q ipykernel
python -m ipykernel install --name py310 --display-name "Python 3.10" --prefix=/opt/conda
pip install pickleshare ipywidgets