#!/bin/bash
set -e

conda create -n py310 python=3.10 ipykernel pickleshare ipywidgets -y
conda run -n py310 python -m ipykernel install --name py310 --display-name "Python 3.10" --prefix=/opt/conda