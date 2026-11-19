#!/bin/bash

source .venv/bin/activate;
python3 -m pip install -e ".[dev]"
python3 -c "import openpyxl; print(openpyxl.__version)"
