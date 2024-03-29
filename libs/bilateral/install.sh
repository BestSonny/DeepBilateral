#!/bin/bash

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

nvcc -c -o src/bilateral_slice_launcher.cu.o src/bilateral_slice_launcher.cu  --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python build.py
