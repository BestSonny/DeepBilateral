# --------------------------------------------------------
# Copyright (c) 2017 Pan He
# Licensed under The MIT License [see LICENSE for details]
# Written by Pan He
# --------------------------------------------------------

"""Set up paths for Deep Bilateral."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

add_path(this_dir)
# Add bilateral to PYTHONPATH
bilateral_path = osp.join(this_dir, '..', 'libs', 'bilateral')
add_path(bilateral_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'libs')
add_path(lib_path)

# Add models to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'models')
add_path(lib_path)


# Add datasets to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'datasets')
add_path(lib_path)
