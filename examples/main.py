
# -*- coding: utf-8 -*-


import sys
from cmbfscnn.utils import *
from cmbfscnn.CMBFS import CMBFSCNN
try:
    config_dir = sys.argv[1]
except:
    print("Please enter the configuration file")
    sys.exit(1)

config = load_pkl(config_dir)
cmbfcnn = CMBFSCNN(config)
cmbfcnn.run_CMBFSCNN()