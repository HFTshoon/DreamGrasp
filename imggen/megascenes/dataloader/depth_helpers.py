import cv2
import numpy as np
import random
import os, sys 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
import re
import math
from datetime import datetime
import pytz
from torch.utils.data import Dataset
from PIL import Image
import warnings

from torchvision.transforms import Compose

import ipdb

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

def invert_depth(depth_map):
    inv = depth_map.clone()
    # disparity_max = 1000
    disparity_min = 0.001
    # inv[inv > disparity_max] = disparity_max
    inv[inv < disparity_min] = disparity_min
    inv = 1.0 / inv
    return inv

    