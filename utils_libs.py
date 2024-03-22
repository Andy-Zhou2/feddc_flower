"""Model creation script for FedDC.

Slightly adapted from the original https://github.com/gaoliang13/FedDC/blob/main/utils_models.py
Copyright 2024 Andy Zhou

This project is licensed under the MIT License.
"""

import os
import numpy as np
from scipy import io
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import copy
from itertools import combinations
import random

