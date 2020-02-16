from __future__ import print_function,division
import torch
import os
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()








#must before the plt.show()
plt.ioff()# this function shows the according image
plt.show()


