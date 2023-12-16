import torch
import sys
sys.path.append("/home/codee/scratch/")
import matplotlib as plt
from sklearn.cluster import KMeans
from model import *
from utils import get_plugin, read_yaml
import numpy as np
from PIL import Image, ImageSequence
from torchvision.transforms import ToTensor
#from infer_engine import run_inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")